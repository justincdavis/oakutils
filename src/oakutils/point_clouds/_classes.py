from __future__ import annotations

import atexit
from threading import Condition, Thread
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from typing_extensions import Self


class PointCloudVisualizer:
    """A class to visualize open3d point clouds.

    Methods
    -------
    stop()
        Stops the visualizer.
    update(pcd: o3d.geometry.PointCloud)
        Updates the point cloud to visualize.
    update_rotation(R_camera_to_world: np.ndarray)
        Updates the rotation matrix to use for the point cloud.

    References
    ----------
    https://github.com/luxonis/depthai-experiments/blob/master/gen2-pointcloud/device-pointcloud/projector_device.py
    """

    def __init__(
        self: Self,
        window_name: str = "PointCloud",
        window_size: tuple[int, int] = (1920, 1080),
        use_threading: bool | None = None,
    ) -> None:
        """Creates a PointCloudVisualizer object.

        Parameters
        ----------
        window_name : str
            The name of the visualization window. Defaults to "PointCloud".
        window_size : Tuple[int, int]
            The size of the visualization window. Defaults to (1920, 1080).
        use_threading : bool
            Whether to use threading for visualization. Defaults to True.
        """
        if use_threading is None:
            use_threading = True

        self._pcd: o3d.geometry.PointCloud | None = None
        self._vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        self._R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(
            np.float64
        )
        self._window_name: str = window_name
        self._window_size: tuple[int, int] = window_size
        self._started: bool = False
        self._stopped: bool = False

        self._use_threading: bool = use_threading
        if self._use_threading:
            self._vis_thread = Thread(target=self._run)
            self._update_condition = Condition()
            self._vis_thread.start()

        atexit.register(self.stop)

    def _close(self: Self) -> None:
        """Closes the visualizer."""
        self._stopped = True

        if self._use_threading:
            with self._update_condition:
                self._update_condition.notify()
            self._vis_thread.join()
        else:
            if self._started:
                self._vis.destroy_window()

    def _run(self: Self) -> None:
        """The main loop of the visualizer when used with a thread."""
        while not self._stopped:
            with self._update_condition:
                self._update_condition.wait()
            if not self._started:
                self._create()
            else:
                self._update()

        if self._started:
            self._vis.destroy_window()

    def _create(self: Self) -> None:
        """Run the first time the point cloud is visualized."""
        self._vis.create_window(
            window_name=self._window_name,
            width=self._window_size[0],
            height=self._window_size[1],
        )
        self._vis.add_geometry(self._pcd)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        self._vis.add_geometry(origin)
        self._started = True

    def _update(self: Self) -> None:
        """Updates the visualizer."""
        if self._pcd is None:
            return
        self._pcd.rotate(
            self._R_camera_to_world, center=np.array([0, 0, 0], dtype=np.float64)
        )
        self._vis.update_geometry(self._pcd)
        self._vis.poll_events()
        self._vis.update_renderer()

    def stop(self: Self) -> None:
        """Stops the visualizer."""
        self._close()

    def update(self: Self, pcd: o3d.geometry.PointCloud) -> None:
        """Updates the point cloud to visualize.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            The point cloud to visualize.

        Raises
        ------
        TypeError
            If pcd is not an open3d.geometry.PointCloud object.
        """
        if not isinstance(pcd, o3d.geometry.PointCloud):
            raise TypeError("pcd must be an open3d.geometry.PointCloud object.")

        if self._pcd is None:
            self._pcd = pcd
        else:
            self._pcd.points = pcd.points
            self._pcd.colors = pcd.colors

        if self._use_threading:
            with self._update_condition:
                self._update_condition.notify()
        else:
            if not self._started:
                self._create()
            self._update()

    def update_rotation(self: Self, rot: np.ndarray) -> None:
        """Updates the rotation matrix of the point cloud.

        Parameters
        ----------
        rot : np.ndarray
            The 3x3 rotation matrix.
        """
        self._R_camera_to_world = rot
