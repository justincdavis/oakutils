from typing import Optional, Tuple
import atexit
from threading import Thread, Condition

import open3d as o3d


class PointCloudVisualizer:
    """
    A class to visualize open3d point clouds.

    Methods
    -------
    stop()
        Stops the visualizer.
    update(pcd: o3d.geometry.PointCloud)
        Updates the point cloud to visualize.
    """

    def __init__(
        self,
        window_name: str = "PointCloud",
        window_size: Tuple[int, int] = (1920, 1080),
        use_threading: bool = True,
    ) -> "PointCloudVisualizer":
        """
        Creates a PointCloudVisualizer object.

        Parameters
        ----------
        window_name : str
            The name of the visualization window. Defaults to "PointCloud".
        window_size : Tuple[int, int]
            The size of the visualization window. Defaults to (1920, 1080).
        use_threading : bool
            Whether to use threading for visualization. Defaults to True.
        """
        self._pcd: Optional[o3d.geometry.PointCloud] = None
        self._vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        self._window_name: str = window_name
        self._window_size: Tuple[int, int] = window_size
        self._started: bool = False
        self._stopped: bool = False

        self._use_threading: bool = use_threading
        if self._use_threading:
            self._vis_thread = Thread(target=self._run)
            self._update_condition = Condition()
            self._vis_thread.start()

        atexit.register(self.stop)

    def _close(self) -> None:
        """
        Closes the visualizer.
        """
        self._stopped = True

        if self._use_threading:
            with self._update_condition:
                self._update_condition.notify()
            self._vis_thread.join()
        else:
            if self._started:
                self._vis.destroy_window()

    def _run(self) -> None:
        """
        The main loop of the visualizer when used with a thread.
        """
        while not self._stopped:
            with self._update_condition:
                self._update_condition.wait()
            if not self._started:
                self._create()
            else:
                self._update()

        if self._started:
            self._vis.destroy_window()

    def _create(self) -> None:
        """
        Run the first time the point cloud is visualized.
        """
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

    def _update(self) -> None:
        """
        Updates the visualizer.
        """
        self._vis.update_geometry(self._pcd)
        self._vis.poll_events()
        self._vis.update_renderer()

    def stop(self) -> None:
        """
        Stops the visualizer.
        """
        self._close()

    def update(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Updates the point cloud to visualize.

        :param pcd: The point cloud to visualize.
        :type pcd: o3d.geometry.PointCloud
        """
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
