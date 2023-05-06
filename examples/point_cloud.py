# Simple example of using the Camera class to display the point cloud
import time

from oakutils import Camera
from oakutils.point_clouds import get_point_cloud_from_depth_image, PointCloudVisualizer


DISPLAY_TIME = 10

# using threading on the Visualizers can potentially improve performance
VIS = PointCloudVisualizer(window_name="Depth Point Cloud")
VIS2 = PointCloudVisualizer(
    window_name="RGB Point Cloud", use_threading=False
)  # showing that both options work

cam = Camera(
    display_point_cloud=False,  # the camera class has built in point cloud for RGBD images but we will be explicit
    compute_point_cloud_on_demand=True,  # on demand, so must call compute_point_cloud to update
)

cam.start_display()
cam.start()

start_time = time.time()
while True:
    if time.time() - start_time > DISPLAY_TIME:
        break

    cam.compute_im3d(block=True)
    cam.compute_point_cloud(block=True)

    pcd = get_point_cloud_from_depth_image(cam.depth, cam.calibration.primary.pinhole)
    VIS.update(pcd)
    VIS2.update(cam.point_cloud)

    time.sleep(0.05)

VIS.stop()
VIS2.stop()
cam.stop()
