import depthai as dai

from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_stereo_depth, create_xout, get_nn_point_cloud
from oakutils.nodes.models.point_cloud import create_point_cloud, create_xyz_matrix
from oakutils.point_clouds import PointCloudVisualizer, create_point_cloud_from_np

pipeline = dai.Pipeline()
pcv = PointCloudVisualizer()

# get the calibration
calibration = get_camera_calibration(
    rgb_size=(1920, 1080),
    mono_size=(640, 400),
    is_primary_mono_left=True,  # make sure to set primary to same as align_socket
)

# create the color camera node
out_nodes = create_stereo_depth(
    pipeline,
    align_socket=dai.CameraBoardSocket.LEFT,  # make sure this is same as primary for calibration
)
depth = out_nodes[0]  # the first node from create_stero_depth is the depth node

pcl, xin_pcl, start_pcl = create_point_cloud(
    pipeline,
    depth_link=depth.depth,
    calibration=calibration,
)
xout_pcl = create_xout(pipeline, pcl.out, "pcl")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("pcl")

    start_pcl(device)

    while True:
        data = queue.get()
        np_pcl = get_nn_point_cloud(data)
        pcl = create_point_cloud_from_np(np_pcl)
        pcv.update(pcl)
