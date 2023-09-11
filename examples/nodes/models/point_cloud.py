import depthai as dai

from oakutils.calibration import get_camera_calibration
from oakutils.nodes import create_stereo_depth, get_nn_point_cloud
from oakutils.nodes.models.point_cloud import create_xyz_matrix, create_point_cloud
from oakutils.point_clouds import PointCloudVisualizer, create_point_cloud_from_np


pipeline = dai.Pipeline()
pcv = PointCloudVisualizer()

# get the calibration
calibration = get_camera_calibration(
    rgb_size=(1920, 1080), 
    mono_size=(640,400), 
    is_primary_mono_left=True,  # make sure to set primary to same as align_socket
)

# create the color camera node
out_nodes = create_stereo_depth(
     pipeline,
     align_socket=dai.CameraBoardSocket.LEFT,  # make sure this is same as primary for calibration
)
depth = out_nodes[0]  # the first node from create_stero_depth is the depth node

xyz_in = pipeline.createXLinkIn()
xyz_in.setMaxDataSize(6144000)
xyz_in.setStreamName("xyz")

nn, xout_nn, nn_stream = create_point_cloud(
      pipeline,
      xyz_link=xyz_in.out,
      input_link=depth.depth,
)
nn.inputs["xyz"].setReusePreviousMessage(True)

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue(nn_stream)

    xyz = create_xyz_matrix(calibration.left.size[0], calibration.left.size[1], calibration.left.K)
    buff = dai.Buffer()
    buff.setData(xyz)
    device.getInputQueue("xyz").send(buff)

    while True:
        data = queue.get()
        np_pcl = get_nn_point_cloud(data)
        pcl = create_point_cloud_from_np(np_pcl)
        pcv.update(pcl)
