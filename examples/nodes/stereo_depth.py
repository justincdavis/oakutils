import cv2
import depthai as dai

from oakutils.nodes import create_stereo_depth_from_mono_cameras, create_left_right_cameras

pipeline = dai.Pipeline()

# create the color camera node
left, xout_left, right, xout_right = create_left_right_cameras(
    pipeline,
    resolution=dai.MonoCameraProperties.SensorResolution.THE_400_P,
    fps=60,
)
right, xout_right = create_stereo_depth_from_mono_cameras(
    pipeline, 
    left,
    right, 
    preset=dai.node.StereoDepth.PresetMode.HIGH_ACCURACY,
    lr_check=True, 
    extended_disparity=True, 
    subpixel=False,
    median_filter=dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
)

with dai.Device(pipeline) as device:
    lq: dai.DataOutputQueue = device.getOutputQueue("synced_left")
    rq: dai.DataOutputQueue = device.getOutputQueue("synced_right")
    depthq: dai.DataOutputQueue = device.getOutputQueue("depth")
    disparityq: dai.DataOutputQueue = device.getOutputQueue("disparity")

    while True:
        left = lq.get()
        right = rq.get()
        depth = depthq.get()
        disparity = disparityq.get()
        cv2.imshow("left", left.getCvFrame())
        cv2.imshow("right", right.getCvFrame())
        cv2.imshow("depth", depth.getFrame())
        cv2.imshow("disparity", disparity.getFrame())
        if cv2.waitKey(1) == ord("q"):
            break
