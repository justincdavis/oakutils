import depthai as dai
import cv2

from oakutils.nodes import get_nn_bgr_frame, create_color_camera
from oakutils.nodes.models import create_laplacian


pipeline = dai.Pipeline()

# create the color camera node
cam, xout_cam = create_color_camera(
    pipeline, preview_size=(640, 480)
)  # set the preview size to the input of the nn

lap, xout_lap, lap_stream = create_laplacian(
    pipeline,
    input_link=cam.preview,
    kernel_size=7,
    blur_kernel_size=3,
    use_blur=True,
)

with dai.Device(pipeline) as device:
    rgb_queue: dai.DataOutputQueue = device.getOutputQueue("rgb")
    lp_queue: dai.DataOutputQueue = device.getOutputQueue(lap_stream)

    while True:
        rgb_data = rgb_queue.get()
        cv2.imshow("rgb", rgb_data.getCvFrame())
        
        lp_data = lp_queue.get()
        lp_frame = get_nn_bgr_frame(lp_data)

        cv2.imshow("laplacian frame", lp_frame)
        if cv2.waitKey(1) == ord("q"):
            break
