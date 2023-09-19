import cv2
import depthai as dai

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
from oakutils.nodes.models import create_sobel

pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(
    pipeline, preview_size=(640, 480)
)  # set the preview size to the input of the nn

sobel = create_sobel(
    pipeline,
    input_link=cam.preview,
)
xout_sobel = create_xout(pipeline, sobel.out, "sobel")

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("sobel")

    while True:
        data = queue.get()
        frame = get_nn_bgr_frame(data, normalization=255.0)

        cv2.imshow("sobel frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
