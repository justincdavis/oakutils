import depthai as dai
import cv2

from oakutils.nodes import get_nn_bgr_frame, create_color_camera
from oakutils.nodes.models import create_sobel


pipeline = dai.Pipeline()

# create the color camera node
cam, xout_cam = create_color_camera(pipeline, preview_size=(640, 480))  # set the preview size to the input of the nn

sobel, xout_sobel, sobel_stream = create_sobel(
      pipeline,
      input_link=cam.preview,
)

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue(sobel_stream)

    while True:
        data = queue.get()
        frame = get_nn_bgr_frame(data)

        cv2.imshow("sobel frame", frame)
        if cv2.waitKey(1) == ord('q'):
                break
