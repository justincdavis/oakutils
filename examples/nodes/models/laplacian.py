import cv2
import depthai as dai

from oakutils.nodes import create_color_camera, create_xout, get_nn_bgr_frame
from oakutils.nodes.models import create_laplacian

pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(
    pipeline, preview_size=(640, 480)
)  # set the preview size to the input of the nn

lap = create_laplacian(
    pipeline,
    input_link=cam.preview,
    kernel_size=7,
    blur_kernel_size=3,
    use_blur=True,
)
xout_lap = create_xout(pipeline, lap.out, "laplacian")

with dai.Device(pipeline) as device:
    lp_queue: dai.DataOutputQueue = device.getOutputQueue("laplacian")

    while True:
        lp_data = lp_queue.get()
        lp_frame = get_nn_bgr_frame(lp_data)

        cv2.imshow("laplacian frame", lp_frame)
        if cv2.waitKey(1) == ord("q"):
            break
