import depthai as dai
import cv2

from oakutils.blobs import models
from oakutils.nodes import create_color_camera, create_neural_network, get_nn_bgr_frame, get_nn_gray_frame


pipeline = dai.Pipeline()

# create the color camera node
cam = create_color_camera(pipeline, resolution=dai.ColorCameraProperties.SensorResolution.THE_1080_P, preview_size=(640, 480))

# create the neural network node
lp, xout_lp = create_neural_network(pipeline, cam.preview, models.LAPLACIAN_15X15, stream_name="laplacian")

# create another neural network node
lp_gray, xout_lp_gray = create_neural_network(pipeline, lp.passthrough, models.LAPLACIANGRAY_15X15, stream_name="laplacian_gray")

with dai.Device(pipeline) as device:
    lp_queue: dai.DataOutputQueue = device.getOutputQueue("laplacian")
    lp_gray_queue: dai.DataOutputQueue = device.getOutputQueue("laplacian_gray")

    while True:
        lp_data = lp_queue.get()
        lp_gray_data = lp_gray_queue.get()

        lp_frame = get_nn_bgr_frame(lp_data)

        # also do this
        lp_gray_data = lp_gray_data.getData()

        lp_gray_frame = get_nn_gray_frame(lp_gray_data)

        cv2.imshow("lp frame", lp_frame)
        if cv2.waitKey(1) == ord('q'):
                break
        cv2.imshow("lp gray frame", lp_gray_frame)
        if cv2.waitKey(1) == ord('q'):
                break
