import depthai as dai
import cv2

from oakutils.nodes import create_color_camera, create_image_manip


pipeline = dai.Pipeline()

# create the color camera
cam = create_color_camera(pipeline)

# create the image manip node
manip, xout_manip = create_image_manip(
    pipeline=pipeline, 
    input_link=cam.preview,
    frame_type=dai.RawImgFrame.Type.GRAY8,
    stream_name="gray",
)

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("gray")

    while True:
        lp_data = queue.get()
        frame = lp_data.getCvFrame()

        cv2.imshow("gray frame", frame)
        if cv2.waitKey(1) == ord('q'):
                break
