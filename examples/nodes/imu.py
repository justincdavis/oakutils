import cv2
import depthai as dai

from oakutils.nodes import create_imu

pipeline = dai.Pipeline()

# create the color camera node
imu, xout_imu = create_imu(
    pipeline,
    accelerometer_rate=400,
    gyroscope_rate=400,
    enable_accelerometer=True,
    enable_gyroscope_calibrated=True,
    enable_game_rotation_vector=True,
)

with dai.Device(pipeline) as device:
    queue: dai.DataOutputQueue = device.getOutputQueue("imu")

    while True:
        left = queue.get()

        print(dir(left))
