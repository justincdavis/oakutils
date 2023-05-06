# Simple example of using the Camera class to display the output of the OAK-D camera
import time

import cv2
from oakutils import Camera


DISPLAY_TIME = 10

cam = Camera(
    display_depth=True,
    display_mono=True,
    display_rectified=True,
    compute_im3d_on_demand=True,  # on demand, so must call compute_im3d to update
)

cam.start_display()
cam.start()

start_time = time.time()
while True:
    if time.time() - start_time > DISPLAY_TIME:
        break

    cam.compute_im3d(block=True)
    cv2.imshow("im3d", cam.im3d)
    cv2.waitKey(1)

    time.sleep(0.05)

cam.stop()
