# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Example showcasing how to use ArucoStream on the Webcam abstraction."""

import cv2
from oakutils import Webcam
from oakutils.aruco import ArucoStream

cam = Webcam()
stream = ArucoStream(
    aruco_dict=cv2.aruco.DICT_5X5_100,
    marker_size=0.2,
    calibration=cam.calibration,
)

while True:
    _, frame = cam.read()
    markers = stream.find(frame)
    cv2.imshow("frame", stream.draw(frame, markers))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
