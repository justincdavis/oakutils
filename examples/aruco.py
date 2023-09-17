import cv2
import depthai as dai
from oakutils.aruco import ArucoFinder
from oakutils.calibration import get_camera_calibration_basic
from oakutils.nodes import create_color_camera, create_xout


def main():
    calibration = get_camera_calibration_basic()
    finder = ArucoFinder(
        cv2.aruco.DICT_4X4_100,
        0.05,
        calibration.rgb
    )

    pipeline = dai.Pipeline()
    cam = create_color_camera(pipeline)
    xout_cam = create_xout(pipeline, cam.video, "rgb")

    with dai.Device(pipeline) as device:
        cam_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_rgb = cam_queue.get()
            frame = in_rgb.getCvFrame()
            markers = finder.find(frame)
            for marker in markers:
                print(marker)
            cv2.imshow("frame", finder.draw(frame, markers))
            if cv2.waitKey(1) == ord("q"):
                break

if __name__ == "__main__":
    main()
