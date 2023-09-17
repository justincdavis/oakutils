import cv2
import depthai as dai
import oakutils as oak


def main():
    calibration = oak.calibration.get_camera_calibration()
    finder = oak.aruco.ArucoFinder(
        cv2.aruco.DICT_4X4_100,
        0.05,
        calibration,
    )

    pipeline = dai.Pipeline()
    cam = oak.nodes.create_color_camera(pipeline)
    xout_cam = oak.nodes.create_xout(pipeline, cam.video, "rgb")

    with dai.Device(pipeline) as device:
        cam_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_rgb = cam_queue.get()
            frame = in_rgb.getCvFrame()
            markers = finder.find(frame)
            print(markers)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord("q"):
                break
