import cv2
from oakutils import Webcam


def main():
    cam = Webcam()
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()
