import time

from oakutils import LegacyCamera


def main():
    cam = LegacyCamera(
        display_depth=True,
    )
    cam.start(block=True)

    time.sleep(10)

    cam.stop()


if __name__ == "__main__":
    main()
