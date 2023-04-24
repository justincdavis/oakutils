from oakutils import Camera


def test_stop():
    cam = Camera()
    cam.start(block=True)

    cam.stop()

    return True
