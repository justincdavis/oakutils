from oakutils import OAK_Camera


def test_stop():
    cam = OAK_Camera()
    cam.start(block=True)

    cam.stop()

    return True
