import unittest

from oakutils import Camera

from _utils import check_method_timout


class TestCamera(unittest.TestCase):
    def test_stop(self):
        cam = Camera()
        cam.start(block=True)

        result = check_method_timout(cam.stop, "cam.stop", timeout=5)

        self.assertIsNone(result)

        cam.stop()

    def test_assignments(self):
        cam = Camera(
            enable_rgb=True,
            enable_mono=True,
        )

        self.assertIsNone(cam.rgb)
        self.assertIsNone(cam.depth)
        self.assertIsNone(cam.left)
        self.assertIsNone(cam.right)
        
        cam.start(block=True)

        self.assertIsNotNone(cam.rgb)
        self.assertIsNotNone(cam.depth)
        self.assertIsNotNone(cam.left)
        self.assertIsNotNone(cam.right)

        cam.stop()

if __name__ == "__main__":
    unittest.main()
