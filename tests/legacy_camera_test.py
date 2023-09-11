import unittest

from oakutils import LegacyCamera

from _utils import check_method_timout


class TestCamera(unittest.TestCase):
    def test_init(self):
        try:
            _ = LegacyCamera()
        except RuntimeError as e:
            if "No available device" in str(e):
                pass
            else:
                raise e

    def test_stop(self):
        cam = LegacyCamera()
        cam.start(block=True)

        result = check_method_timout(cam.stop, "cam.stop", timeout=5)

        self.assertIsNone(result)

        cam.stop()

    def test_assignments(self):
        cam = LegacyCamera(
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
