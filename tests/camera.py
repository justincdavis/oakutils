import concurrent.futures
import unittest

from oakutils import Camera


class TestCamera(unittest.TestCase):
    def test_stop(self):
        cam = Camera()
        cam.start(block=True)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(cam.stop)
            try:
                result = future.result(timeout=5)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError("cam.stop timed out after 5 seconds")

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
