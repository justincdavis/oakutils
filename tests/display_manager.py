import unittest
import random
import concurrent

import oakutils
import oakutils.tools
import oakutils.tools.display

from ._utils.images import generate_random_frame


class TestDisplayManager(unittest.TestCase):
    def test_display_manager(self):
        display_manager = DisplayManager()

        frames = [
            (
            f"test_{random.randint(1,10)}",
            generate_random_frame((640, 480)),
            )
            for _ in range(10)
        ]

        for name, frame in frames:
            display_manager.update(name, frame)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(display_manager.stop)
            try:
                result = future.result(timeout=5)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError("cam.stop timed out after 5 seconds")

        self.assertIsNone(result)
        