import unittest
import random

from oakutils.tools.display import DisplayManager

from _utils import generate_random_frame, check_method_timout


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

        result = check_method_timout(display_manager.stop, "display_manager.stop", timeout=5)

        self.assertIsNone(result)
        
if __name__ == "__main__":
    unittest.main()
