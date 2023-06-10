from typing import Tuple
import random

import cv2
import numpy as np


def generate_random_frame(size: Tuple[int, int], num_rectangles: int = 1) -> np.ndarray:
    width, height = size
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    for _ in range(num_rectangles):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = random.randint(1, 10)
        line_type = random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_type)
    return frame
