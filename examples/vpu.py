import time
from collections import deque

import numpy as np
from oakutils.blobs.models.shave6 import GAUSSIAN_15X15
from oakutils import VPU


def main():
    vpu = VPU()
    vpu.reconfigure(GAUSSIAN_15X15)

    fps_buffer = deque(maxlen=30)
    while True:
        # generate some random data
        data = np.array(np.random.random((640, 480, 3)) * 255.0, dtype=np.uint8)
        t0 = time.perf_counter()
        vpu.run(data)
        t1 = time.perf_counter()
        fps_buffer.append(1.0 / (t1 - t0))
        print(f"FPS: {np.mean(fps_buffer):.2f}")

if __name__ == "__main__":
    main()
