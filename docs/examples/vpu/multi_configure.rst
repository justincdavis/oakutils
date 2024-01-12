.. _examples_vpu/multi_configure:

Example: vpu/multi_configure.py
===============================

.. code-block:: python

	import time
	from collections import deque
	
	import numpy as np
	from oakutils.blobs.models.shave6 import GAUSSIAN_15X15, LAPLACIAN_15X15
	from oakutils import VPU
	
	
	SWAP_TIME = 10
	CURRENT_MODEL = GAUSSIAN_15X15
	
	def get_model():
	    global CURRENT_MODEL
	    if CURRENT_MODEL == GAUSSIAN_15X15:
	        CURRENT_MODEL = LAPLACIAN_15X15
	    else:
	        CURRENT_MODEL = GAUSSIAN_15X15
	    return CURRENT_MODEL
	
	def main():
	    vpu = VPU()
	    vpu.reconfigure(CURRENT_MODEL)
	    fps_buffer = deque(maxlen=SWAP_TIME)
	    counter = 0
	    while True:
	        # reconfigure every 10 frames
	        if counter == SWAP_TIME - 1:
	            vpu.reconfigure(get_model())
	            counter = 0
	            fps_buffer.clear()
	        # generate some random data, then send to camera and wait for the result
	        data = np.array(np.random.random((640, 480, 3)) * 255.0, dtype=np.uint8)
	        t0 = time.perf_counter()
	        vpu.run(data)
	        t1 = time.perf_counter()
	        fps_buffer.append(1.0 / (t1 - t0))
	        print(f"FPS: {np.mean(fps_buffer):.2f}")
	        counter += 1
	
	if __name__ == "__main__":
	    main()

