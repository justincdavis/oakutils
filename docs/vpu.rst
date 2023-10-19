.. _vpu:

Using the onboard VPU as a standlone accelerator
------------------------------------------------

The VPU can be used as a standalone accelerator, without needing
to launch and cameras or other sensors onboard the OAK-D.

An example is as follows:

.. code-block:: python
    import numpy as np
    from oakutils.blobs.models.shave6 import GAUSSIAN_15X15
    from oakutils import VPU


    def main():
        vpu = VPU()
        vpu.reconfigure(GAUSSIAN_15X15)

        fps_buffer = deque(maxlen=30)
        while True:
            # generate some random data, then send to camera and wait for the result
            data = np.array(np.random.random((640, 480, 3)) * 255.0, dtype=np.uint8)
            _ = vpu.run(data)

    if __name__ == "__main__":
        main()

This simple example will generate random data, send it to the VPU, and
wait for the result. This simple abstraction can intake any blob file
and run it on the VPU. The input to the run call MUST be a numpy array 
with the correct size buffer. If the buffer (overall size) of the data
is not the expected size the VPU will crash. 

The VPU can be reconfigured at any time, and the run call will wait
for the result before returning. This allows for a simple abstraction
to the VPU, and allows for the VPU to be used as a standalone
accelerator.

.. _vpu:
