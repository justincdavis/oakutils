.. _examples_aruco/stream:

Example: aruco/stream.py
========================

.. code-block:: python

	import cv2
	import depthai as dai
	from oakutils.aruco import ArucoStream
	from oakutils.calibration import get_camera_calibration_basic
	from oakutils.nodes import create_color_camera, create_xout
	
	
	def main():
	    calibration = get_camera_calibration_basic()
	    stream = ArucoStream(
	        cv2.aruco.DICT_4X4_100,
	        0.05,
	        calibration.rgb,
	        5,
	        5,
	        0.95,
	    )
	
	    pipeline = dai.Pipeline()
	    cam = create_color_camera(pipeline)
	    xout_cam = create_xout(pipeline, cam.video, "rgb")
	
	    with dai.Device(pipeline) as device:
	        cam_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
	
	        while True:
	            in_rgb = cam_queue.get()
	            frame = in_rgb.getCvFrame()
	            markers = stream.find(frame)
	            for marker in markers:
	                print(marker)
	            cv2.imshow("frame", stream.draw(frame, markers))
	            if cv2.waitKey(1) == ord("q"):
	                break
	
	
	if __name__ == "__main__":
	    main()

