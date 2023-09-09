# from __future__ import annotations

# from threading import Thread, Condition
# from typing import Tuple, Optional, Union, Dict, Callable, Iterable
# import atexit
# from functools import partial

# import depthai as dai
# import depthai_sdk as sdk

# from .calibration import CalibrationData, get_camera_calibration
# from .point_clouds import PointCloudVisualizer
# from .tools.display import DisplayManager, get_smaller_size


# class Camera:
#     def __init__(
#         self,
#         # standard sdk.OakCamera args
#         device: Optional[str] = None,
#         usb_speed: Optional[Union[str, dai.UsbSpeed]] = None,
#         replay: Optional[str] = None,
#         rotation: Optional[int] = None,
#         config: Optional[dai.Device.Config] = None,
#         # custom args
#         primary_mono_left: bool = True,
#         color_size: Tuple[int, int] = (1920, 1080),
#         mono_size: Tuple[int, int] = (640, 400),
#     ):
#         # store custom args
#         self._color_size: Tuple[int, int] = color_size
#         self._mono_size: Tuple[int, int] = mono_size
#         self._primary_mono_left: bool = primary_mono_left

#         # handle attributes
#         self._calibration: Optional[CalibrationData] = None
#         self._callbacks: Dict[Union[str, Iterable[str]], Callable] = {}
#         self._pipeline: Optional[dai.Pipeline] = None
#         self._is_built: bool = False

#         # handle custom displays directly for API stuff without visualize
#         self._display_size: Tuple[int, int] = get_smaller_size(
#             self._color_size, self._mono_size
#         )
#         self._displays: Optional[DisplayManager] = None
#         self._pcv: Optional[PointCloudVisualizer] = None

#         # set the arguments
#         self._oak_args = {
#             "device": device,
#             "usb_speed": usb_speed,
#             "replay": replay,
#             "rotation": rotation,
#             "config": config,
#         }

#         # setup the cameras
#         self._oak: Optional[sdk.OakCamera] = None
#         self._device: Optional[dai.Device] = None
#         self._pipeline: Optional[dai.Pipeline] = None

#         # thread for reading camera
#         self._built = False
#         self._started = False
#         self._stopped: bool = False
#         self._thread: Thread = Thread(target=self._run, daemon=True)
#         self._build_condition: Condition = Condition()
#         self._start_condition: Condition = Condition()
#         self._stop_condition: Condition = Condition()
#         self._intialize_condition: Condition = Condition()
#         self._thread.start()

#         # register stop function
#         atexit.register(self.stop)

#         # wait for the camera to be ready
#         with self._intialize_condition:
#             self._intialize_condition.wait()

#     def __del__(self):
#         self.stop()

#     @property
#     def oak(self) -> sdk.OakCamera:
#         """
#         Returns the underlying OakCamera object.

#         Raises
#         ------
#         RuntimeError
#             If the OakCamera has not been built yet.
#         """
#         if self._oak is None:
#             raise RuntimeError(
#                 "OakCamera has not been built yet. Failure in processing thread."
#             )
#         return self._oak

#     @property
#     def device(self) -> dai.Device:
#         """
#         Returns the underlying Device object.

#         Raises
#         ------
#         RuntimeError
#             If the Device has not been built yet.
#         """
#         if self._oak is None:
#             raise RuntimeError(
#                 "Device has not been built yet. Failure in processing thread."
#             )
#         return self._device

#     @property
#     def pipeline(self) -> dai.Pipeline:
#         """
#         Returns the pipeline. If the pipeline has not been built yet, a RuntimeError is raised.
#         This is useful for adding custom nodes to the pipeline.

#         Raises
#         ------
#         RuntimeError
#             If the pipeline has not been built yet.
#         """
#         if self._pipeline is None:
#             raise RuntimeError(
#                 "Pipeline has not been built yet. Failure in depthai_sdk.OakCamera.start() or in processing thread."
#             )
#         return self._pipeline

#     @property
#     def calibration(self) -> CalibrationData:
#         """
#         Returns the calibration data.

#         Raises
#         ------
#         RuntimeError
#             If self.device is not yet available.
#         """
#         if self._calibration is None:
#             self._calibration = get_camera_calibration(
#                 self.device, self._color_size, self._mono_size, self._primary_mono_left
#             )
#         return self._calibration

#     @property
#     def displays(self) -> DisplayManager:
#         """
#         Returns the display manager.
#         """
#         if self._displays is None:
#             self._displays = DisplayManager(display_size=self._display_size)
#         return self._displays

#     @property
#     def pcv(self) -> PointCloudVisualizer:
#         """
#         Returns the point cloud visualizer.
#         """
#         if self._pcv is None:
#             self._pcv = PointCloudVisualizer(window_size=self._display_size)
#         return self._pcv

#     def build(self):
#         """
#         Builds the pipeline. To be done after all sdk calls are made.
#         """
#         with self._build_condition:
#             self._build_condition.notify()
#         self._built = True

#     def start(self, blocking: bool = False):
#         """
#         Starts the camera. To be done after all api calls are made.
#         Will build the pipeline if it has not been built yet.
#         """
#         if not self._built:
#             self.build()

#         with self._start_condition:
#             self._start_condition.notify()
#         self._started = True

#         if blocking:
#             with self._stop_condition:
#                 self._stop_condition.wait()

#     def stop(self):
#         """
#         Stops the camera.
#         """
#         self._stopped = True

#         # call conditions if system never started
#         with self._build_condition:
#             self._build_condition.notify_all()
#         with self._start_condition:
#             self._start_condition.notify_all()

#         try:
#             self._thread.join()
#         except RuntimeError:
#             pass

#     def add_callback(self, name: Union[str, Iterable[str]], callback: Callable):
#         """
#         Adds a callback to be run on the output queue with the given name.

#         Parameters
#         ----------
#         name : str
#             The name of the output queue to add the callback to.
#         callback : Callable
#             The callback to add.
#         """
#         self._callbacks[name] = callback

#     def _run(self):
#         self._oak = sdk.OakCamera(**self._oak_args)
#         self._device = self._oak.device

#         with self._intialize_condition:
#             self._intialize_condition.notify()

#         # wait for the build call, this allows user to define sdk calls
#         with self._build_condition:
#             self._build_condition.wait()

#         # build sdk pipeline
#         self._pipeline = self._oak.build()

#         # wait for the start call, this allows user to define pipeline
#         with self._start_condition:
#             self._start_condition.wait()

#         with self._oak as oak:
#             # start the camera and run the pipeline
#             oak.start()

#             # get the output queues ahead of time
#             queues = {
#                 name: oak.device.geatOutputQueue(name)
#                 for name, _ in self._callbacks.items()
#             }
#             # create a cache for queue results to enable multi queue callbacks
#             data_cache = {name: None for name, _ in self._callbacks.items()}
#             while not self._stopped:
#                 # poll the camera to get new data
#                 oak.poll()
#                 # cache results
#                 for name in data_cache.keys():
#                     data_cache[name] = queues[name].get()
#                 # create callback partials
#                 partials = []
#                 for name, callback in self._callbacks.items():
#                     if isinstance(name, str):
#                         data = data_cache[name]
#                     else:
#                         data = [data_cache[n] for n in name]
#                     partials.append(partial(callback, data))
#                 # run/dispatch the callback partials
#                 # TODO: run in async loop or another thread or process?
#                 for callback in partials:
#                     callback()

#         # call stop conditions if start was called with blocking
#         with self._stop_condition:
#             self._stop_condition.notify()
