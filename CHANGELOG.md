## 1.5.3 (07-27-2024)

### Added

- blobs.benchmark_blob
    - Measure the latency of a compiled blob.
- nodes.get_yolo_data
    - Create a YolomodelData object from a json file.

## 1.5.2 (07-09-2024)

### Added 

- benchmark submodule
    - measure_latency and measure_throughput functions.
        Allow measuring roundtrip latency and max 
        uplink and downlink throughput.
- Info functions for compiled blobs
    - Can get the layer information for input/output layers
        of compiled blob files.
    - Added function for getting blob objects via paths
        with verification.

### Improvements

- create_device
    - Now accepts a usb speed parameter.

## 1.5.1 (05-15-2024)

### Added

- Supported for OAK-1 devices through the calibration API
    - get_camera_calibration now reads device type and pulls
        correct information.
- ApiCamera, LegacyCamera, VPU, and Webcam now accept
    a device_id for specifying which physical OAK device
    to utilize.
- create_device: A function for creating a DepthAI Device
    from a pre-defined pipeline. Accepts parameters for
    where to create said device.
- nodes.buffer submodule
    - Buffer, MultiBuffer, SimpleBuffer classes. Each class
        enables methods for making DataInput/OutputQueues easier
        to use in bulk.
    - create_synced_buffer, allows wrapping multiple output
        streams with a single function getting a single output
        from each stream at once.

### Improvements

- VPU methods refactored to handle multiple neural networks
    - reconfigure parameters changed
    - Added reconfigure_multi
    - Function signature changed for run
    - Pipeline generation and VPU thread use new backend

### Changed

- get_camera_calibration
    - New return type of CalibrationData | ColorCalibrationData
        This comes from the OAK-1 support
    - Could break old code which utilized this function, but
        would primarily affect static type-checkers like mypy
    - Can replace with get_oakd_calibration to achieve 1:1

## 1.5.0 (04-08-2024)

### Changed

- Removed Open3d as a strict requirement
    - point_clouds submodule only imported if open3d is found
    - Pinhole models in calibration will be None unless open3d is found
    - If open3d is found then oakutils will function identically as pre 1.5
    - open3d can be installed with oakutils[o3d]
- Added cv2ext as a strict requirment
    - Switched the DisplayManager to use cv2ext.Display

## 1.4.5 (02-22-2024)

### Added

- get_nn_data function
    - allows getting generic data from an executed neuralnetwork

### Fixed

- VPU
    - Incorrect return type for run
    - blob_path listed incorrectly as parameter for init
    - Changed reconfigure to accept str | Path
- Documentation
    - Resolved indentation errors in docs

### Changed

- Removed README from doc homepage

## 1.4.4 (02-21-2024)

### Fixed

- Package gets detected as being fully typed.

## 1.4.3 (02-10-2024)

### Added

- Support for Python 3.11

### Changed

- Dependency versions for almost every package to ensure more consistent behavior

### Fixed

- CI test workflows not correctly propagating error codes
- Bug with queues on color camera, mono camera, image manip, xout, and stereo depth. 
    Queue flags should all be none and then only assigned if user provides.

## 1.4.2 (01-29-2024)

### Improvements

- Better logging for compiling custom models

### Added

- set_log_level function for controlling log levels programmatically
- clear_cache function in blobs module for wiping all saved models

### Changed

- Exposed setting the onnx opset through the compile_model function
- Exposed setting the openvino version through the compile_model function
- Failed connections from blobconverter are captured successfully
- The model_tester script has been updated to have U8 and FP16 models

### Fixed

- uint8 input models can now be successfully compiled
    - Previously, a double column hack was being used similiar to what was 
      employed in the depthai-experiments for on-device point cloud generation.
      This hack is no longer needed with new version of depthai.

## 1.4.1 (01-25-2024)

### Changed

- Better error messages during blob compilation
    - Capture and parse error message from blobconverter
    - User receives exact error from OpenVINO instead of json dump

### Fixed

- VPU has no "vpu_out" stream, for generic networks.
- Error in model install tests.

## 1.4.0 (01-22-2024)

### Added

- Automatic doc building to include example files
- Stricter linting
    - Increase amount of rules Ruff checks for
    - Changed boolean parameters to be keyword only (Ruff FTB)
- MyPy Static Type checking
- Stub generation with Pyright, static type checking with Pyright (non-CI)

### Changed

- VPU access API
    - Switched from single class to submodule
    - Support for Yolo and TFOD networks
- Switched to Pathlib vs. os module calls for internals
