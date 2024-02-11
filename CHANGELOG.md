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
