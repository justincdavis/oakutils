## 1.4.1 (01-25-2024)

### Changed

- Better error messages during blob compilation
    - Capture and parse error message from blobconverter
    - User receives exact error from OpenVINO instead of json dump

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
