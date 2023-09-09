from . import models

__all__ = [
    "models",
]

try:
    from . import definitions
    from ._compiler import compile, compile_onnx

    __all__ = [
        "models",
        "definitions",
        "compile",
        "compile_onnx",
    ]
except ImportError as e:
    print(e)
    pass
