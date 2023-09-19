.. _compiling:

Compiling Custom Models and CV Functions:
------------------------

Note: This feature requires the compiler dependencies to be installed.  See the
:ref:`installation` page for more information.

Oakutils allows you to compile custom models and cv functions for the camera.
There are two primary ways to do this:

1.  Use the `oakutils.blobs.compile_model` function to compile a single model defined as a torch.nn.Module.
2.  Use the `oakutils.blobs.compile_onnx` function to compile a ONNX model from a file.

