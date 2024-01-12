.. _compiling:

Compiling Custom Models and CV Functions
----------------------------------------

.. note::
    
     This feature requires the compiler dependencies to be installed. 
     See the :ref:`installation` page for more information.

Oakutils allows you to compile custom models and cv functions for the camera.
There are two primary ways to do this:

1.  Use the `oakutils.blobs.compile_model` function to compile a single model defined as a torch.nn.Module.
2.  Use the `oakutils.blobs.compile_onnx` function to compile a ONNX model from a file.

Using the `compile_model` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import oakutils.blobs import compile_model

    model_path = compile_model(
        Custom,  # A oakutils.blobs.definitions.AbstractModel class (superclass of torch.nn.Module)
        {},  # A dictionary of constructor parameters, same as kwargs
        cache=False,  # whether to use the cache, stores results in site-packages for reuse
        shaves=6, # The number of shaves to use for compilation, shaves are processing units on the VPU
        shape_mapping={  # This dict maps predefined InputTypes to (W, H, C) tuples for input generation
            InputType.FP16: (300, 300, 3),  
        },
        creation_func=torch.ones  # A function for creating the actual traced inputs (generally use one which produces a value, not torch.zeros)
    )

The above piece of code will compile the model defined in the `Custom` class.
The `compile_model` function takes an AbstractModel (a superclass of torch.nn.Module)
as it's input to compile. An example of such a class is shown below:

.. code-block:: python

    class Gaussian(AbstractModel):
    """nn.Module wrapper for kornia.filters.gaussian_blur2d."""

        def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
            super().__init__()
            self._kernel_size = kernel_size
            self._sigma = sigma

        @classmethod
        def model_type(cls: Gaussian) -> ModelType:
            """The type of input this model takes."""
            return ModelType.KERNEL

        @classmethod
        def input_names(cls: Gaussian) -> list[tuple[str, InputType]]:
            """The names of the input tensors."""
            return [("input", InputType.FP16)]

        @classmethod
        def output_names(cls: Gaussian) -> list[str]:
            """The names of the output tensors."""
            return ["output"]

        def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
            return kornia.filters.gaussian_blur2d(
                image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
            )

This definition of the Gaussian class will produce a blob which performs a gaussian blur
using the kernel and sigma parameters specified in the constructor. 

As one might notice there are three additional class methods which a typical torch.nn.Module
would not have. These methods allow the `compile_model` function to determine the type/shape of
input the model takes, the names of the input tensors, the names of the output tensors, and
the arrangement of constructor parameters. This additional verbose definition allows
the compilation of arbitrary models without having the need for custom functions or compilation steps.

1. `model_type` - 
This method returns a `ModelType` enum which specifies the style of parameters
which the constructor takes. For user defined (non-distributed) models, setting 
this to `ModelType.NONE` is sufficient.

2. `input_names` - 
This method returns a list of tuples which specify the names of the input tensors 
and the type of input they take. The type of input is specified by the `InputType` enum. 
The `InputType` enum is used to determine the shape of the input tensors. 
This comes from the `shape_mapping` parameter in the `compile_model` function. 

3. `output_names` - 
This method returns a list of strings which specify the names of the output tensors. 
The output type is NOT determined at "compile time" and is determined when decoding 
the output. Some examples of this are: 
`from oakutils.nodes import get_nn_frame, get_nn_bgr_frame, get_nn_gray_frame`. 
Each function is called by the host (not OAK) when processing the xout frames in a buffer. 
An error in the decoding will typically result in a `ValueError` being raised, since 
the buffer does not fit into the allocated array size.

Given are the definitions of the three datatypes used in the above class:

.. code-block:: python

    class AbstractModel(ABC, torch.nn.Module):
        def __init__(self: Self) -> None:
            super().__init__()

        @classmethod
        @abstractmethod
        def model_type(cls: AbstractModel) -> ModelType:
            """The type of input this model takes."""

        @classmethod
        @abstractmethod
        def input_names(cls: AbstractModel) -> list[tuple[str, InputType]]:
            """The names of the input tensors."""

        @classmethod
        @abstractmethod
        def output_names(cls: AbstractModel) -> list[str]:
            """The names of the output tensors."""

    class InputType(Enum):
    """Represents the type of a given input to a model in the forward call
    E.g. FP16, U8, etc.
    """
        FP16 = 0
        U8 = 1
        XYZ = 2

    class ModelType(Enum):
    """Represents the different arguments a model constructor can take."""
        NONE = 0
        KERNEL = 1
        DUAL_KERNEL = 2

Using the `compile_onnx` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import oakutils.blobs import compile_onnx

    def compile_onnx(
        model_path: str,
        output_path: str,
        shaves: int = 6,
        version: str = "2022.1",
        simplify: bool | None = None,
    ) -> None:

The compile_onnx functions as a wrapper around the `blobconverter.from_onnx` function.
It takes an ONNX model file and compiles it to a blob. The `output_path` parameter
specifies the path to the output blob. The `shaves` parameter specifies the number
of shaves to use for compilation. The `version` parameter specifies the version of
OpenVINO to use for compilation. The `simplify` parameter specifies whether to
simplify the model before compilation. Simplfication is done with the onnxsim package.

This function is provided for convenience and is not as flexible as the `compile_model`.

Defining Custom InputTypes
^^^^^^^^^^^^^^^^^^^^^^^^^^

The `InputType` enum is used to determine the shape of the input tensors. This comes from the `shape_mapping` parameter
in the `compile_model` function. If the user wants to define a custom `InputType` they can do so by subclassing the `InputType` enum.
An example of this is shown below:

.. code-block:: python

    class CustomInputType(InputType):
        NEW_INPUT = 3

    class CustomModel(AbstractModel):
        """nn.Module wrapper for kornia.filters.gaussian_blur2d."""

        def __init__(self: Self, kernel_size: int = 3, sigma: float = 1.5) -> None:
            super().__init__()
            self._kernel_size = kernel_size
            self._sigma = sigma

        @classmethod
        def model_type(cls: CustomModel) -> ModelType:
            """The type of input this model takes."""
            return ModelType.KERNEL

        @classmethod
        def input_names(cls: CustomModel) -> list[tuple[str, InputType]]:
            """The names of the input tensors."""
            return [("input", CustomInputType.NEW_INPUT)]

        @classmethod
        def output_names(cls: CustomModel) -> list[str]:
            """The names of the output tensors."""
            return ["output"]

        def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
            return kornia.filters.gaussian_blur2d(
                image, (self._kernel_size, self._kernel_size), (self._sigma, self._sigma)
            )

The above code defines a new `InputType` called `CustomInputType` which has a value of 3.
To use this new `InputType` in the `compile_model` function, the user would need to specify
the `shape_mapping` parameter as follows:

.. code-block:: python

    model_path = compile_model(
        CustomModel,  # Custom class from above
        {},  # A model does not take arguments
        cache=False,  
        shaves=6, 
        shape_mapping={  # This dict maps predefined InputTypes to (W, H, C) tuples for input generation
            CustomInputType.NEW_INPUT: (300, 300, 3),  
        },
    )
