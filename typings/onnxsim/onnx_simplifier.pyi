import onnxsim.onnxsim_cpp2py_export as C
import onnx.numpy_helper
from . import model_checking as model_checking, model_info as model_info, version as version
from _typeshed import Incomplete
from typing import Dict, List, Optional, Sequence, Tuple, Union

command: Incomplete
TensorShape = List[int]
TensorShapes = Dict[str, TensorShape]
TensorShapesWithOptionalKey = Dict[Optional[str], TensorShape]

def get_output_names(model: onnx.ModelProto) -> List[str]: ...
def remove_unused_output(model: onnx.ModelProto, unused_output: Sequence[str]) -> onnx.ModelProto: ...
def remove_initializer_from_input(model: onnx.ModelProto) -> onnx.ModelProto: ...
def check_and_update_input_shapes(model: onnx.ModelProto, input_shapes: Optional[TensorShapesWithOptionalKey]) -> Optional[TensorShapes]: ...

MAX_TENSOR_SIZE_THRESHOLD: str

def simplify(model: Union[str, onnx.ModelProto], check_n: int = ..., perform_optimization: bool = ..., skip_fuse_bn: bool = ..., overwrite_input_shapes: Incomplete | None = ..., test_input_shapes: Incomplete | None = ..., skipped_optimizers: Optional[List[str]] = ..., skip_constant_folding: bool = ..., skip_shape_inference: bool = ..., input_data: Incomplete | None = ..., dynamic_input_shape: bool = ..., custom_lib: Optional[str] = ..., include_subgraph: bool = ..., unused_output: Optional[Sequence[str]] = ..., tensor_size_threshold: str = ..., mutable_initializer: bool = ..., *, input_shapes: Incomplete | None = ...) -> Tuple[onnx.ModelProto, bool]: ...

class PyModelExecutor(C.ModelExecutor):
    def Run(self, model_str: str, inputs_str: List[str]): ...

def main(): ...
