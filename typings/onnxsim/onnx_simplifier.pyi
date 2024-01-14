import onnx
import onnxsim.onnxsim_cpp2py_export as C
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

def simplify(model: Union[str, onnx.ModelProto], check_n: int = 0, perform_optimization: bool = True, skip_fuse_bn: bool = False, overwrite_input_shapes: Incomplete | None = None, test_input_shapes: Incomplete | None = None, skipped_optimizers: Optional[List[str]] = None, skip_constant_folding: bool = False, skip_shape_inference: bool = False, input_data: Incomplete | None = None, dynamic_input_shape: bool = False, custom_lib: Optional[str] = None, include_subgraph: bool = False, unused_output: Optional[Sequence[str]] = None, tensor_size_threshold: str = ..., mutable_initializer: bool = False, *, input_shapes: Incomplete | None = None) -> Tuple[onnx.ModelProto, bool]: ...

class PyModelExecutor(C.ModelExecutor):
    def Run(self, model_str: str, inputs_str: List[str]): ...

def main(): ...
