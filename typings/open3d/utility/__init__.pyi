"""
This type stub file was generated by pyright.
"""

import numpy
from typing import Any, ClassVar, Iterable, Iterator, overload

"""
This type stub file was generated by pyright.
"""
Debug: VerbosityLevel
Error: VerbosityLevel
Info: VerbosityLevel
Warning: VerbosityLevel
class DoubleVector:
    @overload
    def __init__(self, arg0: buffer) -> None:
        ...
    
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: DoubleVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: float) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: float) -> int:
        ...
    
    @overload
    def extend(self, L: DoubleVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: float) -> None:
        ...
    
    @overload
    def pop(self) -> float:
        ...
    
    @overload
    def pop(self, i: int) -> float:
        ...
    
    def remove(self, x: float) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: float) -> bool:
        ...
    
    def __copy__(self) -> DoubleVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> DoubleVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: DoubleVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> DoubleVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> float:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: DoubleVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: float) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: DoubleVector) -> None:
        ...
    


class IntVector:
    @overload
    def __init__(self, arg0: buffer) -> None:
        ...
    
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: IntVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: int) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: int) -> int:
        ...
    
    @overload
    def extend(self, L: IntVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: int) -> None:
        ...
    
    @overload
    def pop(self) -> int:
        ...
    
    @overload
    def pop(self, i: int) -> int:
        ...
    
    def remove(self, x: int) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: int) -> bool:
        ...
    
    def __copy__(self) -> IntVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> IntVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: IntVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> IntVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> int:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: IntVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: int) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: IntVector) -> None:
        ...
    


class Matrix3dVector:
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Matrix3dVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> int:
        ...
    
    @overload
    def extend(self, L: Matrix3dVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        ...
    
    @overload
    def pop(self) -> numpy.ndarray[numpy.float64[3, 3]]:
        ...
    
    @overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[3, 3]]:
        ...
    
    def remove(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> bool:
        ...
    
    def __copy__(self) -> Matrix3dVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> Matrix3dVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: Matrix3dVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> Matrix3dVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[3, 3]]:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: Matrix3dVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: Matrix3dVector) -> None:
        ...
    


class Matrix4dVector:
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Matrix4dVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> int:
        ...
    
    @overload
    def extend(self, L: Matrix4dVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        ...
    
    @overload
    def pop(self) -> numpy.ndarray[numpy.float64[4, 4]]:
        ...
    
    @overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[4, 4]]:
        ...
    
    def remove(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> bool:
        ...
    
    def __copy__(self) -> Matrix4dVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> Matrix4dVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: Matrix4dVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> Matrix4dVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[4, 4]]:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: Matrix4dVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: Matrix4dVector) -> None:
        ...
    


class Vector2dVector:
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: numpy.ndarray[numpy.float64]) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Vector2dVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> int:
        ...
    
    @overload
    def extend(self, L: Vector2dVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        ...
    
    @overload
    def pop(self) -> numpy.ndarray[numpy.float64[2, 1]]:
        ...
    
    @overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[2, 1]]:
        ...
    
    def remove(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> bool:
        ...
    
    def __copy__(self) -> Vector2dVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> Vector2dVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: Vector2dVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> Vector2dVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[2, 1]]:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: Vector2dVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: Vector2dVector) -> None:
        ...
    


class Vector2iVector:
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Vector2iVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> int:
        ...
    
    @overload
    def extend(self, L: Vector2iVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        ...
    
    @overload
    def pop(self) -> numpy.ndarray[numpy.int32[2, 1]]:
        ...
    
    @overload
    def pop(self, i: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        ...
    
    def remove(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> bool:
        ...
    
    def __copy__(self) -> Vector2iVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> Vector2iVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: Vector2iVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> Vector2iVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: Vector2iVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: Vector2iVector) -> None:
        ...
    


class Vector3dVector:
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: numpy.ndarray[numpy.float64]) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Vector3dVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> int:
        ...
    
    @overload
    def extend(self, L: Vector3dVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    
    @overload
    def pop(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    
    @overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    
    def remove(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> bool:
        ...
    
    def __copy__(self) -> Vector3dVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> Vector3dVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: Vector3dVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> Vector3dVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: Vector3dVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: Vector3dVector) -> None:
        ...
    


class Vector3iVector:
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Vector3iVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> int:
        ...
    
    @overload
    def extend(self, L: Vector3iVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        ...
    
    @overload
    def pop(self) -> numpy.ndarray[numpy.int32[3, 1]]:
        ...
    
    @overload
    def pop(self, i: int) -> numpy.ndarray[numpy.int32[3, 1]]:
        ...
    
    def remove(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> bool:
        ...
    
    def __copy__(self) -> Vector3iVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> Vector3iVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: Vector3iVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> Vector3iVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.int32[3, 1]]:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: Vector3iVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: Vector3iVector) -> None:
        ...
    


class Vector4iVector:
    @overload
    def __init__(self) -> None:
        ...
    
    @overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Vector4iVector) -> None:
        ...
    
    @overload
    def __init__(self, arg0: Iterable) -> None:
        ...
    
    def append(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        ...
    
    def clear(self) -> None:
        ...
    
    def count(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> int:
        ...
    
    @overload
    def extend(self, L: Vector4iVector) -> None:
        ...
    
    @overload
    def extend(self, L: Iterable) -> None:
        ...
    
    def insert(self, i: int, x: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        ...
    
    @overload
    def pop(self) -> numpy.ndarray[numpy.int32[4, 1]]:
        ...
    
    @overload
    def pop(self, i: int) -> numpy.ndarray[numpy.int32[4, 1]]:
        ...
    
    def remove(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __contains__(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> bool:
        ...
    
    def __copy__(self) -> Vector4iVector:
        ...
    
    def __deepcopy__(self, arg0: dict) -> Vector4iVector:
        ...
    
    @overload
    def __delitem__(self, arg0: int) -> None:
        ...
    
    @overload
    def __delitem__(self, arg0: slice) -> None:
        ...
    
    def __eq__(self, arg0: Vector4iVector) -> bool:
        ...
    
    @overload
    def __getitem__(self, s: slice) -> Vector4iVector:
        ...
    
    @overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.int32[4, 1]]:
        ...
    
    def __iter__(self) -> Iterator:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __ne__(self, arg0: Vector4iVector) -> bool:
        ...
    
    @overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        ...
    
    @overload
    def __setitem__(self, arg0: slice, arg1: Vector4iVector) -> None:
        ...
    


class VerbosityContextManager:
    def __init__(self, level: VerbosityLevel) -> None:
        ...
    
    def __enter__(self) -> None:
        ...
    
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None:
        ...
    


class VerbosityLevel:
    __members__: ClassVar[dict] = ...
    Debug: ClassVar[VerbosityLevel] = ...
    Error: ClassVar[VerbosityLevel] = ...
    Info: ClassVar[VerbosityLevel] = ...
    Warning: ClassVar[VerbosityLevel] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ge__(self, other: object) -> bool:
        ...
    
    def __gt__(self, other: object) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __index__(self) -> int:
        ...
    
    def __int__(self) -> int:
        ...
    
    def __le__(self, other: object) -> bool:
        ...
    
    def __lt__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    @property
    def value(self) -> int:
        ...
    


def get_verbosity_level() -> Any:
    ...

def reset_print_function() -> None:
    ...

def set_verbosity_level(verbosity_level) -> Any:
    ...

