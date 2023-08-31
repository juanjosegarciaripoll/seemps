import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy.sparse  # type: ignore
from numbers import Complex, Real
from typing import Sequence, Iterator, Iterable, Optional, Union, overload, TypeVar

Weight = Union[float, complex]
Unitary = NDArray

Operator = Union[NDArray, scipy.sparse.sparray]
DenseOperator = NDArray
Vector = NDArray
VectorLike = ArrayLike
Tensor3 = NDArray
Tensor4 = NDArray
Environment = NDArray

MPSLike = Sequence[Tensor3]
