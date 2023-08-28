from typing import Optional
import numpy as np

class Truncation:
    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2

class Strategy:
    DO_NOT_TRUNCATE: int
    RELATIVE_SINGULAR_VALUE: int
    RELATIVE_NORM_SQUARED_ERROR: int

    def __init__(
        self: Strategy,
        method: int = 1,
        tolerance: float = 1e-8,
        max_bond_dimension: Optional[int] = None,
        max_sweeps: int = 16,
        normalize: bool = False,
        simplify: bool = False,
    ):
        pass
    def replace(
        self: Strategy,
        method: Optional[int] = None,
        tolerance: Optional[float] = None,
        max_bond_dimension: Optional[int] = None,
        max_sweeps: Optional[int] = None,
        normalize: Optional[bool] = None,
        simplify: Optional[bool] = None,
    ) -> Strategy:
        pass
    def set_normalization(self: Strategy, normalize: bool) -> Strategy:
        pass
    def get_tolerance(self) -> float:
        pass
    def get_max_bond_dimension(self) -> int:
        pass
    def get_max_sweeps(self) -> int:
        pass
    def get_normalize_flag(self) -> bool:
        pass
    def get_simplify_flag(self) -> bool:
        pass

DEFAULT_TOLERANCE: float

NO_TRUNCATION: Strategy

DEFAULT_STRATEGY: Strategy

def truncate_vector(s: np.ndarray, strategy: Strategy):
    pass
