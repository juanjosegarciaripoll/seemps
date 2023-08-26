from typing import Optional
import numpy as np

class TruncationStrategy:
    DO_NOT_TRUNCATE: int
    RELATIVE_SINGULAR_VALUE: int
    RELATIVE_NORM_SQUARED_ERROR: int

    def __init__(
        self: TruncationStrategy,
        method: int = 1,
        tolerance: float = 1e-8,
        max_bond_dimension: Optional[int] = None,
        normalize: bool = False,
    ):
        pass
    def set_normalization(
        self: TruncationStrategy, normalize: bool
    ) -> TruncationStrategy:
        pass
    def get_tolerance(self) -> float:
        pass
    def get_max_bond_dimension(self) -> int:
        pass
    def get_normalize_flag(self) -> bool:
        pass

DEFAULT_TOLERANCE: float

NO_TRUNCATION: TruncationStrategy

DEFAULT_TRUNCATION: TruncationStrategy

def truncate_vector(s: np.ndarray, strategy: TruncationStrategy):
    pass
