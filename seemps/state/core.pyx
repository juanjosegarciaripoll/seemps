import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from libcpp cimport bool

cdef class TruncationStrategy:
    cdef double tolerance
    cdef int max_bond_dimension
    cdef bool normalize

    def __init__(self,
                 tolerance: float = np.finfo(np.float64).eps,
                 max_bond_dimension: Optional[int] = None,
                 normalize: bool = False):
        if max_bond_dimension is None:
            max_bond_dimension = np.iinfo(int).max
        self.tolerance = tolerance
        self.max_bond_dimension = int(max_bond_dimension)
        self.normalize = normalize

    def set_normalization(self, normalize: bool) -> TruncationStrategy:
        return TruncationStrategy(self.tolerance, self.max_bond_dimension, normalize)

    def get_tolerance(self) -> float:
        return self.tolerance

    def get_max_bond_dimension(self) -> int:
        return self.max_bond_dimension

    def get_normalize_flag(self) -> bool:
        return self.normalize

DEFAULT_TOLERANCE = TruncationStrategy(tolerance = np.finfo(np.float64).eps,
                                       max_bond_dimension = np.iinfo(int).max,
                                       normalize = False)

NO_TRUNCATION = TruncationStrategy(tolerance = 0.0,
                                   max_bond_dimension = np.iinfo(int).max,
                                   normalize = False)

cdef cnp.float64_t[::1] errors = np.zeros(128, dtype=np.float64)

def truncate_vector(cnp.ndarray[cnp.float64_t, ndim=1] s,
                    TruncationStrategy strategy = DEFAULT_TOLERANCE) -> cnp.ndarray[cnp.float64_t]:
    global errors

    cdef:
        Py_ssize_t N, i, ndx
        double total, max_error, new_norm

    if strategy.tolerance == 0.0:
        return s, 0.0
    N = s.size
    if errors.size <= N:
        errors = np.empty(2*N)
    #
    # Compute the cumulative sum of the reduced density matrix eigen values
    # in reversed order. Thus errors[i] is the error we make when we drop
    # i singular values.
    #
    total = 0.0
    j = N-1
    for i in range(N):
        errors[i] = total
        total += s[j]**2
        j -= 1
    errors[N] = total

    max_error = total * strategy.tolerance
    ndx = 1
    final_error = 0.0
    for i in range(N):
        if errors[i] >= max_error:
            i -= 1
            break
    final_size = min(N - i, strategy.max_bond_dimension)
    if strategy.normalize:
        new_norm = sqrt(total - errors[i])
        s = s[:final_size] / new_norm
        return s, errors[i]
    return s[:final_size], errors[i]
