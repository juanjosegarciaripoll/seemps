import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from libcpp cimport bool

cdef class TruncationStrategy:
    cdef int method
    cdef double tolerance
    cdef int max_bond_dimension
    cdef bool normalize

    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2

    def __init__(self,
                 method: int = RELATIVE_SINGULAR_VALUE,
                 tolerance: float = 1e-8,
                 max_bond_dimension: Optional[int] = None,
                 normalize: bool = False):
        if max_bond_dimension is None:
            max_bond_dimension = np.iinfo(int).max
        if self.method < 0 or self.method > 2:
            raise AssertionError("Invalid method argument passed to TruncationStrategy")
        self.method = method
        if self.tolerance < 0 or self.tolerance >= 1.0:
            raise AssertionError("Invalid tolerance argument passed to TruncationStrategy")
        self.tolerance = tolerance
        max_bond_dimension = int(max_bond_dimension)
        if max_bond_dimension <= 0:
            raise AssertionError("Invalid bond dimension in TruncationStrategy")
        self.max_bond_dimension = max_bond_dimension
        self.normalize = normalize

    def set_normalization(self, normalize: bool) -> TruncationStrategy:
        return TruncationStrategy(self.tolerance, self.max_bond_dimension, normalize)

    def get_tolerance(self) -> float:
        return self.tolerance

    def get_max_bond_dimension(self) -> int:
        return self.max_bond_dimension

    def get_normalize_flag(self) -> bool:
        return self.normalize

DEFAULT_TOLERANCE = TruncationStrategy(method = TruncationStrategy.RELATIVE_NORM_SQUARED_ERROR,
                                       tolerance = np.finfo(np.float64).eps,
                                       max_bond_dimension = np.iinfo(int).max,
                                       normalize = False)

NO_TRUNCATION = TruncationStrategy(method = TruncationStrategy.DO_NOT_TRUNCATE)

cdef cnp.float64_t[::1] errors_buffer = np.zeros(1024, dtype=np.float64)

cdef cnp.float64_t *get_errors_buffer(Py_ssize_t N) noexcept:
    global errors_buffer
    if errors_buffer.size <= N:
        errors_buffer = np.zeros(2 * N, dtype=np.float64)
    return &errors_buffer[0]

def truncate_vector(cnp.ndarray[cnp.float64_t, ndim=1] s,
                    TruncationStrategy strategy):
    global errors_buffer

    cdef:
        Py_ssize_t i, j, ndx, N = s.size
        double total, max_error, new_norm
        cnp.float64_t *errors = get_errors_buffer(N)
        cnp.float64_t *data

    if strategy.method == 0:
        max_error = 0.0
    elif strategy.method == 1:
        data = &s[0]
        max_error = strategy.tolerance * data[0]
        for i in range(N):
            if data[i] <= max_error:
                break
        if i == 0:
            i = 1
        elif i > strategy.max_bond_dimension:
            i = strategy.max_bond_dimension
        final_size = i
        max_error = 0.0
        while i < N:
            max_error += data[i] * data[i]
            i += i
        s = s[:final_size]
    else:
        #
        # Compute the cumulative sum of the reduced density matrix eigen values
        # in reversed order. Thus errors[i] is the error we make when we drop
        # i singular values.
        #
        total = 0.0
        data = &s[N-1]
        for i in range(N):
            errors[i] = total
            total += data[0]*data[0]
            data -= 1
        errors[N] = total

        max_error = total * strategy.tolerance
        ndx = 1
        final_error = 0.0
        for i in range(N):
            if errors[i] >= max_error:
                i -= 1
                break
        final_size = min(N - i, strategy.max_bond_dimension)
        max_error = errors[i]
        if final_size < N:
            s = s[:final_size]
        if strategy.normalize:
            new_norm = sqrt(total - max_error)
            data = &s[0]
            for i in range(final_size):
                data[i] /= new_norm
    return s, max_error