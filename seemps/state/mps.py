import copy
import math
from numbers import Number
from typing import Iterable, Optional, Union
import numpy as np
from .. import expectation
from .schmidt import vector2mps
from .core import DEFAULT_STRATEGY, Strategy
import warnings

Weight = Union[float, complex]


class TensorArray(object):
    """TensorArray class.

    This class provides the basis for all tensor networks. The class abstracts
    a one-dimensional array of tensors that is freshly copied whenever the
    object is cloned. Two TensorArray's can share the same tensors and be
    destructively modified.

    Attributes:
    size = number of tensors in the array
    """

    _data: list[np.ndarray]
    size: int

    def __init__(self, data):
        """Create a new TensorArray from a list of tensors. `data` is an
        iterable object, such as a list or other sequence. The list is cloned
        before storing it into this object, so as to avoid side effects when
        destructively modifying the array."""
        self._data = list(data)
        self.size = len(self._data)

    def __getitem__(self, k):
        #
        # Get MP matrix at position `k`. If 'A' is an MP, we can now
        # do A[k]
        #
        return self._data[k]

    def __setitem__(self, k, value):
        #
        # Replace matrix at position `k` with new tensor `value`. If 'A'
        # is an MP, we can now do A[k] = value
        #
        self._data[k] = value
        return value

    def __copy__(self):
        #
        # Return a copy of the MPS with a fresh new array.
        #
        return type(self)(self._data)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self.size

    def copy(self):
        """Return a fresh new TensorArray that shares the same tensor as its
        sibling, but which can be destructively modified without affecting it.
        """
        return self.__copy__()


class MPS(TensorArray):
    """MPS (Matrix Product State) class.

    This implements a bare-bones Matrix Product State object with open
    boundary conditions. The tensors have three indices, `A[α,i,β]`, where
    `α,β` are the internal labels and `i` is the physical state of the given
    site.

    Parameters
    ----------
    data    -- A list of tensors that form the MPS. The class assumes they
               have three legs and are well formed--i.e. the bond dimensions
               of neighboring sites match.
    error   -- Accumulated error of the simplifications of the MPS.
    maxsweeps, tolerance, normalize, max_bond_dimension -- arguments used by
                 the simplification routine, if simplify is True.
    """

    _error: float
    strategy: Strategy

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(self, data, error=0, strategy: Strategy = DEFAULT_STRATEGY):
        super(MPS, self).__init__(data)
        assert data[0].shape[0] == data[-1].shape[-1] == 1
        self._error = error
        self.strategy = strategy

    def dimension(self) -> int:
        """Return the total size of the Hilbert space in which this MPS lives."""
        return math.prod([a.shape[1] for a in self._data])

    def to_vector(self) -> np.ndarray:
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return _mps2vector(self._data)

    @staticmethod
    def from_vector(ψ: np.ndarray, dimensions: list[int], **kwdargs) -> "MPS":
        return MPS(vector2mps(ψ, dimensions, **kwdargs))

    def __add__(self, φ: Union["MPS", "MPSSum"]) -> "MPSSum":
        """Add an MPS or an MPSSum to the MPS.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        if isinstance(φ, MPS):
            return MPSSum([1, 1], [self, φ], self.strategy)
        if isinstance(φ, MPSSum):
            return MPSSum([1] + φ.weights, [self] + φ.states, self.strategy)
        return NotImplemented

    def __sub__(self, φ: Union["MPS", "MPSSum"]) -> "MPSSum":
        """Subtract an MPS or an MPSSum from the MPS.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        if isinstance(φ, MPS):
            return MPSSum([1, -1], [self, φ], self.strategy)
        if isinstance(φ, MPSSum):
            return MPSSum(
                [1] + list((-1) * np.asarray(φ.weights)),
                [self] + φ.states,
                self.strategy,
            )
        return NotImplemented

    def __mul__(self, n: Weight) -> "MPS":
        """Multiply an MPS quantum state by a scalar n (MPS * n)

        Parameters
        ----------
        n    -- Scalar to multiply the MPS by.

        Output
        ------
        mps    -- New mps.
        """
        if isinstance(n, (float, complex)):
            mps_mult = copy.deepcopy(self)
            mps_mult._data[0] = n * mps_mult._data[0]
            mps_mult._error = np.abs(n) ** 2 * mps_mult._error
            return mps_mult
        return NotImplemented

    def __rmul__(self, n: Weight) -> "MPS":
        """Multiply an MPS quantum state by a scalar n (n * MPS).

        Parameters
        ----------
        n    -- Scalar to multiply the MPS by.

        Output
        ------
        mps    -- New mps.
        """
        if isinstance(n, (float, complex)):
            mps_mult = copy.deepcopy(self)
            mps_mult._data[0] = n * mps_mult._data[0]
            mps_mult._error = np.abs(n) ** 2 * mps_mult._error
            return mps_mult
        return NotImplemented

    def norm2(self) -> float:
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        warnings.warn(
            "method norm2 is deprecated, use norm_squared", category=DeprecationWarning
        )
        return abs(expectation.scprod(self, self))

    def norm_squared(self) -> float:
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        return abs(expectation.scprod(self, self))

    def norm(self) -> float:
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        return np.sqrt(abs(expectation.scprod(self, self)))

    def expectation1(self, operator, n) -> Number:
        """Return the expectation value of `operator` acting on the `n`-th
        site of the MPS. See `mps.expectation.expectation1()`."""
        return expectation.expectation1(self, operator, n)

    def expectation2(self, operator1, operator2, i, j=None) -> Number:
        """Return the expectation value of `operator1` and `operator2` acting
        on the sites `i` and `j`. See `mps.expectation.expectation2()`"""
        return expectation.expectation2(self, operator1, operator2, i, j)

    def all_expectation1(self, operator) -> Number:
        """Return all expectation values of `operator` acting on all possible
        sites of the MPS. See `mps.expectation.all_expectation1()`."""
        return expectation.all_expectation1(self, operator)

    def left_environment(self, site: int) -> np.ndarray:
        ρ = expectation.begin_environment()
        for A in self[:site]:
            ρ = expectation.update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site: int) -> np.ndarray:
        ρ = expectation.begin_environment()
        for A in self[-1:site:-1]:
            ρ = expectation.update_right_environment(A, A, ρ)
        return ρ

    def error(self) -> float:
        """Return any recorded norm-2 truncation errors in this state. More
        precisely, ‖exact - actual‖^2."""
        return self._error

    def update_error(self, delta: float) -> float:
        """Update an estimate of the norm-2 truncation errors. We use the
        triangle inequality to guess an upper bound."""
        self._error = (np.sqrt(self._error) + np.sqrt(delta)) ** 2
        return self._error

    # TODO: We have to change the signature and working of this function, so that
    # 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
    # needed. In this case, 'dimensions' will only list the dimensions of the added
    # sites, not all of them.
    def extend(
        self,
        L: int,
        sites: Optional[Iterable[int]] = None,
        dimensions: Union[int, list[int]] = 2,
    ):
        """Enlarge an MPS so that it lives in a Hilbert space with `L` sites.

        Parameters
        ----------
        L          -- The new size
        dimensions -- If it is an integer, it is the dimension of the new sites.
                      If it is a list, it is the dimension of all sites.
        sites      -- Where to place the tensors of the original MPO.

        Output
        ------
        mpo        -- A new MPO.
        """
        assert L >= self.size
        if isinstance(dimensions, int):
            final_dimensions = [dimensions] * L
        else:
            final_dimensions = dimensions.copy()
        if sites is None:
            sites = range(self.size)

        data: list[np.ndarray] = [np.ndarray(())] * L
        for ndx, A in zip(sites, self):
            data[ndx] = A
            final_dimensions[ndx] = A.shape[1]
        D = 1
        for i, A in enumerate(data):
            if A.size == 0:
                d = final_dimensions[i]
                A = np.zeros((D, d, D))
                A[:, 0, :] = np.eye(D)
                data[i] = A
            else:
                D = A.shape[-1]
        return MPS(data, strategy=self.strategy)


def _mps2vector(data: list[np.ndarray]) -> np.ndarray:
    #
    # Input:
    #  - data: list of tensors for the MPS (unchecked)
    # Output:
    #  - Ψ: Vector of complex numbers with all the wavefunction amplitudes
    #
    # We keep Ψ[D,β], a tensor with all matrices contracted so far, where
    # 'D' is the dimension of the physical subsystems up to this point and
    # 'β' is the last uncontracted internal index.
    #
    Ψ: np.ndarray = np.ones(1)
    for A in reversed(data):
        α, d, β = A.shape
        # Ψ = np.einsum("Da,akb->Dkb", Ψ, A)
        Ψ = np.dot(A.reshape(α * d, β), Ψ).reshape(α, -1)
    return Ψ.reshape(-1)


class MPSSum:
    """MPSSum class.

    Stores the MPS as a list  for its future combination when an MPO acts on it.

    Parameters
    ----------
    weights    -- weights of the linear combination of MPS.
    states    --  states of the linear combination of MPS.
    maxsweeps, tolerance, normalize, max_bond_dimension -- arguments used by
                 the simplification routine, if simplify is True.
    """

    weights: list[Weight]
    states: list[MPS]
    strategy: Strategy

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(
        self,
        weights: list[Weight],
        states: list[MPS],
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.weights = weights
        self.states = states
        self.strategy = strategy

    def __add__(self, φ: Union[MPS, "MPSSum"]) -> "MPSSum":
        """Add an MPS or an MPSSum to the MPSSum.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        if isinstance(φ, MPS):
            return MPSSum(
                self.weights + [1.0],
                self.states + [φ],
                self.strategy,
            )
        elif isinstance(φ, MPSSum):
            return MPSSum(
                self.weights + φ.weights,
                self.states + φ.states,
                self.strategy,
            )
        return NotImplemented

    def __sub__(self, φ: Union[MPS, "MPSSum"]) -> "MPSSum":
        """Subtract an MPS or an MPSSum from the MPSSum.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        if isinstance(φ, MPS):
            return MPSSum(self.weights + [-1], self.states + [φ], self.strategy)
        if isinstance(φ, MPSSum):
            return MPSSum(
                self.weights + [-w for w in φ.weights],
                self.states + φ.states,
                self.strategy,
            )
        return NotImplemented

    def __mul__(self, n: Weight) -> "MPSSum":
        """Multiply an MPSSum quantum state by an scalar n (MPSSum * n)

        Parameters
        ----------
        n    -- Scalar to multiply the MPSSum by.

        Output
        ------
        mps    -- New mps.
        """
        if isinstance(n, (float, complex)):
            return MPSSum([n * w for w in self.weights], self.states, self.strategy)
        return NotImplemented

    def __rmul__(self, n: Weight) -> "MPSSum":
        """Multiply an MPSSum quantum state by an scalar n (n * MPSSum).

        Parameters
        ----------
        n    -- Scalar to multiply the MPSSum by.

        Output
        ------
        mps    -- New mps.
        """
        if isinstance(n, (float, complex)):
            return MPSSum([n * w for w in self.weights], self.states, self.strategy)
        return NotImplemented

    def to_vector(self) -> np.ndarray:
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return np.asarray(
            sum(wa * A.to_vector() for wa, A in zip(self.weights, self.states))
        )

    def toMPS(
        self, normalize: Optional[bool] = None, strategy: Optional[Strategy] = None
    ) -> MPS:
        from ..truncate.combine import combine

        if strategy is None:
            strategy = self.strategy
        ψ, _ = combine(
            self.weights,
            self.states,
            maxsweeps=strategy.get_max_sweeps(),
            tolerance=strategy.get_tolerance(),
            normalize=strategy.get_normalize_flag() if normalize is None else normalize,
            max_bond_dimension=strategy.get_max_bond_dimension(),
        )
        return ψ
