import copy
import numpy as np
from .array import TensorArray
from .. import expectation
from .schmidt import vector2mps
from .truncation import DEFAULT_TOLERANCE
import warnings


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

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(
        self,
        data,
        error=0,
        maxsweeps=16,
        tolerance=DEFAULT_TOLERANCE,
        normalize=False,
        max_bond_dimension=None,
    ):
        super(MPS, self).__init__(data)
        assert data[0].shape[0] == data[-1].shape[-1] == 1
        self._error = error
        self.maxsweeps = maxsweeps
        self.tolerance = tolerance
        self.normalize = normalize
        self.max_bond_dimension = max_bond_dimension

    def dimension(self):
        """Return the total size of the Hilbert space in which this MPS lives."""
        return np.product([a.shape[1] for a in self._data])

    def to_vector(self):
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return _mps2vector(self)

    @staticmethod
    def from_vector(ψ, dimensions, **kwdargs):
        return MPS(vector2mps(ψ, dimensions, **kwdargs))

    def __add__(self, φ):
        """Add an MPS or an MPSSum to the MPS.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        if isinstance(φ, MPS):
            new_weights = [1, 1]
            new_states = [self, φ]
        elif isinstance(φ, MPSSum):
            new_weights = [1] + list((1) * np.asarray(φ.weights))
            new_states = [self] + φ.states
        else:
            raise TypeError(f"Invalid addition between MPS and object of type {φ}")
        if self.max_bond_dimension is None:
            max_bond_dimension = φ.max_bond_dimension
        elif φ.max_bond_dimension is None:
            max_bond_dimension = self.max_bond_dimension
        else:
            max_bond_dimension = min(self.max_bond_dimension, φ.max_bond_dimension)
        return MPSSum(
            weights=new_weights,
            states=new_states,
            maxsweeps=min(self.maxsweeps, φ.maxsweeps),
            tolerance=min(self.tolerance, φ.tolerance),
            normalize=self.normalize,
            max_bond_dimension=max_bond_dimension,
        )

    def __sub__(self, φ):
        """Subtract an MPS or an MPSSum from the MPS.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        if isinstance(φ, MPS):
            new_weights = [1, -1]
            new_states = [self, φ]
        elif isinstance(φ, MPSSum):
            new_weights = [1] + list((-1) * np.asarray(φ.weights))
            new_states = [self] + φ.states
        else:
            raise TypeError(f"Invalid subtraction between MPS and object of type {φ}")
        if self.max_bond_dimension is None:
            max_bond_dimension = φ.max_bond_dimension
        elif φ.max_bond_dimension is None:
            max_bond_dimension = self.max_bond_dimension
        else:
            max_bond_dimension = min(self.max_bond_dimension, φ.max_bond_dimension)
        return MPSSum(
            weights=new_weights,
            states=new_states,
            maxsweeps=min(self.maxsweeps, φ.maxsweeps),
            tolerance=min(self.tolerance, φ.tolerance),
            normalize=self.normalize,
            max_bond_dimension=max_bond_dimension,
        )

    def __mul__(self, n):
        """Multiply an MPS quantum state by a scalar n (MPS * n)

        Parameters
        ----------
        n    -- Scalar to multiply the MPS by.

        Output
        ------
        mps    -- New mps.
        """
        if np.isscalar(n):
            mps_mult = copy.deepcopy(self)
            mps_mult._data[0] = n * mps_mult._data[0]
            mps_mult._error = np.abs(n) ** 2 * mps_mult._error
            return mps_mult
        else:
            raise TypeError(
                f"Invalid multiplication between MPS and object of type {n}"
            )

    def __rmul__(self, n):
        """Multiply an MPS quantum state by a scalar n (n * MPS).

        Parameters
        ----------
        n    -- Scalar to multiply the MPS by.

        Output
        ------
        mps    -- New mps.
        """
        if np.isscalar(n):
            mps_mult = copy.deepcopy(self)
            mps_mult._data[0] = n * mps_mult._data[0]
            mps_mult._error = np.abs(n) ** 2 * mps_mult._error
            return mps_mult
        else:
            raise TypeError(
                f"Invalid multiplication between MPS and object of type {n}"
            )

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        warnings.warn(
            "method norm2 is deprecated, use norm_squared", category=DeprecationWarning
        )
        return abs(expectation.scprod(self, self))

    def norm_squared(self):
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        return abs(expectation.scprod(self, self))

    def norm(self):
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        return np.sqrt(abs(expectation.scprod(self, self)))

    def expectation1(self, operator, n):
        """Return the expectation value of `operator` acting on the `n`-th
        site of the MPS. See `mps.expectation.expectation1()`."""
        return expectation.expectation1(self, operator, n)

    def expectation2(self, operator1, operator2, i, j=None):
        """Return the expectation value of `operator1` and `operator2` acting
        on the sites `i` and `j`. See `mps.expectation.expectation2()`"""
        return expectation.expectation2(self, operator1, operator2, i, j)

    def all_expectation1(self, operator):
        """Return all expectation values of `operator` acting on all possible
        sites of the MPS. See `mps.expectation.all_expectation1()`."""
        return expectation.all_expectation1(self, operator)

    def left_environment(self, site):
        ρ = expectation.begin_environment()
        for A in self[:site]:
            ρ = expectation.update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site):
        ρ = expectation.begin_environment()
        for A in self[-1:site:-1]:
            ρ = expectation.update_right_environment(A, A, ρ)
        return ρ

    def error(self):
        """Return any recorded norm-2 truncation errors in this state. More
        precisely, ‖exact - actual‖^2."""
        return self._error

    def update_error(self, delta):
        """Update an estimate of the norm-2 truncation errors. We use the
        triangle inequality to guess an upper bound."""
        self._error = (np.sqrt(self._error) + np.sqrt(delta)) ** 2
        return self._error

    def extend(self, L, sites=None, dimensions=2):
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
        if np.isscalar(dimensions):
            dimensions = [dimensions] * L
        if sites is None:
            sites = range(self.size)
        else:
            assert len(sites) == self.size

        data = [None] * L
        for (ndx, A) in zip(sites, self):
            data[ndx] = A
            dimensions[ndx] = A.shape[1]
        D = 1
        for (i, A) in enumerate(data):
            if A is None:
                d = dimensions[i]
                A = np.zeros((D, d, D))
                A[:, 0, :] = np.eye(D)
                data[i] = A
            else:
                D = A.shape[-1]
        return MPS(
            data,
            maxsweeps=self.maxsweeps,
            tolerance=self.tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )


def _mps2vector(data):
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
    Ψ = np.ones((1, 1))
    D = 1
    for A in data:
        α, d, β = A.shape
        # Ψ = np.einsum("Da,akb->Dkb", Ψ, A)
        Ψ = np.dot(Ψ, A.reshape(α, d * β))
        D = D * d
        Ψ = Ψ.reshape(D, β)
    return Ψ.flatten()


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

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(
        self,
        weights,
        states,
        maxsweeps=16,
        tolerance=DEFAULT_TOLERANCE,
        normalize=False,
        max_bond_dimension=None,
    ):
        self.weights = weights
        self.states = states
        self.maxsweeps = maxsweeps
        self.tolerance = tolerance
        self.normalize = normalize
        self.max_bond_dimension = max_bond_dimension

    def __add__(self, φ):
        """Add an MPS or an MPSSum to the MPSSum.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        maxsweeps = min(self.maxsweeps, φ.maxsweeps)
        tolerance = min(self.tolerance, φ.tolerance)
        if self.max_bond_dimension is None:
            max_bond_dimension = φ.max_bond_dimension
        elif φ.max_bond_dimension is None:
            max_bond_dimension = self.max_bond_dimension
        else:
            max_bond_dimension = min(self.max_bond_dimension, φ.max_bond_dimension)
        if isinstance(φ, MPS):
            new_weights = self.weights + [1]
            new_states = self.states + [φ]
        elif isinstance(φ, MPSSum):
            new_weights = self.weights + φ.weights
            new_states = self.states + φ.states
        return MPSSum(
            weights=new_weights,
            states=new_states,
            maxsweeps=maxsweeps,
            tolerance=tolerance,
            normalize=self.normalize,
            max_bond_dimension=max_bond_dimension,
        )

    def __sub__(self, φ):
        """Subtract an MPS or an MPSSum from the MPSSum.

        Parameters
        ----------
        φ    -- MPS or MPSSum object.

        Output
        ------
        mps_list    -- New MPSSum.
        """
        maxsweeps = min(self.maxsweeps, φ.maxsweeps)
        tolerance = min(self.tolerance, φ.tolerance)
        if self.max_bond_dimension is None:
            max_bond_dimension = φ.max_bond_dimension
        elif φ.max_bond_dimension is None:
            max_bond_dimension = self.max_bond_dimension
        else:
            max_bond_dimension = min(self.max_bond_dimension, φ.max_bond_dimension)
        if isinstance(φ, MPS):
            new_weights = self.weights + [-1]
            new_states = self.states + [φ]
        elif isinstance(φ, MPSSum):
            new_weights = self.weights + list((-1) * np.asarray(φ.weights))
            new_states = self.states + φ.states
        return MPSSum(
            weights=new_weights,
            states=new_states,
            maxsweeps=maxsweeps,
            tolerance=tolerance,
            normalize=self.normalize,
            max_bond_dimension=max_bond_dimension,
        )

    def __mul__(self, n):
        """Multiply an MPSSum quantum state by an scalar n (MPSSum * n)

        Parameters
        ----------
        n    -- Scalar to multiply the MPSSum by.

        Output
        ------
        mps    -- New mps.
        """
        if np.isscalar(n):
            return MPSSum(
                weights=[n * w for w in self.weights],
                states=self.states,
                maxsweeps=self.maxsweeps,
                tolerance=self.tolerance,
                normalize=self.normalize,
                max_bond_dimension=self.max_bond_dimension,
            )
        raise Exception(f"Cannot multiply MPSSum by {n}")

    def __rmul__(self, n):
        """Multiply an MPSSum quantum state by an scalar n (n * MPSSum).

        Parameters
        ----------
        n    -- Scalar to multiply the MPSSum by.

        Output
        ------
        mps    -- New mps.
        """
        if np.isscalar(n):
            return MPSSum(
                weights=[n * w for w in self.weights],
                states=self.states,
                maxsweeps=self.maxsweeps,
                tolerance=self.tolerance,
                normalize=self.normalize,
                max_bond_dimension=self.max_bond_dimension,
            )
        raise Exception(f"Cannot multiply MPSSum by {n}")

    def to_vector(self):
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return sum(wa * A.to_vector() for wa, A in zip(self.weights, self.states))

    def toMPS(
        self, normalize=None, tolerance=None, maxsweeps=16, max_bond_dimension=None
    ):
        from ..truncate.combine import combine

        if normalize is None:
            normalize = self.normalize
        if tolerance is None:
            tolerance = self.tolerance
        if maxsweeps is None:
            maxsweeps = self.maxsweeps
        if max_bond_dimension is None:
            max_bond_dimension = self.max_bond_dimension
        ψ, _ = combine(
            self.weights,
            self.states,
            maxsweeps=maxsweeps,
            tolerance=tolerance,
            normalize=normalize,
            max_bond_dimension=max_bond_dimension,
        )
        return ψ
