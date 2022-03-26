import numpy as np
import copy
import mps.truncate
from mps import expectation
from .truncation import DEFAULT_TOLERANCE


class TensorArray(object):
    """TensorArray class.

    This class provides the basis for all tensor networks. The class abstracts
    a one-dimensional array of tensors that is freshly copied whenever the
    object is cloned. Two TensorArray's can share the same tensors and be
    destructively modified.

    Attributes:
    size = number of tensors in the array
    """

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

    def tovector(self):
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return _mps2vector(self)

    @staticmethod
    def fromvector(ψ, dimensions, **kwdargs):
        from .factories import vector2mps

        return MPS(vector2mps(ψ, dimensions, **kwdargs))

    def __add__(self, φ):
        """Add an MPS or an MPSList to the MPS.

        Parameters
        ----------
        φ    -- MPS or MPSList object.

        Output
        ------
        mps_list    -- New MPSList.
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
            new_weights = [1, 1]
            new_states = [self, φ]
        elif isinstance(φ, MPSList):
            new_weights = [1] + φ.weights
            new_states = [self] + φ.states
        new_MPSList = MPSList(
            weights=new_weights,
            states=new_states,
            maxsweeps=maxsweeps,
            tolerance=tolerance,
            normalize=self.normalize,
            max_bond_dimension=max_bond_dimension,
        )
        return new_MPSList

    def __sub__(self, φ):
        """Subtract an MPS or an MPSList from the MPS.

        Parameters
        ----------
        φ    -- MPS or MPSList object.

        Output
        ------
        mps_list    -- New MPSList.
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
            new_weights = [1, -1]
            new_states = [self, φ]
        elif isinstance(φ, MPSList):
            new_weights = [1] + list((-1) * np.asarray(φ.weights))
            new_states = [self] + φ.states
        new_MPSList = MPSList(
            weights=new_weights,
            states=new_states,
            maxsweeps=maxsweeps,
            tolerance=tolerance,
            normalize=self.normalize,
            max_bond_dimension=max_bond_dimension,
        )
        return new_MPSList

    def __mul__(self, n):
        """Multiply an MPS quantum state by an scalar n (MPS * n)

        Parameters
        ----------
        n    -- Scalar to multiply the MPS by.

        Output
        ------
        mps    -- New mps.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPS by {n}")
        mps_mult = copy.deepcopy(self)
        mps_mult._data[0] = n * mps_mult._data[0]
        mps_mult._error = np.abs(n) ** 2 * mps_mult._error
        return mps_mult

    def __rmul__(self, n):
        """Multiply an MPS quantum state by an scalar n (n * MPS).

        Parameters
        ----------
        n    -- Scalar to multiply the MPS by.

        Output
        ------
        mps    -- New mps.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPS by {n}")
        mps_mult = copy.deepcopy(self)
        mps_mult._data[0] = n * mps_mult._data[0]
        mps_mult._error = np.abs(n) ** 2 * mps_mult._error
        return mps_mult

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        return expectation.scprod(self, self)

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
        return MPS(data)


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
    Ψ = np.ones((1, 1,))
    D = 1
    for (i, A) in enumerate(data):
        α, d, β = A.shape
        Ψ = np.einsum("Da,akb->Dkb", Ψ, A)
        D = D * d
        Ψ = np.reshape(Ψ, (D, β))
    return Ψ.reshape((Ψ.size,))


class MPSList:
    """MPSList class.
    
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
        """Add an MPS or an MPSList to the MPSList.

        Parameters
        ----------
        φ    -- MPS or MPSList object.

        Output
        ------
        mps_list    -- New MPSList.
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
        elif isinstance(φ, MPSList):
            new_weights = self.weights + φ.weights
            new_states = self.states + φ.states
        new_MPSList = MPSList(
            weights=new_weights,
            states=new_states,
            maxsweeps=maxsweeps,
            tolerance=tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return new_MPSList

    def __sub__(self, φ):
        """Subtract an MPS or an MPSList from the MPSList.

        Parameters
        ----------
        φ    -- MPS or MPSList object.

        Output
        ------
        mps_list    -- New MPSList.
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
        elif isinstance(φ, MPSList):
            new_weights = self.weights + list((-1) * np.asarray(φ.weights))
            new_states = self.states + φ.states
        new_MPSList = MPSList(
            weights=new_weights,
            states=new_states,
            maxsweeps=maxsweeps,
            tolerance=tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return new_MPSList

    def __mul__(self, n):
        """Multiply an MPSList quantum state by an scalar n (MPSList * n)

        Parameters
        ----------
        n    -- Scalar to multiply the MPSList by.

        Output
        ------
        mps    -- New mps.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPSList by {n}")
        new_states = [n * mps for mps in self.states]
        new_MPSList = MPSList(
            weights=self.weights,
            states=new_states,
            maxsweeps=self.maxsweeps,
            tolerance=self.tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return new_MPSList

    def __rmul__(self, n):
        """Multiply an MPSList quantum state by an scalar n (n * MPSList).

        Parameters
        ----------
        n    -- Scalar to multiply the MPSList by.

        Output
        ------
        mps    -- New mps.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPSList by {n}")
        new_states = [n * mps for mps in self.states]
        new_MPSList = MPSList(
            weights=self.weights,
            states=new_states,
            maxsweeps=self.maxsweeps,
            tolerance=self.tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return new_MPSList

    def toMPS(self, normalize=None):
        if normalize is None:
            normalize = self.normalize
        ψ, _ = mps.truncate.combine(
            self.weights,
            self.states,
            maxsweeps=self.maxsweeps,
            tolerance=self.tolerance,
            normalize=normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return ψ
