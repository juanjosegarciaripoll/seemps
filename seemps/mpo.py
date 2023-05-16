import numpy as np
import copy
from .state import MPS, MPSSum, TensorArray, DEFAULT_TOLERANCE
from .truncate import simplify, combine
from .tools import log


def mpo_multiply_tensor(A, B):
    C = np.einsum("aijb,cjd->acibd", A, B)
    s = C.shape
    return C.reshape(s[0] * s[1], s[2], s[3] * s[4])


class MPO(TensorArray):
    """MPO (Matrix Product Operator) class.

    This implements a bare-bones Matrix Product Operator object with open
    boundary conditions. The tensors have four indices, A[α,i,j,β], where
    'α,β' are the internal labels and 'i,j' the physical indices ar the given
    site.

    Parameters
    ----------
    data      -- A list of the tensors that form the MPO
    simplify  -- Use the simplification algorithm after applying the MPO
                 Defaults to False
    maxsweeps, tolerance, normalize, max_bond_dimension -- arguments used by
                 the simplification routine, if simplify is True.
    """

    __array_priority__ = 10000

    def __init__(
        self,
        data,
        simplify=False,
        maxsweeps=16,
        tolerance=DEFAULT_TOLERANCE,
        normalize=False,
        max_bond_dimension=None,
    ):
        super(MPO, self).__init__(data)
        assert data[0].shape[0] == data[-1].shape[-1] == 1
        self.maxsweeps = maxsweeps
        self.tolerance = tolerance
        self.normalize = normalize
        self.max_bond_dimension = max_bond_dimension
        self.simplify = simplify

    def __add__(self, A):
        """Add an MPO, MPOList or MPOSum to the MPO.

        Parameters
        ----------
        A    -- MPO, MPOList or MPOSum object.

        Output
        ------
        mpo_sum    -- New MPOSum.
        """
        if isinstance(A, MPO):
            new_weights = [1, 1]
            new_mpos = [self, A]
        elif isinstance(A, MPOList):
            new_weights = [1, 1]
            new_mpos = [self, A]
        elif isinstance(A, MPOSum):
            new_weights = [1] + A.weights
            new_mpos = [self] + A.mpos
        new_MPOSum = MPOSum(mpos=new_mpos, weights=new_weights)
        return new_MPOSum

    def __sub__(self, A):
        """Add an MPO or an MPOSum to the MPO.

        Parameters
        ----------
        A    -- MPO or MPOSum object.

        Output
        ------
        mpo_sum    -- New MPOSum.
        """
        if isinstance(A, MPO):
            new_weights = [1, -1]
            new_mpos = [self, A]
        elif isinstance(A, MPOList):
            new_weights = [1, -1]
            new_mpos = [self, A]
        elif isinstance(A, MPOSum):
            new_weights = [1] + list((-1) * np.asarray(A.weights))
            new_mpos = [self] + A.mpos
        new_MPOSum = MPOSum(mpos=new_mpos, weights=new_weights)
        return new_MPOSum

    def __mul__(self, n):
        """Multiply an MPO quantum state by an scalar n (MPO * n)

        Parameters
        ----------
        n          -- Scalar to multiply the MPO by.

        Output
        ------
        mpo -- New mpo.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPO by {n}")
        mpo_mult = copy.deepcopy(self)
        mpo_mult._data[0] = n * mpo_mult._data[0]
        return mpo_mult

    def __rmul__(self, n):
        """Multiply an MPO quantum state by an scalar n (n * MPO).

        Parameters
        ----------
        n          -- Scalar to multiply the MPO by.

        Output
        ------
        mpo -- New mpo.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPO by {n}")
        mpo_mult = copy.deepcopy(self)
        mpo_mult._data[0] = n * mpo_mult._data[0]
        return mpo_mult

    def dimensions(self):
        """Return the local dimensions of the MPO."""
        return [A.shape[1] for A in self._data]

    def tomatrix(self):
        """Return the matrix representation of this MPO."""
        D = 1  # Total physical dimension so far
        out = np.array([[[1.0]]])
        for A in self._data:
            _, i, _, b = A.shape
            out = np.einsum("lma,aijb->limjb", out, A)
            D *= i
            out = out.reshape(D, D, b)
        return out[:, :, 0]

    def apply(self, b):
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        if isinstance(b, MPSSum):
            b = b.toMPS()
        if isinstance(b, MPS):
            assert self.size == b.size
            log(f"Total error before applying MPO {b.error()}")
            err = 0.0
            b = MPS(
                [mpo_multiply_tensor(A, B) for A, B in zip(self._data, b)],
                error=b.error(),
            )
            if self.simplify:
                b, err, _ = simplify(
                    b,
                    maxsweeps=self.maxsweeps,
                    tolerance=self.tolerance,
                    normalize=self.normalize,
                    max_bond_dimension=self.max_bond_dimension,
                )
            log(f"Total error after applying MPO {b.error()}, incremented by {err}")
            return b
        else:
            raise Exception(f"Cannot multiply MPO with {b}")

    def __matmul__(self, b):
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        return self.apply(b)

    def extend(self, L, sites=None, dimensions=2):
        """Enlarge an MPO so that it acts on a larger Hilbert space with 'L' sites.

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
        D = 1
        for (i, A) in enumerate(data):
            if A is None:
                d = dimensions[i]
                A = np.eye(D).reshape(D, 1, 1, D) * np.eye(d).reshape(1, d, d, 1)
                data[i] = A
            else:
                D = A.shape[-1]
        return MPO(
            data,
            simplify=self.simplify,
            tolerance=self.tolerance,
            normalize=self.normalize,
            maxsweeps=self.maxsweeps,
            max_bond_dimension=self.max_bond_dimension,
        )


class MPOList(object):
    """MPO (Matrix Product Operator) list.

    This implements a list of MPOs that are applied sequentially.

    Parameters
    ----------
    mpos  -- A list of the MPOs
    simplify  -- Use the simplification algorithm after applying the MPO
                 Defaults to False
    maxsweeps, tolerance, normalize, max_bond_dimension -- arguments used by
                 the simplification routine, if simplify is True.
    """

    __array_priority__ = 10000

    def __init__(
        self,
        mpos,
        simplify=False,
        maxsweeps=16,
        tolerance=DEFAULT_TOLERANCE,
        normalize=False,
        max_bond_dimension=None,
    ):
        self.mpos = mpos
        self.maxsweeps = maxsweeps
        self.tolerance = tolerance
        self.normalize = normalize
        self.max_bond_dimension = max_bond_dimension
        self.simplify = simplify

    def __add__(self, A):
        """Add an MPO, MPOList or MPOSum to the MPO.

        Parameters
        ----------
        A    -- MPO, MPOList or MPOSum object.

        Output
        ------
        mpo_sum    -- New MPOSum.
        """
        if isinstance(A, MPO):
            new_weights = [1, 1]
            new_mpos = [self, A]
        elif isinstance(A, MPOList):
            new_weights = [1, 1]
            new_mpos = [self, A]
        elif isinstance(A, MPOSum):
            new_weights = [1] + A.weights
            new_mpos = [self] + A.mpos
        new_MPOSum = MPOSum(mpos=new_mpos, weights=new_weights)
        return new_MPOSum

    def __sub__(self, A):
        """Add an MPO or an MPOSum to the MPO.

        Parameters
        ----------
        A    -- MPO or MPOSum object.

        Output
        ------
        mpo_sum    -- New MPOSum.
        """
        if isinstance(A, MPO):
            new_weights = [1, -1]
            new_mpos = [self, A]
        elif isinstance(A, MPOList):
            new_weights = [1, -1]
            new_mpos = [self, A]
        elif isinstance(A, MPOSum):
            new_weights = [1] + list((-1) * np.asarray(A.weights))
            new_mpos = [self] + A.mpos
        new_MPOSum = MPOSum(mpos=new_mpos, weights=new_weights)
        return new_MPOSum

    def __mul__(self, n):
        """Multiply an MPOList quantum state by an scalar n (MPOList * n).

        Parameters
        ----------
        n          -- Scalar to multiply the MPOList by.

        Output
        ------
        mpo -- New mpo.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPOList by {n}")
        if isinstance(self.mpos[0], MPOSum):
            mpo_mult = copy.deepcopy(self)
            mpo_mult.mpos[0] = n * mpo_mult.mpos[0]
        else:
            mpo_mult = copy.deepcopy(self)
            mpo_mult.mpos[0]._data[0] = n * mpo_mult.mpos[0]._data[0]
        return mpo_mult

    def __rmul__(self, n):
        """Multiply an MPOList quantum state by an scalar n (n * MPOList).

        Parameters
        ----------
        n          -- Scalar to multiply the MPOList by.

        Output
        ------
        mpo -- New mpo.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPOList by {n}")
        if isinstance(self.mpos[0], MPOSum):
            mpo_mult = copy.deepcopy(self)
            mpo_mult.mpos[0] = n * mpo_mult.mpos[0]
        else:
            mpo_mult = copy.deepcopy(self)
            mpo_mult.mpos[0]._data[0] = n * mpo_mult.mpos[0]._data[0]
        return mpo_mult

    def tomatrix(self):
        """Return the matrix representation of this MPO."""
        A = self.mpos[0].tomatrix()
        for mpo in self.mpos[1:]:
            A = A @ mpo.tomatrix()
        return A

    def apply(self, b):
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        if isinstance(b, MPSSum):
            b = b.toMPS()
        for mpo in self.mpos:
            # log(f'Total error before applying MPOList {b.error()}')
            b = mpo.apply(b)
            err = 0.0
            if self.simplify and not mpo.simplify:
                b, err, _ = simplify(
                    b,
                    maxsweeps=self.maxsweeps,
                    tolerance=self.tolerance,
                    normalize=self.normalize,
                    max_bond_dimension=self.max_bond_dimension,
                )
            log(f"Total error after applying MPOList {b.error()}, incremented by {err}")
        return b

    def __matmul__(self, b):
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        return self.apply(b)

    def extend(self, L, sites=None, dimensions=2):
        """Enlarge an MPOList so that it acts on a larger Hilbert space with 'L' sites.

        Parameters
        ----------
        L          -- The new size
        dimensions -- If it is an integer, it is the dimension of the new sites.
                      If it is a list, it is the dimension of all sites.
        sites      -- Where to place the tensors of the original MPO.

        Output
        ------
        mpo        -- A new MPOList.
        """
        data = [mpo.extend(L, sites=sites, dimensions=dimensions) for mpo in self.mpos]
        return MPOList(
            data,
            simplify=self.simplify,
            tolerance=self.tolerance,
            normalize=self.normalize,
            maxsweeps=self.maxsweeps,
            max_bond_dimension=self.max_bond_dimension,
        )


class MPOSum(object):
    """MPO (Matrix Product Operator) sum.

    This a sum of MPOs acting on an MPS.

    Parameters
    ----------
    mpos  -- A list of the MPOs.
    weights    -- A list of the scalars multiplying each MPO.
    simplify  -- Use the simplification algorithm after applying the MPO
                 Defaults to False.
    maxsweeps, tolerance, normalize, max_bond_dimension -- arguments used by
                 the simplification routine, if simplify is True.
    """

    __array_priority__ = 10000

    def __init__(
        self,
        mpos,
        weights=None,
        simplify=False,
        maxsweeps=16,
        tolerance=DEFAULT_TOLERANCE,
        normalize=False,
        max_bond_dimension=None,
    ):
        self.mpos = mpos
        if weights is None:
            weights = list(np.ones(len(self.mpos)))
        self.weights = weights
        self.maxsweeps = maxsweeps
        self.tolerance = tolerance
        self.normalize = normalize
        self.max_bond_dimension = max_bond_dimension
        self.simplify = simplify

    def __add__(self, A):
        """Add an MPO or an MPOSum from the MPOSum.

        Parameters
        ----------
        A    -- MPO or MPOSum object.

        Output
        ------
        mpo_sum    -- New MPOSum.
        """
        if isinstance(A, MPO):
            new_weights = self.weights + [1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOList):
            new_weights = self.weights + [1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOSum):
            new_weights = self.weights + A.weights
            new_mpos = self.mpos + A.mpos
        new_MPOSum = MPOSum(mpos=new_mpos, weights=new_weights)
        return new_MPOSum

    def __sub__(self, A):
        """Subtract an MPO, MPOList or MPOSum from the MPOSum.

        Parameters
        ----------
        A    -- MPO, MPOList or MPOSum object.

        Output
        ------
        mpo_sum    -- New MPOSum.
        """
        if isinstance(A, MPO):
            new_weights = self.weights + [-1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOList):
            new_weights = self.weights + [-1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOSum):
            new_weights = self.weights + list((-1) * np.asarray(A.weights))
            new_mpos = self.mpos + A.mpos
        new_MPOSum = MPOSum(mpos=new_mpos, weights=new_weights)
        return new_MPOSum

    def __mul__(self, n):
        """Multiply an MPOSum quantum state by an scalar n (MPOSum * n)

        Parameters
        ----------
        n    -- Scalar to multiply the MPOSum by.

        Output
        ------
        mps    -- New mps.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPOSum by {n}")
        new_weights = [n * weight for weight in self.weights]
        new_MPOSum = MPOSum(
            mpos=self.mpos,
            weights=new_weights,
            maxsweeps=self.maxsweeps,
            tolerance=self.tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return new_MPOSum

    def __rmul__(self, n):
        """Multiply an MPOSum quantum state by an scalar n (n * MPOSum).

        Parameters
        ----------
        n    -- Scalar to multiply the MPOSum by.

        Output
        ------
        mps    -- New mps.
        """
        if not np.isscalar(n):
            raise Exception(f"Cannot multiply MPOSum by {n}")
        new_weights = [n * weight for weight in self.weights]
        new_MPOSum = MPOSum(
            mpos=self.mpos,
            weights=new_weights,
            maxsweeps=self.maxsweeps,
            tolerance=self.tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return new_MPOSum

    def tomatrix(self):
        """Return the matrix representation of this MPO."""
        A = self.weights[0] * self.mpos[0].tomatrix()
        for i, mpo in enumerate(self.mpos[1:]):
            A = A + self.weights[i + 1] * mpo.tomatrix()
        return A

    def apply(self, b):
        """Implement multiplication A @ b between an MPOSum 'A' and
        a Matrix Product State 'b'."""
        if isinstance(b, MPSSum):
            b = b.toMPS()
        states = []
        for mpo in self.mpos:
            state = mpo.apply(b)
            err = 0.0
            if self.simplify and not mpo.simplify:
                state, err, _ = simplify(
                    state,
                    maxsweeps=self.maxsweeps,
                    tolerance=self.tolerance,
                    normalize=self.normalize,
                    max_bond_dimension=self.max_bond_dimension,
                )
            states.append(state)
            log(f"Total error after applying MPOList {b.error()}, incremented by {err}")
        state, _ = combine(
            self.weights,
            states,
            maxsweeps=self.maxsweeps,
            tolerance=self.tolerance,
            normalize=self.normalize,
            max_bond_dimension=self.max_bond_dimension,
        )
        return state

    def __matmul__(self, b):
        """Implement multiplication A @ b between an MPOSum 'A' and
        a Matrix Product State 'b'."""
        return self.apply(b)

    def extend(self, L, sites=None, dimensions=2):
        """Enlarge an MPOSum so that it acts on a larger Hilbert space with 'L' sites.

        Parameters
        ----------
        L          -- The new size
        dimensions -- If it is an integer, it is the dimension of the new sites.
                      If it is a list, it is the dimension of all sites.
        sites      -- Where to place the tensors of the original MPO.

        Output
        ------
        mpo        -- A new MPOSum.
        """
        data = [mpo.extend(L, sites=sites, dimensions=dimensions) for mpo in self.mpos]
        return MPOSum(
            mpos=data,
            weights=self.weights,
            simplify=self.simplify,
            tolerance=self.tolerance,
            normalize=self.normalize,
            maxsweeps=self.maxsweeps,
            max_bond_dimension=self.max_bond_dimension,
        )
