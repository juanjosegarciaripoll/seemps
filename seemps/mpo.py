import numpy as np
from .typing import *
import copy
from .state import MPS, MPSSum, array, DEFAULT_STRATEGY, Strategy, Weight
from . import truncate
from .tools import log, InvalidOperation


def mpo_multiply_tensor(A, B):
    C = np.einsum("aijb,cjd->acibd", A, B)
    s = C.shape
    return C.reshape(s[0] * s[1], s[2], s[3] * s[4])


class MPO(array.TensorArray):
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

    strategy: Strategy

    __array_priority__ = 10000

    def __init__(self, data: list[np.ndarray], strategy: Strategy = DEFAULT_STRATEGY):
        super(MPO, self).__init__(data)
        assert data[0].shape[0] == data[-1].shape[-1] == 1
        self.strategy = strategy

    def __mul__(self, n: Weight) -> "MPO":
        """Multiply an MPO quantum state by an scalar n (MPO * n)

        Parameters
        ----------
        n          -- Scalar to multiply the MPO by.

        Output
        ------
        mpo -- New mpo.
        """
        if isinstance(n, (float, complex)):
            mpo_mult = copy.deepcopy(self)
            mpo_mult._data[0] = n * mpo_mult._data[0]
            return mpo_mult
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> "MPO":
        """Multiply an MPO quantum state by an scalar n (n * MPO).

        Parameters
        ----------
        n          -- Scalar to multiply the MPO by.

        Output
        ------
        mpo -- New mpo.
        """
        if isinstance(n, (float, complex)):
            mpo_mult = copy.deepcopy(self)
            mpo_mult._data[0] = n * mpo_mult._data[0]
            return mpo_mult
        raise InvalidOperation("*", n, self)

    def dimensions(self) -> list[int]:
        """Return the local dimensions of the MPO."""
        return [A.shape[1] for A in self._data]

    def tomatrix(self) -> np.ndarray:
        """Return the matrix representation of this MPO."""
        D = 1  # Total physical dimension so far
        out = np.array([[[1.0]]])
        for A in self._data:
            _, i, _, b = A.shape
            out = np.einsum("lma,aijb->limjb", out, A)
            D *= i
            out = out.reshape(D, D, b)
        return out[:, :, 0]

    def apply(
        self,
        b: Union[MPS, MPSSum],
        strategy: Optional[Strategy] = None,
        simplify: Optional[bool] = None,
    ) -> MPS:
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        # TODO: Remove implicit conversion of MPSSum to MPS
        if isinstance(b, MPSSum):
            state: MPS = b.toMPS(strategy=strategy)
        elif isinstance(b, MPS):
            state = b
        else:
            raise TypeError(f"Cannot multiply MPO with {b}")
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()
        assert self.size == state.size
        log(f"Total error before applying MPO {state.error()}")
        err = 0.0
        state = MPS(
            [mpo_multiply_tensor(A, B) for A, B in zip(self._data, state._data)],
            error=state.error(),
        )
        if strategy.get_simplify_flag():
            state, err, _ = truncate.simplify(
                state,
                maxsweeps=strategy.get_max_sweeps(),
                tolerance=strategy.get_tolerance(),
                normalize=strategy.get_normalize_flag(),
                max_bond_dimension=strategy.get_max_bond_dimension(),
            )
        log(f"Total error after applying MPO {state.error()}, incremented by {err}")
        return state

    def __matmul__(self, b: Union[MPS, MPSSum]) -> MPS:
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        if isinstance(b, (MPS, MPSSum)):
            return self.apply(b)
        raise InvalidOperation("@", self, b)

    # TODO: We have to change the signature and working of this function, so that
    # 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
    # needed. In this case, 'dimensions' will only list the dimensions of the added
    # sites, not all of them.
    def extend(
        self,
        L: int,
        sites: Optional[Iterable[int]] = None,
        dimensions: Union[int, list[int]] = 2,
    ) -> "MPO":
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
        if isinstance(dimensions, list):
            final_dimensions = dimensions
        else:
            final_dimensions = [dimensions] * L
        if sites is None:
            sites = range(self.size)

        data: list[np.ndarray] = [np.ndarray(())] * L
        for ndx, A in zip(sites, self):
            data[ndx] = A
            final_dimensions[ndx] = A.shape[2]
        D = 1
        for i, A in enumerate(data):
            if A.ndim == 0:
                d = final_dimensions[i]
                A = np.eye(D).reshape(D, 1, 1, D) * np.eye(d).reshape(1, d, d, 1)
                data[i] = A
            else:
                D = A.shape[-1]
        return MPO(data, strategy=self.strategy)


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

    mpos: list[MPO]
    strategy: Strategy

    def __init__(self, mpos: list[MPO], strategy: Strategy = DEFAULT_STRATEGY):
        self.mpos = mpos
        self.strategy = strategy

    def __mul__(self, n: Complex) -> "MPOList":
        """Multiply an MPOList quantum state by an scalar n (MPOList * n).

        Parameters
        ----------
        n          -- Scalar to multiply the MPOList by.

        Output
        ------
        mpo -- New mpo.
        """
        if isinstance(n, (float, complex)):
            return MPOList([n * self.mpos[0]] + self.mpos[1:], self.strategy)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Complex) -> "MPOList":
        """Multiply an MPOList quantum state by an scalar n (n * MPOList).

        Parameters
        ----------
        n          -- Scalar to multiply the MPOList by.

        Output
        ------
        mpo -- New mpo.
        """
        if isinstance(n, (float, complex)):
            return MPOList([n * self.mpos[0]] + self.mpos[1:], self.strategy)
        raise InvalidOperation("*", n, self)

    def tomatrix(self) -> np.ndarray:
        """Return the matrix representation of this MPO."""
        A = self.mpos[0].tomatrix()
        for mpo in self.mpos[1:]:
            A = A @ mpo.tomatrix()
        return A

    def apply(
        self,
        b: Union[MPS, MPSSum],
        strategy: Optional[Strategy] = None,
        simplify: Optional[bool] = None,
    ) -> MPS:
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        state: MPS
        if isinstance(b, MPSSum):
            state = b.toMPS()
        else:
            state = b
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()
        for mpo in self.mpos:
            # log(f'Total error before applying MPOList {b.error()}')
            state = mpo.apply(state)
        err = 0.0
        if simplify:
            state, err, _ = truncate.simplify(
                state,
                maxsweeps=strategy.get_max_sweeps(),
                tolerance=strategy.get_tolerance(),
                normalize=strategy.get_normalize_flag(),
                max_bond_dimension=strategy.get_max_bond_dimension(),
            )
        log(f"Total error after applying MPOList {state.error()}, incremented by {err}")
        return state

    def __matmul__(self, b: Union[MPS, MPSSum]) -> MPS:
        """Implement multiplication A @ b between an MPO 'A' and
        a Matrix Product State 'b'."""
        if isinstance(b, (MPS, MPSSum)):
            return self.apply(b)
        raise InvalidOperation("@", self, b)

    def extend(
        self, L: int, sites: Optional[list[int]] = None, dimensions: int = 2
    ) -> "MPOList":
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
        return MPOList(
            [mpo.extend(L, sites=sites, dimensions=dimensions) for mpo in self.mpos],
            strategy=self.strategy,
        )
