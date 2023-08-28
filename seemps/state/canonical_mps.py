from numbers import Number
from typing import Any, Iterable, Optional
import warnings
import numpy as np
from .mps import MPS
from . import schmidt
from .core import DEFAULT_STRATEGY, Strategy, Truncation, DEFAULT_TOLERANCE
from .. import expectation


def _update_in_canonical_form(
    Ψ: list[np.ndarray], A: np.ndarray, site: int, direction: int, truncation: Strategy
) -> tuple[int, float]:
    """Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process.

    Arguments:
    ----------
    Ψ = MPS in CanonicalMPS form
    A = tensor to be orthonormalized and inserted at "site" of MPS
    site = the index of the site with respect to which
    orthonormalization is carried out
    direction = if greater (less) than zero right (left) orthonormalization
    is carried out
    tolerance = truncation tolerance for the singular values
    (see truncate_vector in File 1a - MPS class)
    """
    if direction > 0:
        if site + 1 == len(Ψ):
            Ψ[site] = A
            err = 0.0
        else:
            Ψ[site], sV, err = schmidt.ortho_right(A, truncation)
            site += 1
            Ψ[site] = np.einsum("ab,bic->aic", sV, Ψ[site])
    else:
        if site == 0:
            Ψ[site] = A
            err = 0.0
        else:
            Ψ[site], Us, err = schmidt.ortho_left(A, truncation)
            site -= 1
            Ψ[site] = np.einsum("aib,bc->aic", Ψ[site], Us)
    return site, err


def _canonicalize(Ψ: list[np.ndarray], center: int, truncation: Strategy) -> float:
    err = 0.0
    for i in range(0, center):
        _, errk = _update_in_canonical_form(Ψ, Ψ[i], i, +1, truncation)
        err += errk
    for i in range(len(Ψ) - 1, center, -1):
        _, errk = _update_in_canonical_form(Ψ, Ψ[i], i, -1, truncation)
        err += errk
    return err


class CanonicalMPS(MPS):
    """Canonical MPS class.

    This implements a Matrix Product State object with open boundary
    conditions, that is always on canonical form with respect to a given site.
    The tensors have three indices, `A[α,i,β]`, where `α,β` are the internal
    labels and `i` is the physical state of the given site.

    Parameters
    ----------
    data      -- a list of MPS tensors, an MPS or a CanonicalMPS
    center    -- site to make the canonical form. If defaults either to
                 the center of the CanonicalMPS or to zero.
    error     -- norm-2 squared truncation error that we carry on
    tolerance -- truncation tolerance when creating the canonical form
    normalize -- normalize the state after finishing the canonical form
    """

    center: int

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    def __init__(self, data: Iterable[np.ndarray], center=None, **kwdargs):
        super(CanonicalMPS, self).__init__(data, **kwdargs)
        if isinstance(data, CanonicalMPS):
            self.center = data.center
            self._error = data._error
            if center is not None:
                self.recenter(center)
        else:
            self.center = center = self._interpret_center(
                0 if center is None else center
            )
            self.update_error(_canonicalize(self._data, center, self.strategy))
        if self.strategy.get_normalize_flag():
            A = self[center]
            self[center] = A / np.linalg.norm(A)

    @classmethod
    def from_vector(
        cls,
        ψ: np.ndarray,
        dimensions: list[int],
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
        **kwdargs
    ) -> "CanonicalMPS":
        return CanonicalMPS(
            schmidt.vector2mps(ψ, dimensions, strategy, normalize),
            center=kwdargs.get("center", 0),
            strategy=strategy,
        )

    def norm2(self) -> float:
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        warnings.warn(
            "method norm2 is deprecated, use norm_squared", category=DeprecationWarning
        )
        return self.norm_squared()

    def norm_squared(self) -> float:
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        A = self._data[self.center]
        return np.vdot(A, A)

    def left_environment(self, site: int) -> np.ndarray:
        start = min(site, self.center)
        ρ = expectation.begin_environment(self[start].shape[0])
        for A in self[start:site]:
            ρ = expectation.update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site: int) -> np.ndarray:
        start = max(site, self.center)
        ρ = expectation.begin_environment(self[start].shape[-1])
        for A in self[start:site:-1]:
            ρ = expectation.update_right_environment(A, A, ρ)
        return ρ

    def entanglement_entropy(self, site: Optional[int] = None) -> float:
        """Return the entanglement entropy of the state divided at 'site',
        which defaults to the canonical state's center."""
        if site is None:
            site = self.center
        if site != self.center:
            return self.copy().recenter(site).entanglement_entropy()
        A = self._data[site]
        d1, d2, d3 = A.shape
        s = schmidt.svd(
            A.reshape(d1 * d2, d3),
            full_matrices=False,
            compute_uv=False,
            check_finite=False,
            lapack_driver=schmidt.SVD_LAPACK_DRIVER,
        )
        return -np.sum(2 * s * s * np.log2(s))

    def update_canonical(
        self, A: np.ndarray, direction: int, truncation: Strategy
    ) -> float:
        self.center, err = _update_in_canonical_form(
            self._data, A, self.center, direction, truncation
        )
        self.update_error(err)
        return err

    def update_2site(
        self, AA: np.ndarray, site: int, direction: int, truncation: Strategy
    ) -> float:
        """Split a two-site tensor into two one-site tensors by
        left/right orthonormalization and insert the tensor in
        canonical form into the MPS Ψ at the given site and the site
        on its left/right. Update the neighboring sites in the process.

        Arguments:
        ----------
        Ψ = MPS in CanonicalMPS form
        AA = two-site tensor to be split by orthonormalization
        site = the index of the site with respect to which
        orthonormalization is carried out
        direction = if greater (less) than zero right (left) orthonormalization
        is carried out
        tolerance = truncation tolerance for the singular values
        (see truncate_vector in File 1a - MPS class)
        """
        assert site <= self.center <= site + 1
        if direction < 0:
            self._data[site], self._data[site + 1], err = schmidt.right_orth_2site(
                AA, truncation
            )
            self.center = site
        else:
            self._data[site], self._data[site + 1], err = schmidt.left_orth_2site(
                AA, truncation
            )
            self.center = site + 1
        self.update_error(err)
        return err

    def _interpret_center(self, center: int) -> int:
        """Converts `center` into an integer between [0,size-1], with the
        convention that -1 = size-1, -2 = size-2, etc. Trows an exception of
        `center` if out of bounds."""
        size = self.size
        if 0 <= center < size:
            return center
        center += size
        if 0 <= center < size:
            return center
        raise IndexError()

    def recenter(
        self, center: int, strategy: Optional[Strategy] = None
    ) -> "CanonicalMPS":
        """Update destructively the state to be in canonical form with respect
        to a different site."""
        center = self._interpret_center(center)
        old = self.center
        if strategy is None:
            strategy = self.strategy
        if center != old:
            dr = +1 if center > old else -1
            for i in range(old, center, dr):
                self.update_canonical(self._data[i], dr, strategy)
        return self

    def normalize_inplace(self) -> "CanonicalMPS":
        n = self.center
        A = self._data[n]
        self._data[n] = A / np.linalg.norm(A)
        return self

    def __copy__(self):
        #
        # Return a copy of the MPS with a fresh new array.
        #
        return type(self)(
            self, center=self.center, strategy=self.strategy, error=self.error
        )

    def copy(self):
        """Return a fresh new TensorArray that shares the same tensor as its
        sibling, but which can be destructively modified without affecting it.
        """
        return self.__copy__()
