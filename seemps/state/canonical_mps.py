import numpy as np
from .mps import MPS
from .svd import svd
from .schmidt import vector2mps
from .truncation import truncate_vector, DEFAULT_TOLERANCE
from .. import expectation


def _ortho_right(A, tol, normalize):
    α, i, β = A.shape
    U, s, V = svd(A.reshape(α * i, β), full_matrices=False)
    s, err = truncate_vector(s, tol, None)
    if normalize:
        s /= np.linalg.norm(s)
    D = s.size
    return U[:, :D].reshape(α, i, D), s.reshape(D, 1) * V[:D, :], err


def _ortho_left(A, tol, normalize):
    α, i, β = A.shape
    U, s, V = svd(A.reshape(α, i * β), full_matrices=False)
    s, err = truncate_vector(s, tol, None)
    if normalize:
        s /= np.linalg.norm(s)
    D = s.size
    return V[:D, :].reshape(D, i, β), U[:, :D] * s.reshape(1, D), err


def _update_in_canonical_form(Ψ, A, site, direction, tolerance, normalize):
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
        if site + 1 == Ψ.size:
            Ψ[site] = A
            err = 0.0
        else:
            Ψ[site], sV, err = _ortho_right(A, tolerance, normalize)
            site += 1
            Ψ[site] = np.einsum("ab,bic->aic", sV, Ψ[site])
    else:
        if site == 0:
            Ψ[site] = A
            err = 0.0
        else:
            Ψ[site], Us, err = _ortho_left(A, tolerance, normalize)
            site -= 1
            Ψ[site] = np.einsum("aib,bc->aic", Ψ[site], Us)
    return site, err


def _canonicalize(Ψ, center, tolerance, normalize):
    err = 0.0
    for i in range(0, center):
        center, errk = _update_in_canonical_form(Ψ, Ψ[i], i, +1, tolerance, normalize)
        err += errk
    for i in range(Ψ.size - 1, center, -1):
        center, errk = _update_in_canonical_form(Ψ, Ψ[i], i, -1, tolerance, normalize)
        err += errk
    return err


def left_orth_2site(AA, tolerance, normalize, max_bond_dimension):
    α, d1, d2, β = AA.shape
    Ψ = AA.reshape(α * d1, β * d2)
    U, S, V = svd(Ψ, full_matrices=False)
    S, err = truncate_vector(S, tolerance, max_bond_dimension)
    if normalize:
        S /= np.linalg.norm(S)
    D = S.size
    U = U[:, :D].reshape(α, d1, D)
    SV = (S.reshape(D, 1) * V[:D, :]).reshape(D, d2, β)
    return U, SV, err


def right_orth_2site(AA, tolerance, normalize, max_bond_dimension):
    α, d1, d2, β = AA.shape
    Ψ = AA.reshape(α * d1, β * d2)
    U, S, V = svd(Ψ, full_matrices=False)
    S, err = truncate_vector(S, tolerance, max_bond_dimension)
    if normalize:
        S /= np.linalg.norm(S)
    D = S.size
    US = (U[:, :D] * S).reshape(α, d1, D)
    V = V[:D, :].reshape(D, d2, β)
    return US, V, err


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

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    def __init__(
        self, data, center=None, error=0, normalize=False, tolerance=DEFAULT_TOLERANCE
    ):
        super(CanonicalMPS, self).__init__(data, error=error)
        if isinstance(data, CanonicalMPS):
            self.center = data.center
            self._error = data._error
            if center is not None:
                self.recenter(center, tolerance, normalize)
        else:
            self.center = center = self._interpret_center(
                0 if center is None else center
            )
            self.update_error(_canonicalize(self, center, tolerance, normalize))
        if normalize:
            A = self[center]
            self[center] = A / np.linalg.norm(A)
        self.tolerance = tolerance

    @classmethod
    def from_vector(
        ψ, dimensions, center=0, normalize=False, tolerance=DEFAULT_TOLERANCE
    ):
        return CanonicalMPS(
            vector2mps(ψ, dimensions, tolerance),
            center=center,
            normalize=normalize,
            tolerance=tolerance,
        )

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖^2 = <ψ|ψ>."""
        A = self._data[self.center]
        return np.vdot(A, A)

    def left_environment(self, site):
        start = min(site, self.center)
        ρ = expectation.begin_environment(self[start].shape[0])
        for A in self[start:site]:
            ρ = expectation.update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site):
        start = max(site, self.center)
        ρ = expectation.begin_environment(self[start].shape[-1])
        for A in self[start:site:-1]:
            ρ = expectation.update_right_environment(A, A, ρ)
        return ρ

    def expectation1(self, operator, site=None):
        """Return the expectated value of `operator` acting on the given `site`."""
        if site is None or site == self.center:
            A = self._data[self.center]
            return np.vdot(A, np.einsum("ij,ajb->aib", operator, A))
        else:
            return expectation.expectation1(self, operator, site)

    def entanglement_entropyAtCenter(self):
        A = self._data[self.center]
        d1, d2, d3 = A.shape
        u, s, v = svd(A.reshape(d1 * d2, d3), full_matrices=False)
        return -np.sum(2 * s * s * np.log2(s))

    def update_canonical(
        self, A, direction, tolerance=DEFAULT_TOLERANCE, normalize=False
    ):
        self.center, err = _update_in_canonical_form(
            self, A, self.center, direction, tolerance, normalize
        )
        self.update_error(err)
        return err

    def update_2site(
        self,
        AA,
        site,
        direction,
        tolerance=DEFAULT_TOLERANCE,
        normalize=False,
        max_bond_dimension=None,
    ):
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
            self._data[site], self._data[site + 1], err = right_orth_2site(
                AA, tolerance, normalize, max_bond_dimension
            )
            self.center = site
        else:
            self._data[site], self._data[site + 1], err = left_orth_2site(
                AA, tolerance, normalize, max_bond_dimension
            )
            self.center = site + 1
        self.update_error(err)
        return err

    def _interpret_center(self, center):
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

    def recenter(self, center, tolerance=DEFAULT_TOLERANCE, normalize=False):
        """Update destructively the state to be in canonical form with respect
        to a different site."""
        center = self._interpret_center(center)
        old = self.center
        if center != old:
            dr = +1 if center > old else -1
            for i in range(old, center, dr):
                self.update_canonical(self._data[i], dr, tolerance, normalize)
        return self

    def __copy__(self):
        #
        # Return a copy of the MPS with a fresh new array.
        #
        return type(self)(self)

    def copy(self):
        """Return a fresh new TensorArray that shares the same tensor as its
        sibling, but which can be destructively modified without affecting it.
        """
        return self.__copy__()
