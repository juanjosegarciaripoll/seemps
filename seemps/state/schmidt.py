import numpy as np
from .truncation import truncate_vector, DEFAULT_TOLERANCE
from . import core
from scipy.linalg import svd

#
# Type of LAPACK driver used for solving singular value decompositions.
# The "gesdd" algorithm is the default in Python and is faster, but it
# may produced wrong results, specially in ill-conditioned matrices.
#
SVD_LAPACK_DRIVER = "gesdd"


def _schmidt_split(ψ, strategy, overwrite):
    U, s, V = svd(
        ψ,
        full_matrices=False,
        overwrite_a=overwrite,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s, _ = core.truncate_vector(s, strategy)
    D = s.size
    return U[:, :D], s.reshape(D, 1) * V[:D, :]


def ortho_right(A, tol, normalize):
    α, i, β = A.shape
    U, s, V = svd(
        A.reshape(α * i, β),
        full_matrices=False,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s, err = truncate_vector(s, tol, None)
    if normalize:
        s /= np.linalg.norm(s)
    D = s.size
    return U[:, :D].reshape(α, i, D), s.reshape(D, 1) * V[:D, :], err


def ortho_left(A, tol, normalize):
    α, i, β = A.shape
    U, s, V = svd(
        A.reshape(α, i * β),
        full_matrices=False,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s, err = truncate_vector(s, tol, None)
    if normalize:
        s /= np.linalg.norm(s)
    D = s.size
    return V[:D, :].reshape(D, i, β), U[:, :D] * s.reshape(1, D), err


def left_orth_2site(AA, tolerance, normalize, max_bond_dimension):
    α, d1, d2, β = AA.shape
    Ψ = AA.reshape(α * d1, β * d2)
    U, S, V = svd(
        Ψ, full_matrices=False, check_finite=False, lapack_driver=SVD_LAPACK_DRIVER
    )
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
    U, S, V = svd(Ψ, full_matrices=False, lapack_driver=SVD_LAPACK_DRIVER)
    S, err = truncate_vector(S, tolerance, max_bond_dimension)
    if normalize:
        S /= np.linalg.norm(S)
    D = S.size
    US = (U[:, :D] * S).reshape(α, d1, D)
    V = V[:D, :].reshape(D, d2, β)
    return US, V, err


def vector2mps(ψ, dimensions, tolerance=DEFAULT_TOLERANCE, normalize=True):
    """Construct a list of tensors for an MPS that approximates the state ψ
    represented as a complex vector in a Hilbert space.

    Parameters
    ----------
    ψ         -- wavefunction with \\prod_i dimensions[i] elements
    dimensions -- list of dimensions of the Hilbert spaces that build ψ
    tolerance -- truncation criterion for dropping Schmidt numbers
    normalize -- boolean to determine if the MPS is normalized
    """

    if np.prod(dimensions) != ψ.size:
        raise Exception("Wrong dimensions specified when converting a vector to MPS")
    output = [0] * len(dimensions)
    Da = 1
    strategy = core.TruncationStrategy(tolerance=tolerance, normalize=False)
    for i, d in enumerate(dimensions[:-1]):
        # We split a new subsystem and group the left bond dimension
        # and the physical index into a large index.
        # We then split the state using the Schmidt decomposition. This
        # produces a tensor for the site we are looking at and leaves
        # us with a (hopefully) smaller state for the rest
        A, ψ = _schmidt_split(ψ.reshape(Da * d, -1), strategy, overwrite=(i > 0))
        output[i] = A.reshape(Da, d, -1)
        Da = ψ.shape[0]

    if normalize is True:
        ψ /= np.linalg.norm(ψ)
    output[-1] = ψ.reshape(Da, dimensions[-1], 1)

    return output
