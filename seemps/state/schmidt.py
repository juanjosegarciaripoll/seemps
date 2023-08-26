import numpy as np
from .truncation import truncate_vector, DEFAULT_TOLERANCE
from .svd import svd


def SchmidtSplit(ψ, tolerance, overwrite=False):
    a, _ = ψ.shape
    U, s, V = svd(ψ, full_matrices=False, overwrite_a=overwrite)
    s, _ = truncate_vector(s, tolerance, None)
    D = s.size
    return U[:, :D].reshape(a, D), s.reshape(D, 1) * V[:D, :]


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

    Da = 1
    dimensions = np.array(dimensions, dtype=int)
    Db = np.prod(dimensions)
    if Db != ψ.size:
        raise Exception("Wrong dimensions specified when converting a vector to MPS")
    output = [0] * len(dimensions)
    for i, d in enumerate(dimensions[:-1]):
        # We split a new subsystem and group the left bond dimension
        # and the physical index into a large index
        ψ = ψ.reshape(Da * d, int(Db / d))
        #
        # We then split the state using the Schmidt decomposition. This
        # produces a tensor for the site we are looking at and leaves
        # us with a (hopefully) smaller state for the rest
        A, ψ = SchmidtSplit(ψ, tolerance, overwrite=(i > 0))
        output[i] = A.reshape(Da, d, A.shape[1])
        Da, Db = ψ.shape

    if normalize is True:
        ψ /= np.linalg.norm(ψ)
    output[-1] = ψ.reshape(Da, Db, 1)

    return output
