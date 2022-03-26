import numpy as np
import scipy.linalg
from .truncation import truncate_vector, DEFAULT_TOLERANCE
from .mps import MPS


def vector2mps(ψ, dimensions, tolerance=DEFAULT_TOLERANCE, normalize=True):
    """Construct a list of tensors for an MPS that approximates the state ψ
    represented as a complex vector in a Hilbert space.

    Parameters
    ----------
    ψ         -- wavefunction with \prod_i dimensions[i] elements
    dimensions -- list of dimensions of the Hilbert spaces that build ψ
    tolerance -- truncation criterion for dropping Schmidt numbers
    normalize -- boolean to determine if the MPS is normalized
    """

    def SchmidtSplit(ψ, tolerance):
        a, b = ψ.shape
        U, s, V = scipy.linalg.svd(
            ψ,
            full_matrices=False,
            check_finite=False,
            overwrite_a=True,
            lapack_driver="gesdd",
        )
        s, err = truncate_vector(s, tolerance, None)
        D = s.size
        return np.reshape(U[:, :D], (a, D)), np.reshape(s, (D, 1)) * V[:D, :]

    Da = 1
    dimensions = np.array(dimensions, dtype=int)
    Db = np.prod(dimensions)
    output = [0] * len(dimensions)
    for (i, d) in enumerate(dimensions[:-1]):
        # We split a new subsystem and group the left bond dimension
        # and the physical index into a large index
        ψ = np.reshape(ψ, (Da * d, int(Db / d)))
        #
        # We then split the state using the Schmidt decomposition. This
        # produces a tensor for the site we are looking at and leaves
        # us with a (hopefully) smaller state for the rest
        A, ψ = SchmidtSplit(ψ, tolerance)
        output[i] = np.reshape(A, (Da, d, A.shape[1]))
        Da, Db = ψ.shape

    output[-1] = np.reshape(ψ, (Da, Db, 1))
    if normalize == True:
        output[-1] /= np.linalg.norm(ψ)

    return output


def product(vectors, length=None):
    #
    # If `length` is `None`, `vectors` will be a list of complex vectors
    # representing the elements of the product state.
    #
    # If `length` is an integer, `vectors` is a single complex vector and
    # it is repeated `length` times to build a product state.
    #
    def to_tensor(v):
        v = np.array(v)
        return np.reshape(v, (1, v.size, 1))

    if length is not None:
        return MPS([to_tensor(vectors)] * length)
    else:
        return MPS([to_tensor(v) for v in vectors])


def GHZ(n):
    """Return a GHZ state with `n` qubits in MPS form."""
    a = np.zeros((2, 2, 2))
    b = a.copy()
    a[0, 0, 0] = a[0, 1, 1] = 1.0 / np.sqrt(2.0)
    b[0, 0, 0] = 1.0
    b[1, 1, 1] = 1.0
    data = [a] + [b] * (n - 1)
    data[0] = a[0:1, :, :]
    b = data[n - 1]
    data[n - 1] = b[:, :, 1:2] + b[:, :, 0:1]
    return MPS(data)


def W(n):
    """Return a W with one excitation over `n` qubits."""
    a = np.zeros((2, 2, 2))
    a[0, 0, 0] = 1.0
    a[0, 1, 1] = 1.0 / np.sqrt(n)
    a[1, 0, 1] = 1.0
    data = [a] * n
    data[0] = a[0:1, :, :]
    data[n - 1] = data[n - 1][:, :, 1:2]
    return MPS(data)


def wavepacket(ψ):
    #
    # Create an MPS for a spin 1/2 system with the given amplitude
    # of the excited state on each site. In other words, we create
    #
    #   \sum_i Ψ[i] σ^+ |0000...>
    #
    # The MPS is created with a single tensor: A(i,s,j)
    # The input index "i" can take two values, [0,1]. If it is '0'
    # it means we have not applied any σ^+ anywhere else, and we can
    # excite a spin here. Therefore, we have two possible values:
    #
    #   A(0,0,0) = 1.0
    #   A(0,1,1) = ψ[n] (n=given site)
    #
    # If i=1, then we cannot excite any further spin and
    #   A(1,0,1) = 1.0
    #
    # All other elements are zero. Of course, we have to impose
    # boundary conditions that the first site only has A(0,s,j)
    # and the last site only has A(i,s,1) (at least one spin has
    # been excited)
    #
    ψ = np.array(ψ)
    data = [0] * ψ.size
    for n in range(0, ψ.size):
        B = np.zeros((2, 2, 2), dtype=ψ.dtype)
        B[0, 0, 0] = B[1, 0, 1] = 1.0
        B[0, 1, 1] = ψ[n]
        data[n] = B
    data[0] = data[0][0:1, :, :]
    data[-1] = data[-1][:, :, 1:]
    return MPS(data)


def graph(n, mps=True):
    """Create a one-dimensional graph state of `n` qubits."""
    # Choose entangled pair state as : |00>+|11>
    # Apply Hadamard H on the left virtual spins (which are the right spins of the entangled bond pairs)
    assert n > 1
    H = np.array([[1, 1], [1, -1]])
    # which gives |0>x(|0>+|1>)+|1>x(|0>-|1>) = |00>+|01>+|10>-|11>
    # Project as  |0><00| + |1><11|
    # We get the following MPS projectors:
    A0 = np.dot(np.array([[1, 0], [0, 0]]), H)
    A1 = np.dot(np.array([[0, 0], [0, 1]]), H)
    AA = np.array([A0, A1])
    AA = np.swapaxes(AA, 0, 1)
    data = [AA] * n
    data[0] = np.dot(np.array([[[1, 0], [0, 1]]]), H)
    data[-1] = np.swapaxes(np.array([[[1, 0], [0, 1]]]), 0, 2) / np.sqrt(2 ** n)
    return MPS(data)


# open boundary conditions
# free virtual spins at both ends are taken to be zero


def AKLT(n, mps=True):
    """Return an AKL state with `n` spin-1 particles."""
    assert n > 1
    # Choose entangled pair state as : |00>+|11>
    # Apply i * Pauli Y matrix on the left virtual spins (which are the right spins of the entangled bond pairs)
    iY = np.array([[0, 1], [-1, 0]])
    # which gives -|01>+|10>
    # Project as  |-1><00| +|0> (<01|+ <10|)/ \sqrt(2)+ |1><11|
    # We get the following MPS projectors:
    A0 = np.dot(np.array([[1, 0], [0, 0]]), iY)
    A1 = np.dot(np.array([[0, 1], [1, 0]]), iY)
    A2 = np.dot(np.array([[0, 0], [0, 1]]), iY)

    AA = np.array([A0, A1, A2]) / np.sqrt(2)
    AA = np.swapaxes(AA, 0, 1)
    data = [AA] * n
    data[-1] = np.array([[[1, 0], [0, 1], [0, 0]]])
    data[0] = np.array(np.einsum("ijk,kl->ijl", data[-1], iY)) / np.sqrt(2)
    data[-1] = np.swapaxes(data[-1], 0, 2)

    return MPS(data)


def random(d, N, D=1):
    """Create a random state with 'N' elements of dimension 'd' and bond
    dimension 'D'."""
    mps = [1] * N
    DR = 1
    for i in range(N):
        DL = DR
        if N > 60 and i != N - 1:
            DR = D
        else:
            DR = np.min([DR * d, D, d ** (N - i - 1)])
        mps[i] = np.random.rand(DL, d, DR)
    return MPS(mps)


def gaussian(n, x0, w0, k0, mps=True):
    #
    # Return a W state with `n` components in MPS form or
    # in vector form
    #
    xx = np.arange(n, dtype=complex)
    coefs = np.exp(-((xx - x0) ** 2) / w0 ** 2 + 1j * k0 * xx, dtype=complex)
    return wavepacket(coefs / np.linalg.norm(coefs))

