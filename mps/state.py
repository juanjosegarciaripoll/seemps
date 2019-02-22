import numpy as np
from mps import expectation


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
        """Create a new TensorArray from a list of tensors. 'data' is an
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

    def copy(self):
        """Return a fresh new TensorArray that shares the same tensor as its
        sibling, but which can be destructively modified without affecting it.
        """
        return self.__copy__()


class MPS(TensorArray):
    """MPS (Matrix Product State) class.

    This implements a bare-bones Matrix Product State object with open
    boundary conditions. The tensors have three indices, A[α,i,β], where
    'α,β' are the internal labels and 'i' is the physical state of the given
    site.

    Attributes:
    size = number of tensors in the array
    """

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    def __init__(self, data):
        super(MPS, self).__init__(data)

    def dimension(self):
        """Return the total size of the Hilbert space in which this MPS lives."""
        return np.product([a.shape[1] for a in self._data])

    def tovector(self):
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return _mps2vector(self)

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖**2 = <ψ|ψ>."""
        return expectation.scprod(self, self)

    def expectation1(self, operator, n):
        """Return the expectation value of 'operator' acting on the 'n'-th
        site of the MPS."""
        return expectation.expectation1_non_canonical(self, operator, n)

    def all_expectation1(self, operator):
        """Return all expectation values of 'operator' acting on all possible
        sites of the MPS."""
        return expectation.all_expectation1_non_canonical(self, operator)


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
        Ψ = np.einsum('Da,akb->Dkb', Ψ, A)
        D = D * d
        Ψ = np.reshape(Ψ, (D, β))
    return Ψ.reshape((Ψ.size,))


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
        return MPS(map(to_tensor, vectors))
    


def GHZ(n):
    """Return a GHZ state with `n` qubits in MPS form."""
    a = np.zeros((2, 2, 2))
    b = a.copy()
    a[0, 0, 0] = a[0, 1, 1] = 1.0/np.sqrt(2.0)
    b[0, 0, 0] = 1.0
    b[1, 1, 1] = 1.0
    data = [a]+[b] * (n-1)
    data[0] = a[0:1, :, :]
    b = data[n-1]
    data[n-1] = (b[:, :, 1:2] + b[:, :, 0:1])
    return MPS(data)


def W(n):
    """Return a W with one excitation over `n` qubits."""
    a = np.zeros((2, 2, 2))
    a[0, 0, 0] = 1.0
    a[0, 1, 1] = 1.0/np.sqrt(n)
    a[1, 0, 1] = 1.0
    data = [a] * n
    data[0] = a[0:1, :, :]
    data[n-1] = data[n-1][:, :, 1:2]
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
    H = np.array([[1, 1], [1, -1]])
    # which gives |0>x(|0>+|1>)+|1>x(|0>-|1>) = |00>+|01>+|10>-|11>
    # Project as  |0><00| + |1><11|
    # We get the following MPS projectors:
    A0 = np.dot(np.array([[1, 0], [0, 0]]), H)
    A1 = np.dot(np.array([[0, 0], [0, 1]]), H)
    AA = np.array([A0, A1])
    AA = np.swapaxes(AA, 0, 1)
    data = [AA]*n
    data[0] = np.dot(np.array([[[1, 0], [0, 1]]]), H)
    data[-1] = np.swapaxes(np.array([[[1, 0], [0, 1]]]), 0, 2) / np.sqrt(2**n)

    return MPS(data)

# open boundary conditions
# free virtual spins at both ends are taken to be zero


def AKLT(n, mps=True):
    """Return an AKL state with `n` spin-1 particles."""
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
    data = [AA]*n
    data[-1] = np.array([[[1, 0], [0, 1], [0, 0]]])
    data[0] = np.array(np.einsum('ijk,kl->ijl',
                                 data[-1], iY))/np.sqrt(2)
    data[-1] = np.swapaxes(data[-1], 0, 2)

    return MPS(data)



def random(d, N, D=1):
    """Create a random state with 'N' elements of dimension 'd' and bond
    dimension 'D'."""
    mps = [1]*N
    DR = 1
    for i in range(N):
        DL = DR
        if N > 60 and i != N-1:
            DR = D
        else:
            DR = np.min([DR*d, D, d**(N-i-1)])
        mps[i] = np.random.rand(DL, d, DR)
    return MPS(mps)


def _ortho_right(A, tol):
    α, i, β = A.shape
    U, s, V = np.linalg.svd(np.reshape(A, (α*i, β)), full_matrices=False)
    s = _truncate_vector(s, tol)
    D = s.size
    return np.reshape(U[:,:D], (α, i, D)), np.reshape(s, (D, 1)) * V[:D, :]


def _ortho_left(A, tol):
    α, i, β = A.shape
    U, s, V = np.linalg.svd(np.reshape(A, (α, i*β)), full_matrices=False)
    s = _truncate_vector(s, tol)
    D = s.size
    return np.reshape(V[:D,:], (D, i, β)), U[:, :D] * np.reshape(s, (1, D))


def _update_in_canonical_form(Ψ, A, site, direction, tolerance):
    """Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process."""

    if direction > 0:
        if site+1 == Ψ.size:
            Ψ[site] = A
        else:
            Ψ[site], sV = _ortho_right(A, tolerance)
            site += 1
            Ψ[site] = np.einsum('ab,bic->aic', sV, Ψ[site])
    else:
        if site == 0:
            Ψ[site] = A
        else:
            Ψ[site], Us = _ortho_left(A, tolerance)
            site -= 1
            Ψ[site] = np.einsum('aib,bc->aic', Ψ[site], Us)
    return site

def _canonicalize(Ψ, center, tolerance):
    for i in range(0, center):
        _update_in_canonical_form(Ψ, Ψ[i], i, +1, tolerance)
    for i in range(Ψ.size-1, center, -1):
        _update_in_canonical_form(Ψ, Ψ[i], i, -1, tolerance)


DEFAULT_TOLERANCE = np.finfo(np.float64).eps

def _truncate_vector(S, tolerance):
    #
    # Input:
    # - S: a vector containing singular values in descending order
    # - tolerance: truncation relative tolerance, which specifies an
    #   upper bound for the sum of the squares of the singular values
    #   eliminated. 0 <= tolerance <= 1
    #
    # Output:
    # - truncS: truncated version of S
    #
    if tolerance == 0:
        #log('--no truncation')
        return S
    # We sum all reduced density matrix eigenvalues, starting from
    # the smallest ones, to avoid rounding errors
    err = np.cumsum(np.flip(S, axis=0)**2)
    #
    # This is the sum of all values
    total = err[-1]
    #
    # we find the number of values we can drop within the relative
    # tolerance
    ndx = np.argmax(err >= tolerance*total)
    # and use that to estimate the size of the array
    # log('--S='+str(S))
    #log('--truncated to '+str(ndx))
    return S[0:(S.size - ndx)]


class CanonicalMPS(MPS):
    """Canonical MPS class.

    This implements a Matrix Product State object with open boundary
    conditions, that is always on canonical form with respect to a given site.
    The tensors have three indices, A[α,i,β], where 'α,β' are the internal
    labels and 'i' is the physical state of the given site.

    Attributes:
    size = number of tensors in the array
    center = site that defines the canonical form of the MPS
    """

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    def __init__(self, data, center=0, tolerance=DEFAULT_TOLERANCE):
        super(MPS, self).__init__(data)
        _canonicalize(self, center, tolerance)
        self.center = center

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖**2 = <ψ|ψ>."""
        A = self._data[self.center]
        return np.vdot(A, A)

    def expectationAtCenter(self, operator):
        """Return the expectation value of 'operator' acting on the central
        site of the MPS."""
        A = self._data[self.center]
        return np.vdot(A, np.einsum('ij,ajb->aib', operator, A))/np.vdot(A,A)

    def update_canonical(self, A, direction):
        self.center = _update_in_canonical_form(self, A, self.center,
                                                direction)
