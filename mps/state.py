import numpy as np
import scipy.linalg
from mps import expectation

DEFAULT_TOLERANCE = np.finfo(np.float64).eps


def vector2mps(ψ, dimensions, tolerance=DEFAULT_TOLERANCE):
    """Construct a list of tensors for an MPS that approximates the state ψ
    represented as a complex vector in a Hilbert space.

    Parameters
    ----------
    ψ         -- wavefunction with \prod_i dimensions[i] elements
    dimension -- list of dimensions of the Hilbert spaces that build ψ
    tolerance -- truncation criterion for dropping Schmidt numbers
    """

    def SchmidtSplit(ψ, tolerance):
        a, b = ψ.shape
        U, s, V = np.linalg.svd(ψ, full_matrices=False)
        s, err = _truncate_vector(s, tolerance, None)
        D = s.size
        return np.reshape(U[:, :D], (a, D)), np.reshape(s, (D, 1)) * V[:D, :]

    Da = 1
    dimensions = np.array(dimensions, dtype=np.int)
    Db = np.prod(dimensions)
    output = [0] * len(dimensions)
    for (i, d) in enumerate(dimensions):
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

    return output


def _truncate_vector(S, tolerance, dimension):
    """Given a vector of Schmidt numbers 'S', a 'tolerance' and a maximum
    bond 'dimension', determine where to truncate the vector and return
    the absolute error in norm-2 made by that truncation.
    
    Parameters
    ----------
    S         -- non-negative Schmidt numbers in decreasing order.
    tolerance -- absolute error allowed by the truncation
    dimension -- maximum size allowed by the truncation (or None)
    
    Output
    ------
    S         -- new, truncated vector of Schmidt numbers
    error     -- norm-2 error made by the truncation
    """
    if tolerance == 0:
        #log('--no truncation')
        return S, 0
    # We sum all reduced density matrix eigenvalues, starting from
    # the smallest ones, to avoid rounding errors
    err = np.cumsum(np.flip(S, axis=0)**2)
    #
    # This is the sum of all values
    total = err[-1]
    #
    # The vector err[0],err[1],...,err[k] is the error that we make
    # by keeping the singular values from 0 to N-k
    # We find the first position that is above tolerance err[k],
    # which tells us which is the smallest singular value that we
    # have to keep, s[k-1]
    #
    ndx = np.argmax(err >= tolerance*total)
    if dimension is not None:
        ndx = min(ndx, dimension)
    #
    # We use that to estimate the size of the array and the actual error
    #
    return S[0:(S.size - ndx)], err[ndx-1] if ndx else 0.


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
    def __init__(self, data, error=0.):
        super(MPS, self).__init__(data)
        assert data[0].shape[0] == data[-1].shape[-1] == 1
        self._error = error

    def dimension(self):
        """Return the total size of the Hilbert space in which this MPS lives."""
        return np.product([a.shape[1] for a in self._data])

    def tovector(self):
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return _mps2vector(self)

    @staticmethod
    def fromvector(ψ, dimensions, **kwdargs):
        return MPS(vector2mps(ψ, dimensions, **kwdargs))

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖**2 = <ψ|ψ>."""
        return expectation.scprod(self, self)

    def expectation1(self, operator, n):
        """Return the expectation value of 'operator' acting on the 'n'-th
        site of the MPS."""
        return expectation.expectation1(self, operator, n)

    def expectation2(self, operator1, operator2, n):
        """Return the expectation value of 'operator1' and 'operator2' acting
        on the 'n'-th and 'n+1'-th sites of the MPS."""
        return expectation.expectation2(self, operator1, operator2, n)

    def all_expectation1(self, operator):
        """Return all expectation values of 'operator' acting on all possible
        sites of the MPS."""
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
        precisely, |exact - actual|^2."""
        return self._error

    def update_error(self, delta):
        """Update an estimate of the norm-2 truncation errors. We use the
        triangle inequality to guess an upper bound."""
        self._error = (np.sqrt(self._error)+np.sqrt(delta))**2
        return self._error

    def extend(self, L, sites=None, dimensions=2):
        """Enlarge an MPS so that it lives in a Hilbert space with 'L' sites.

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

        data = [None]*L
        for (ndx, A) in zip(sites, self):
            data[ndx] = A
            dimensions[ndx] = A.shape[1]
        D = 1
        for (i, A) in enumerate(data):
            if A is None:
                d = dimensions[i]
                A = np.zeros((D,d,D))
                A[:,0,:] = np.eye(D)
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
        return MPS([to_tensor(v) for v in vectors])
    


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
    assert n > 1
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


def gaussian(n, x0, w0, k0, mps=True):
    #
    # Return a W state with `n` components in MPS form or
    # in vector form
    #
    xx = np.arange(n, dtype=complex)
    coefs = np.exp(-(xx-x0)**2 / w0**2 + 1j * k0*xx, dtype=complex)
    return wavepacket(coefs / np.linalg.norm(coefs))

def _ortho_right(A, tol, normalize):
    α, i, β = A.shape
    U, s, V = scipy.linalg.svd(np.reshape(A, (α*i, β)), full_matrices=False,
                               lapack_driver='gesvd')
    s, err = _truncate_vector(s, tol, None)
    if normalize:
        s /= np.linalg.norm(s)
    D = s.size
    return np.reshape(U[:,:D], (α, i, D)), np.reshape(s, (D, 1)) * V[:D, :], err


def _ortho_left(A, tol, normalize):
    α, i, β = A.shape
    U, s, V = scipy.linalg.svd(np.reshape(A, (α, i*β)), full_matrices=False,
                               lapack_driver='gesvd')
    s, err = _truncate_vector(s, tol, None)
    if normalize:
        s /= np.linalg.norm(s)
    D = s.size
    return np.reshape(V[:D,:], (D, i, β)), U[:, :D] * np.reshape(s, (1, D)), err


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
    (see _truncate_vector in File 1a - MPS class)           
    """
    if direction > 0:
        if site+1 == Ψ.size:
            Ψ[site] = A
            err = 0.
        else:
            Ψ[site], sV, err = _ortho_right(A, tolerance, normalize)
            site += 1
            Ψ[site] = np.einsum('ab,bic->aic', sV, Ψ[site])
    else:
        if site == 0:
            Ψ[site] = A
            err = 0.
        else:
            Ψ[site], Us, err = _ortho_left(A, tolerance, normalize)
            site -= 1
            Ψ[site] = np.einsum('aib,bc->aic', Ψ[site], Us)
    return site, err


def _canonicalize(Ψ, center, tolerance, normalize):
    err = 0.
    for i in range(0, center):
        center, errk = _update_in_canonical_form(Ψ, Ψ[i], i, +1, tolerance, normalize)
        err += errk
    for i in range(Ψ.size-1, center, -1):
        center, errk = _update_in_canonical_form(Ψ, Ψ[i], i, -1, tolerance, normalize)
        err += errk
    return err

def left_orth_2site(AA, tolerance, normalize, dimension):
    α, d1, d2, β = AA.shape
    Ψ = np.reshape(AA, (α*d1, β*d2))
    U, S, V = scipy.linalg.svd(Ψ, full_matrices=False, lapack_driver='gesvd')
    S, err = _truncate_vector(S, tolerance, None)
    if normalize:
        S /= np.linalg.norm(S)
    D = S.size
    U = np.reshape(U[:,:D], (α, d1, D))
    SV = np.reshape( np.reshape(S, (D,1)) * V[:D,:], (D,d2,β) )
    return U, SV, err

def right_orth_2site(AA, tolerance, normalize, dimension):
    α, d1, d2, β = AA.shape
    Ψ = np.reshape(AA, (α*d1, β*d2))
    U, S, V = scipy.linalg.svd(Ψ, full_matrices=False, lapack_driver='gesvd')
    S, err = _truncate_vector(S, tolerance, dimension)
    if normalize:
        S /= np.linalg.norm(S)
    D = S.size    
    US = np.reshape(U[:,:D] * np.reshape(S, (1, D)), (α, d1, D))
    V = np.reshape(V[:D,:], (D,d2,β))
    return US, V, err


class CanonicalMPS(MPS):
    """Canonical MPS class.

    This implements a Matrix Product State object with open boundary
    conditions, that is always on canonical form with respect to a given site.
    The tensors have three indices, A[α,i,β], where 'α,β' are the internal
    labels and 'i' is the physical state of the given site.

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
    def __init__(self, data, error=0, center=None, normalize=False, tolerance=DEFAULT_TOLERANCE):
        super(CanonicalMPS, self).__init__(data, error=error)
        if isinstance(data, CanonicalMPS):
            self.center = data.center
            self._error = data._error
            if center is not None:
                self.recenter(center, tolerance, normalize)
        else:
            self.center = center = self._interpret_center(0 if center is None else center)
            self.update_error(_canonicalize(self, center, tolerance, normalize))
        if normalize:
            A = self[center]
            self[center] = A / np.linalg.norm(A)

    @classmethod
    def fromvector(ψ, dimensions, center=0, normalize=False,
                   tolerance=DEFAULT_TOLERANCE):
        return CanonicalMPS(mps.state.vector2mps(ψ, dimensions, tolerance),
                            center=center, normalize=normalize,
                            tolerance=tolerance)

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖**2 = <ψ|ψ>."""
        A = self._data[self.center]
        return np.vdot(A, A)
    
    def left_environment(self, site):
        start = min(site, center)
        ρ = expectation.begin_environment(self[start].shape[0])
        for A in self[start:site]:
            ρ = expectation.update_left_environment(A, A, ρ)
        return ρ
    
    def right_environment(self, site):
        start = max(site, center)
        ρ = expectation.begin_environment(self[start].shape[-1])
        for A in self[start:site:-1]:
            ρ = expectation.update_left_environment(A, A, ρ)
        return ρ
    
    def expectation1(self, operator, site=None):
        """Return the expectated value of 'operator' acting on the given 'site'."""
        if site is None or site == self.center:
            A = self._data[self.center]
            return np.vdot(A, np.einsum('ij,ajb->aib', operator, A))
        else:
            return expectation.expectation1(ψ, operator, site)

    def entanglement_entropyAtCenter(self):
        d1, d2, d3 = self._data[self.center].shape
        u,s,v = np.linalg.svd(np.reshape(self._data[self.center], (d1*d2,d3)))
        return -np.sum(2 * s * s * np.log2(s))
    
    def update_canonical(self, A, direction, tolerance=DEFAULT_TOLERANCE, normalize=False):
        self.center, err = _update_in_canonical_form(self, A, self.center,
                                                     direction, tolerance, normalize)
        self.update_error(err)
        return err
        
    def update_2site(self, AA, site, direction, tolerance=DEFAULT_TOLERANCE, normalize=False, dimension=None):
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
        (see _truncate_vector in File 1a - MPS class)           
        """
        assert site <= self.center <= site+1
        if direction < 0:
            self._data[site], self._data[site+1], err = right_orth_2site(AA, tolerance, normalize, dimension)
            self.center = site
        else:
            self._data[site], self._data[site+1], err = left_orth_2site(AA, tolerance, normalize, dimension)
            self.center = site+1
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
        return type(self)(self, self.center)

    def copy(self):
        """Return a fresh new TensorArray that shares the same tensor as its
        sibling, but which can be destructively modified without affecting it.
        """
        return self.__copy__()
