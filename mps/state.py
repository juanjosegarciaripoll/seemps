import numpy as np


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
        """Create a new TensorArray from a list of tensors. The list is cloned
        before storing it into this object, so as to avoid side effects when
        destructively modifying the array."""
        if type(data) is list:
            self._data = data.copy()
            self.size = len(self._data)
        else:
            raise TypeError()

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
        return np.product([a.shape[1] for a in self.data])

    def tovector(self):
        """Return one-dimensional complex vector of dimension() elements, with
        the complete wavefunction that is encoded in the MPS."""
        return _mps2vector(self)

    def norm2(self):
        """Return the square of the norm-2 of this state, ‖ψ‖**2 = <ψ|ψ>."""
        return mps.expectation.scprod(self.data, self.data)

    def expectation1(self, operator, n):
        """Return the expectation value of 'operator' acting on the 'n'-th
        site of the MPS."""
        return mps.expectation.expectation1_non_canonical(self, operator, n)

    def all_expectation1(self, operator):
        """Return all expectation values of 'operator' acting on all possible
        sites of the MPS."""
        return mps.expectation.all_expectation1_non_canonical(self, operator)


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
