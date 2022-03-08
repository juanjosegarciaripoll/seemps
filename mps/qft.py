import numpy as np
from numpy import pi as π
import math
from mps.state import MPS
from mps.mpo import MPOList, MPO

def qft_mpo(N, sign=-1, **kwargs):
    """Create an MPOList object representing a Quantum Fourier Transform
    for a quantum register with `N` qubits.
    
    Parameters
    ----------
    N         -- Number of qubits in a quantum register
    kwargs   -- All other arguments accepted by MPO
    
    Output
    ------
    mpolist   -- An MPOList object that can be applied `@` to an MPS.
    """
    
    def fix_last(mpo_list):
        A = mpo_list[-1]
        shape = A.shape
        A = np.sum(A, -1).reshape(shape[0],shape[1],shape[2],1)
        return mpo_list[:-1]+[A]
    
    # Tensor doing nothing
    noop = np.eye(2).reshape(1,2,2,1)
    #
    # Beginning Hadamard
    H = np.array([[1, 1],[1,-1]])/np.sqrt(2.)
    Hop = np.zeros((1,2,2,2))
    Hop[0,1,:,1] = H[1,:]
    Hop[0,0,:,0] = H[0,:]
    #
    # Conditional rotations
    R0 = np.zeros((2,2,2,2))
    R0[0,0,0,0] = 1.0
    R0[0,1,1,0] = 1.0
    R0[1,0,0,1] = 1.0
    R1 = np.zeros((2,2,2,2))
    R1[1,1,1,1] = 1.0
    jϕ = sign*1j*π
    rots = [R0 + R1 * np.exp(jϕ/(2**n)) for n in range(1, N)]
    #
    return MPOList([MPO(fix_last([noop]*n + [Hop] + rots[:N-n-1]), **kwargs)
                    for n in range(0, N)], **kwargs)

def iqft_mpo(N, **kwargs):
    """Implement the inverse of the qft_mpo() operator."""
    return qft_mpo(N, +1, **kwargs)

def qft(Ψmps, **kwargs):
    """Apply the quantum Fourier transform onto a quantum register
    of qubits encoded in the matrix product state 'Ψ'"""
    return qft_mpo(len(Ψmps), sign=-1, **kwargs).apply(Ψmps)

def iqft(Ψmps, **kwargs):
    """Apply the inverse quantum Fourier transform onto a quantum register
    of qubits encoded in the matrix product state 'Ψ'"""
    return qft_mpo(len(Ψmps), sign=+1, **kwargs).apply(Ψmps)

def qft_flip(Ψmps):
    """Swap the qubits in the quantum register, to fix the reversal
    suffered during the quantum Fourier transform."""
    return MPS([np.moveaxis(A, [0,1,2],[2,1,0]) for A in reversed(Ψmps)], error=Ψmps.error())

def qft_wavefunction(Ψ):
    N = int(round(math.log2(Ψ.size)))
    return np.fft.fft(Ψ)/np.sqrt(Ψ.size)

def qft_nd_mpo(sites, N=None, sign=-1, **kwargs):
    """Create an MPOList object representing a Quantum Fourier Transform
    for subset of qubits in a quantum register with `N` qubits.
    
    Parameters
    ----------
    sites     -- Sites on which to apply the QFT, in order of decreasing
                 significance.
    N         -- Number of qubits in a quantum register.
                 Defaults to `max(sites)+1`.
    sign      -- Sign of the FFT (defaults to -1, direct FFT)
    kwargs   -- All other arguments accepted by `MPO`
    
    Output
    ------
    mpolist   -- An MPOList object that can be applied `@` to an MPS.
    """
    if N is None:
        N = max(sites)+1
    #
    # Construct a bare transformation that does nothing
    small_noop = np.eye(2).reshape(1,2,2,1)
    noop = np.eye(2).reshape(2,1,1,2) * small_noop
    #
    # Beginning Hadamard
    H = np.array([[1, 1],[1,-1]])/np.sqrt(2.)
    Hop = np.zeros((2,2,2,2))
    Hop[1,1,:,1] = H[1,:]
    Hop[0,0,:,0] = H[0,:]
    #
    # Conditional rotations
    R0 = np.zeros((2,2,2,2))
    R0[0,0,0,0] = 1.0
    R0[0,1,1,0] = 1.0
    R0[1,0,0,1] = 1.0
    R1 = np.zeros((2,2,2,2))
    R1[1,1,1,1] = 1.0
    jϕ = sign*1j*π
    #
    # Place the Hadamard and rotations according to the instructions
    # in 'sites'. The first index is the control qubit, the other ones
    # are the following qubits in order of decreasing significance.
    def make_layer(sites):
        l = [noop] * N
        for (i,ndx) in enumerate(sites):
            if i == 0:
                l[ndx] = Hop
            else:
                l[ndx] = R0 + R1 * np.exp(jϕ/(2**i))
        for (n,A) in enumerate(l):
            if A is noop:
                l[n] = small_noop
            else:
                a, i, j, b = A.shape
                l[n] = np.sum(A,0).reshape(1,i,j,b)
                break
        for n in reversed(range(N)):
            A = l[n]
            if A is noop:
                l[n] = small_noop
            else:
                a, i, j, b = A.shape
                l[n] = np.sum(A,-1).reshape(a,i,j,1)
                break
        return MPO(l, **kwargs)
    #
    return MPOList([make_layer(sites[i:]) for i in range(len(sites))], **kwargs)

def iqft_nd_mpo(sites, N=None, **kwargs):
    """Implement the inverse of the qft_nd_mpo() operator."""
    return qft_nd_mpo(sites, N=N, sign=+1, **kwargs)
