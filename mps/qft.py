import numpy as np
from numpy import pi as π
from mps.state import MPS
from mps.mpo import MPOList, MPO

def qft_mpo(N, sign=-1, **kwdargs):
    """Create an MPOList object representing a Quantum Fourier Transform
    for a quantum register with 'N' qubits.
    
    Parameters
    ----------
    N         -- Number of qubits in a quantum register
    **kwdargs -- All other arguments accepted by MPO
    
    Output
    ------
    mpolist   -- An MPOList object that can be applied '@' to an MPS.
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
    return MPOList([MPO(fix_last([noop]*n + [Hop] + rots[:N-n-1]), **kwdargs)
                    for n in range(0, N)])

def iqft_mpo(N, **kwdargs):
    return qft_mpo(N, +1, kwdargs)

def qft(Ψmps, **kwdargs):
    """Apply the Quantum Fourier Transform onto a quantum register
    of qubits encoded in the matrix product state 'Ψ'"""
    return qft_mpo(len(Ψmps), sign=-1, **kwdargs).apply(Ψmps)

def iqft(Ψmps, **kwdargs):
    """Apply the Quantum Fourier Transform onto a quantum register
    of qubits encoded in the matrix product state 'Ψ'"""
    return qft_mpo(len(Ψmps), sign=+1, **kwdargs).apply(Ψmps)

def qft_flip(Ψmps):
    return MPS([np.moveaxis(A, [0,1,2],[2,1,0]) for A in reversed(Ψmps)], error=Ψmps.error())

def qft_wavefunction(Ψ):
    N = int(round(math.log2(Ψ.size)))
    return np.fft.fft(Ψ)/np.sqrt(Ψ.size)
