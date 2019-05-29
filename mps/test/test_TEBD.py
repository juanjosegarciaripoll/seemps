from mps.evolution import *

import unittest
import mps.state
import mps.tools
from mps.test.tools import *
import scipy.sparse as sp

def random_wavefunction(n):
    ψ = np.random.rand(n) - 0.5
    return ψ / np.linalg.norm(ψ)

class TestTEBD_sweep(unittest.TestCase):
    
    def test_apply_pairwise_unitaries(self):
        N = 20
        tt = -np.pi/2
        ω = np.pi
        dt = 1e-3
        ψwave = random_wavefunction(N)
        ψ = CanonicalMPS(mps.state.wavepacket(ψwave))
        # We use the tight-binding Hamiltonian
        HMat = sp.diags([[tt,0]*(N//2), [ω]+[ω/2]*(N-2)+[ω], [tt,0]*(N//2)],
                  offsets=[-1,0,+1],
                  shape=(N,N),
                  dtype=np.complex128)
        ψwave_final = sp.linalg.expm_multiply(-1j * dt * HMat, ψwave)
        
        H=make_ti_Hamiltonian(N, [tt * annihilation_op(2) ,tt * creation_op(2)], 
                            [creation_op(2), annihilation_op(2)], 
                              local_term=ω*creation_op(2)@ annihilation_op(2))
        U = pairwise_unitaries(H, dt)
        start = 0
        direction = 1
        apply_pairwise_unitaries(U, ψ, start, direction, tol=DEFAULT_TOLERANCE)
        
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψ.tovector())))
        
                  
    
