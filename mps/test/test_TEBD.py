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
        
    def test_TEBD_evolution_first_order(self):
        #
        #
        #
        N = 19
        t = 0.1
        ω = 0.5
        dt = 1e-7
        Nt = int(10000)
        ψwave = random_wavefunction(N)
        ψmps = CanonicalMPS(mps.state.wavepacket(ψwave))
        # We use the tight-binding Hamiltonian
        H=make_ti_Hamiltonian(N, [t * annihilation_op(2) , t * creation_op(2)], 
                            [creation_op(2), annihilation_op(2)], 
                              local_term=ω*creation_op(2)@ annihilation_op(2))
        #for i in range(Nt):
        #    
        #    ψmps = TEBD_sweep(H, ψmps, dt, 1, 0, tol=DEFAULT_TOLERANCE)
        #    ψmps = TEBD_sweep(H, ψmps, dt, -1, 1, tol=DEFAULT_TOLERANCE)
        ψmps = TEBD_evolution(H, dt, timesteps=Nt, order=1, tol=DEFAULT_TOLERANCE).evolve(ψmps)
        Hmat = sp.diags([[t]*(N), ω, [t]*(N)],
                  offsets=[-1,0,+1],
                  shape=(N,N),
                  dtype=np.complex128)
        
        ψwave_final = sp.linalg.expm_multiply(-1j * dt*Nt * Hmat, ψwave)
        
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps.tovector())))
                

    def test_TEBD_evolution_second_order(self):
        #
        #
        #
        N = 21
        t = 0.1
        ω = 0.5
        dt = 1e-7
        Nt = int(100)
        ψwave = random_wavefunction(N)
        ψmps = CanonicalMPS(mps.state.wavepacket(ψwave))
        # We use the tight-binding Hamiltonian
        H=make_ti_Hamiltonian(N, [t * annihilation_op(2) , t * creation_op(2)], 
                            [creation_op(2), annihilation_op(2)], 
                              local_term=ω*creation_op(2)@ annihilation_op(2))
        #for i in range(Nt):
        #    
        #    ψmps = TEBD_sweep(H, ψmps, dt, 1, 0, tol=DEFAULT_TOLERANCE)
        #    ψmps = TEBD_sweep(H, ψmps, dt, -1, 1, tol=DEFAULT_TOLERANCE)
        ψmps = TEBD_evolution(H, dt, timesteps=Nt, order=2, tol=DEFAULT_TOLERANCE).evolve(ψmps)
        Hmat = sp.diags([[t]*(N), ω, [t]*(N)],
                  offsets=[-1,0,+1],
                  shape=(N,N),
                  dtype=np.complex128)
        
        ψwave_final = sp.linalg.expm_multiply(-1j * dt*Nt * Hmat, ψwave)
        
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps.tovector())))
                


    
