import unittest
import mps.state
import mps.tools
from tools import *
from mps.tools import σx, σy, σz
import scipy.sparse as sp
import scipy.sparse.linalg
i2 = sp.eye(2)

class TestHamiltonians(unittest.TestCase):
    
    def test_nn_construct(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, σx)
        M2 = H2.interaction_term(0)
        A2 = sp.kron(σx, i2)
        self.assertTrue(similar(M2, A2))
    
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(1, σy)
        M2 = H2.interaction_term(0)
        A2 = sp.kron(i2, σy)
        self.assertTrue(similar(M2, A2))

        H3 = ConstantNNHamiltonian(3, 2)
        H3.add_local_term(1, σy)
        M3 = H3.interaction_term(0)
        A3 = sp.kron(i2, 0.5*σy)
        self.assertTrue(similar(M3, A3))
        M3 = H3.interaction_term(1)
        A3 = sp.kron(0.5*σy, i2)
        self.assertTrue(similar(M3, A3))
    
    def test_sparse_matrix(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_interaction_term(0, σz, σz)
        M2 = H2.tomatrix()
        A2 = sp.kron(σz,σz)
        self.assertTrue(similar(M2, A2))
        
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, 3.5*σx)
        M2 = H2.tomatrix()
        A2 = sp.kron(3.5*σx, i2)
        self.assertTrue(similar(M2, A2))
        
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(1, -2.5*σy)
        M2 = H2.tomatrix()
        A2 = sp.kron(i2, -2.5*σy)
        self.assertTrue(similar(M2, A2))
        
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, 3.5*σx)
        H2.add_local_term(1, -2.5*σy)
        H2.add_interaction_term(0, σz, σz)
        M2 = H2.tomatrix()
        A2 = sp.kron(i2, -2.5*σy) + sp.kron(σz,σz) + sp.kron(3.5*σx, i2)
        self.assertTrue(similar(M2, A2))
import unittest
import scipy.sparse as sp
import scipy.sparse.linalg
from mps.state import CanonicalMPS
from mps.tools import *
from tools import *
from mps.evolution import *
from mps.hamiltonians import make_ti_Hamiltonian, ConstantNNHamiltonian

def random_wavefunction(n):
    ψ = np.random.rand(n) - 0.5
    return ψ / np.linalg.norm(ψ)

class TestTEBD_sweep(unittest.TestCase):
    
    @staticmethod
    def hopping_model(N, t, ω):
        a = annihilation(2)
        ad = creation(2)
        return make_ti_Hamiltonian(N, [t*a, t*ad], [ad, a], local_term = ω*(ad@a))

    @staticmethod
    def hopping_model_Trotter_matrix(N, t, ω):
        #
        # Hamiltonian that generates the evolution of the odd hoppings
        # and local frequencies
        return sp.diags([[t,0]*(N//2), [ω]+[ω/2]*(N-2)+[ω], [t,0]*(N//2)],
                        offsets=[-1,0,+1], shape=(N,N), dtype=np.float64)
    
    @staticmethod
    def hopping_model_matrix(N, t, ω):
        return sp.diags([[t]*(N), ω, [t]*(N)], offsets=[-1,0,+1], shape=(N,N))

    def inactive_test_apply_pairwise_unitaries(self):
        N = 2
        tt = -np.pi/2
        ω = np.pi
        dt = 0.1
        #
        # Numerically exact solution using Scipy's exponentiation routine
        ψwave = random_wavefunction(N)
        print(mps.state.wavepacket(ψwave).tovector())
        HMat = self.hopping_model_Trotter_matrix(N, tt, ω)
        ψwave_final = sp.linalg.expm_multiply(+1j * dt * HMat, ψwave)
        print(mps.state.wavepacket(ψwave_final).tovector())
        print(HMat.todense())
        #
        # Evolution using Trrotter
        H = self.hopping_model(N, tt, ω)
        U = pairwise_unitaries(H, dt)
        ψ = CanonicalMPS(mps.state.wavepacket(ψwave))
        start = 0
        direction = 1
        apply_pairwise_unitaries(U, ψ, start, direction, tol=DEFAULT_TOLERANCE)
        print(ψ.tovector())
        print(np.abs(mps.state.wavepacket(ψwave_final).tovector() - ψ.tovector()))
        
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψ.tovector())))
        
    def test_TEBD_evolution_first_order(self):
        #
        #
        #
        N = 19
        t = - np.pi/2
        ω = np.pi
        dt = 1e-6
        Nt = int(1000)
        #ψwave = random_wavefunction(N)
        xx=np.arange(N)
        x0 = int(N//2)
        w0 = 5
        k0 = np.pi/2
        #
        # Approximate evolution of a wavepacket in a tight-binding model
        ψwave = np.exp(-(xx-x0)**2 / w0**2 + 1j * k0*xx) 
        ψwave = ψwave / np.linalg.norm(ψwave)
        Hmat = self.hopping_model_matrix(N, t, ω)
        ψwave_final = sp.linalg.expm_multiply(-1j * dt* Nt * Hmat, ψwave)
        #
        # Trotter solution
        ψmps = CanonicalMPS(mps.state.wavepacket(ψwave))
        H = self.hopping_model(N, t, ω)
        ψmps = TEBD_evolution(ψmps, H, dt, timesteps=Nt, order=1, tol=DEFAULT_TOLERANCE).evolve()
        
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps.tovector())))               
        
    def test_TEBD_evolution_second_order(self):
        #
        #
        #
        N = 21
        t = 0.1
        ω = 0.5
        dt = 1e-6
        Nt = int(1000)
        #ψwave = random_wavefunction(N)
        xx=np.arange(N)
        x0 = int(N//2)
        w0 = 5
        k0 = np.pi/2
        #
        # Approximate evolution of a wavepacket in a tight-binding model
        ψwave = np.exp(-(xx-x0)**2 / w0**2 + 1j * k0*xx) 
        ψwave = ψwave / np.linalg.norm(ψwave) 
        Hmat = self.hopping_model_matrix(N, t, ω)
        ψwave_final = sp.linalg.expm_multiply(-1j * dt * Nt * Hmat, ψwave)
        #
        # Trotter evolution
        H = self.hopping_model(N, t, ω)
        ψmps = CanonicalMPS(mps.state.wavepacket(ψwave))
        ψmps = TEBD_evolution(ψmps, H, dt, timesteps=Nt, order=2, tol=DEFAULT_TOLERANCE).evolve()
        
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps.tovector())))
