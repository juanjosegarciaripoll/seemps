import unittest
import mps.state
import mps.tools
from mps.evolution import *
from mps.test.tools import *
import scipy.sparse as sp


        
def random_wavefunction(n):
    ψ = np.random.rand(n) - 0.5
    return ψ / np.linalg.norm(ψ)

class TestTEBD_sweep(unittest.TestCase):
    
    def test_orthonormalization(self):
        #
        # We verify that our two-site orthonormalization procedure, 
        # does not change the state
        #
        δt = 0.1

        def ok(Ψ):
            H = make_ti_Hamiltonian(Ψ.size, [mps.tools.random_Pauli()], [mps.tools.random_Pauli()])
            Trotter = Trotter_unitaries(H, δt)
            for start in range(Ψ.size-2):
                AA = apply_2siteTrotter(Trotter.twosite_unitary(start) , 
                                                      Ψ, start)
                A, AC = mps.state.left_orth_2site(AA, DEFAULT_TOLERANCE)
                AA_orth = np.einsum("ijk,klm -> ijlm", A, AC)
                self.assertTrue(similar(AA,AA_orth))    
                
                AA = apply_2siteTrotter(Trotter.twosite_unitary(start) , 
                                                      Ψ, start)
                A, AC = mps.state.right_orth_2site(AA, DEFAULT_TOLERANCE)
                AA_orth = np.einsum("ijk,klm -> ijlm", AC, A)
                self.assertTrue(similar(AA,AA_orth))
            
            
        test_over_random_mps(ok)
        
    def test_right_sweep(self):
        #
        # We verify the truncation procedure does not change the resulting state. 
        # We evolve the mps only through one sweep to eliminate the error due to 
        # non-commuting terms in the Hamiltonian. We compare the TEBD results with 
        # exact diagonalization results.
        #
        # Note that there is a phase difference between the two wavefunctions.
        # However absolute values of the corresponding coefficients are equal 
        # as the test verifies.
        #             
        N = 6
        t = 1
        ω = 0.5
        dt = 1
        ψwave = random_wavefunction(N)
        ψmps = CanonicalMPS(mps.state.wavepacket(ψwave), center=0)
        # We use the tight-binding Hamiltonian
        # Test for evenodd = 0      
        H=make_ti_Hamiltonian(N, [t * annihilation_op(2) , t * creation_op(2)], 
                            [creation_op(2), annihilation_op(2)], 
                              local_term=ω*creation_op(2)@ annihilation_op(2))
        ψmps_final = TEBD_sweep(H, ψmps, dt, 1, 0, tol=DEFAULT_TOLERANCE)
        Hmat = sp.diags([[t,0]*(N//2), [ω]+[ω/2]*(N-2)+[ω], [t,0]*(N//2)],
                  offsets=[-1,0,+1],
                  shape=(N,N),
                  dtype=np.complex128)
        ψwave_final = sp.linalg.expm_multiply(-1j * dt * Hmat, ψwave)
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps_final.tovector())))
        
        # Test for evenodd = 1      
        ψmps = CanonicalMPS(mps.state.wavepacket(ψwave), center=1)
        ψmps_final = TEBD_sweep(H, ψmps, dt, 1, 1, tol=DEFAULT_TOLERANCE)
        Hmat = sp.diags([[0,t]*(N//2), [ω/2]*(N-1)+[ω], [0,t]*(N//2)],
                  offsets=[-1,0,+1],
                  shape=(N,N),
                  dtype=np.complex128)
        ψwave_final = sp.linalg.expm_multiply(-1j * dt * Hmat, ψwave)
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps_final.tovector())))
     
    
    def test_left_sweep(self):
        #
        # We verify the truncation procedure does not change the resulting state. 
        # We evolve the mps only through one sweep to eliminate the error due to 
        # non-commuting terms in the Hamiltonian. We compare the TEBD results with 
        # exact diagonalization results.
        #
        # Note that there is a phase difference between the two wavefunctions.
        # However absolute values of the corresponding coefficients are equal 
        # as the test verifies.
        #       
        N = 5
        t = 0.1
        ω = 0.5
        dt = 1
        ψwave = random_wavefunction(N)
        ψmps = CanonicalMPS(mps.state.wavepacket(ψwave), center=N-2)
        # We use the tight-binding Hamiltonian
        # Test for evenodd = 0        
        H=make_ti_Hamiltonian(N, [t * annihilation_op(2) , t * creation_op(2)], 
                            [creation_op(2), annihilation_op(2)], 
                              local_term=ω*creation_op(2)@ annihilation_op(2))
        
  
        ψmps_final = TEBD_sweep(H, ψmps, dt, -1, 0, tol=DEFAULT_TOLERANCE)
        Hmat = sp.diags([[t,0]*(N//2), [ω]+[ω/2]*(N-2)+[ω], [t,0]*(N//2)],
                  offsets=[-1,0,+1],
                  shape=(N,N),
                  dtype=np.complex128)
        ψwave_final = sp.linalg.expm_multiply(-1j * dt * Hmat, ψwave)
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps_final.tovector())))
     
        
        
    def test_TEBD_evolution(self):
        #
        #
        #
        N = 19
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
        for i in range(Nt):
            
            ψmps = TEBD_sweep(H, ψmps, dt, 1, 0, tol=DEFAULT_TOLERANCE)
            ψmps = TEBD_sweep(H, ψmps, dt, -1, 1, tol=DEFAULT_TOLERANCE)
        
        Hmat = sp.diags([[t,0]*(N//2), ω, [t,0]*(N//2)],
                  offsets=[-1,0,+1],
                  shape=(N,N),
                  dtype=np.complex128)
        ψwave_final = sp.linalg.expm_multiply(-1j * dt*Nt * Hmat, ψwave)
        
        self.assertTrue(similar(abs(mps.state.wavepacket(ψwave_final).tovector()), 
                                abs(ψmps.tovector())))
                


    
