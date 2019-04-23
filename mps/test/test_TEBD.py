import unittest
import mps.state
from mps.evolution import *
from mps.test.tools import *



class TestTEBD_sweep(unittest.TestCase):
    def test_orthonormalization(self):
        δt = 0.1
        tol = 0
        H = mps.evolution.TINNHamiltonian(0*σx, σx, σx)
        Trotter = mps.evolution.Trotter_unitaries(H, δt)

        def left_ok(Ψ):
            for start in range(Ψ.size-1):            
                AA = mps.evolution.apply_2siteTrotter(Trotter.twosite_unitary(start) , Ψ, start)
                A, AC = mps.state.left_orth_2site(AA,tol)
                AA_orth = np.einsum("ijk,klm -> ijlm", A, AC)
                self.assertTrue(similar(AA,AA_orth))
            
        test_over_random_mps(left_ok)

        
        def right_ok(Ψ):
            δt = 0.1
            ξ = Ψ.copy()
            start = Ψ.size //2
            H = mps.evolution.TINNHamiltonian(0*σx, σx, σx)
            Trotter = mps.evolution.Trotter_unitaries(H, δt)
            AA = mps.evolution.apply_2siteTrotter(Trotter.twosite_unitary(start) , ξ, start)
            tol = 0
            A, AC = mps.state.right_orth_2site(AA,tol)
            AA_orth = np.einsum("ijk,klm -> ijlm", AC, A)
            return np.isclose(AA,AA_orth)
            
        test_over_random_mps(right_ok)



