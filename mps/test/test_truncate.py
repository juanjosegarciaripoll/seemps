
import unittest
from mps.test.tools import *

class TestLinearForm(unittest.TestCase):
    
    def test_canonical_env(self):
        #
        # When we pass two identical canonical form MPS to LinearForm, the
        # left and right environments are the identity
        #
        def ok(ψ):
            for center in range(ψ.size):
                ϕ = CanonicalMPS(ψ, center)
                LF = AntilinearForm(ϕ, ϕ, center)
                self.assertTrue(almostIdentity(LF.L[center],+1))
                self.assertTrue(almostIdentity(LF.R[center],+1))
        
        test_over_random_mps(ok)
    
    def test_tensor(self):
        #
        # Use ϕ and ψ=O*ϕ, where O is a local operator, to verify that the
        # LinearForm of ψ and ϕ returns <ϕ|O|ϕ>
        #
        O = np.zeros((2,2))
        def ok(Φ):
            for center in range(Φ.size):
                ϕ = CanonicalMPS(Φ, center, normalize=True)
                for n in range(ϕ.size):
                    Odesired = ϕ.expectation1(O, n)
                    ψ = ϕ.copy()
                    ψ[n] = np.einsum('ij,ajb->aib', O, ψ[n])
                    LF = AntilinearForm(ψ, ϕ, center)
                    U = LF.tensor1site()
                    D = ϕ[center]
                    Oestimate = np.einsum('aib,aib', D.conj(), U)
                    self.assertAlmostEqual(Oestimate, Odesired)
                    if n >= center:
                        self.assertTrue(almostIdentity(LF.L[center],+1))
                    if n <= center:
                        self.assertTrue(almostIdentity(LF.R[center],+1))
        
        O = np.array([[0.3,0.2+1.0j],[0.2-1.0j,2.0]])
        test_over_random_mps(ok)
