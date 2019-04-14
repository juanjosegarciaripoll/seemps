
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
    
    def tensor1siteok(self, aϕ, O):
        for center in range(aϕ.size):
            ϕ = CanonicalMPS(aϕ, center, normalize=True)
            for n in range(ϕ.size):
                #
                # Take an MPS Φ, construct a new state ψ = O1*ϕ with a local
                # operator on the 'n-th' site
                #
                ψ = ϕ.copy()
                ψ[n] = np.einsum('ij,ajb->aib', O, ψ[n])
                #
                # and make sure that the AntilinearForm provides the right tensor to
                # compute <ϕ|ψ> = <ϕ|O1|ϕ>
                #
                Odesired = ϕ.expectation1(O, n)
                LF = AntilinearForm(ψ, ϕ, ϕ.center)
                Oestimate = np.einsum('aib,aib', ϕ[ϕ.center], LF.tensor1site())
                self.assertAlmostEqual(Oestimate, Odesired)
                if n >= center:
                    self.assertTrue(almostIdentity(LF.L[center],+1))
                if n <= center:
                    self.assertTrue(almostIdentity(LF.R[center],+1))

    def test_tensor1site_product(self):
        O = np.array([[0.3,0.2+1.0j],[0.2-1.0j,2.0]])
        test_over_random_mps(lambda ϕ: self.tensor1siteok(ϕ, O), D=1)

    def test_tensor1site(self):
        O = np.array([[0.3,0.2+1.0j],[0.2-1.0j,2.0]])
        test_over_random_mps(lambda ϕ: self.tensor1siteok(ϕ, O))
    
    def tensor2siteok(self, aϕ, O1, O2):
        for center in range(aϕ.size):
            ϕ = CanonicalMPS(aϕ, center, normalize=True)
            for n in range(ϕ.size-1):
                #
                # Take an MPS Φ, construct a new state ψ = O1*ϕ with a local
                # operator on the 'n-th' site
                #
                ψ = ϕ.copy()
                ψ[n] = np.einsum('ij,ajb->aib', O1, ψ[n])
                ψ[n+1] = np.einsum('ij,ajb->aib', O2, ψ[n+1])
                #
                # and make sure that the AntilinearForm provides the right tensor to
                # compute <ϕ|ψ> = <ϕ|O1|ϕ>
                #
                Odesired = ϕ.expectation2(O1, O2, n)
                LF = AntilinearForm(ψ, ϕ, center)
                if center+1 < ϕ.size:
                    D = np.einsum('ijk,klm->ijlm', ϕ[center], ϕ[center+1])
                    Oestimate = np.einsum('aijb,aijb', D, LF.tensor2site(+1))
                    self.assertAlmostEqual(Oestimate, Odesired)
                if center > 0:
                    D = np.einsum('ijk,klm->ijlm', ϕ[center-1], ϕ[center])
                    Oestimate = np.einsum('aijb,aijb', D, LF.tensor2site(-1))
                    self.assertAlmostEqual(Oestimate, Odesired)
                if n >= center:
                    self.assertTrue(almostIdentity(LF.L[center],+1))
                if n+1 <= center:
                    self.assertTrue(almostIdentity(LF.R[center],+1))

    def test_tensor2site_product(self):
        O1 = np.array([[0.3,0.2+1.0j],[0.2-1.0j,2.0]])
        O2 = np.array([[0.34,0.4-0.7j],[0.4+0.7j,-0.6]])
        test_over_random_mps(lambda ϕ: self.tensor2siteok(ϕ, O1, O2), D=1)

    def test_tensor2site(self):
        O1 = np.array([[0.3,0.2+1.0j],[0.2-1.0j,2.0]])
        O2 = np.array([[0.34,0.4-0.7j],[0.4+0.7j,-0.6]])
        test_over_random_mps(lambda ϕ: self.tensor2siteok(ϕ, O1, O2))
