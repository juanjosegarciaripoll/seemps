import unittest
import mps.state
from mps.test.tools import *


class TestCanonicalForm(unittest.TestCase):

    def test_local_update_canonical(self):
        #
        # We verify that _update_in_canonical_form() leaves a tensor that
        # is an approximate isometry.
        #
        def ok(Ψ):
            for i in range(Ψ.size-1):
                ξ = Ψ.copy()
                _update_in_canonical_form(ξ, ξ[i], i, +1,
                                          DEFAULT_TOLERANCE)
                self.assertTrue(approximateIsometry(ξ[i], +1))
            for i in range(1, Ψ.size):
                ξ = Ψ.copy()
                _update_in_canonical_form(ξ, ξ[i], i, -1,
                                          DEFAULT_TOLERANCE)
                self.assertTrue(approximateIsometry(ξ[i], -1))

        test_over_random_mps(ok)

    def test_canonicalize(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ = Ψ.copy()
                _canonicalize(ξ, center, DEFAULT_TOLERANCE)
                #
                # All sites to the left and to the right are isometries
                #
                for i in range(center):
                    self.assertTrue(approximateIsometry(ξ[i], +1))
                for i in range(center+1, ξ.size):
                    self.assertTrue(approximateIsometry(ξ[i], -1))
                #
                # Both states produce the same wavefunction
                #
                self.assertTrue(similar(ξ.tovector(), Ψ.tovector()))
        test_over_random_mps(ok)

    def test_canonical_mps(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ = CanonicalMPS(Ψ, center=center)
                #
                # All sites to the left and to the right are isometries
                #
                for i in range(center):
                    self.assertTrue(approximateIsometry(ξ[i], +1))
                for i in range(center+1, ξ.size):
                    self.assertTrue(approximateIsometry(ξ[i], -1))
                #
                # Both states produce the same wavefunction
                #
                self.assertTrue(similar(ξ.tovector(), Ψ.tovector()))
                #
                # The norm is correct
                #
                self.assertAlmostEqual(ξ.norm2()/Ψ.norm2(), 1.0)
                #
                # Local observables give the same
                #
                O = np.array([[0, 0], [0, 1]])

                self.assertAlmostEqual(ξ.expectationAtCenter(O),
                                       Ψ.expectation1(O, center))
                #
                # The canonical form is the same when we use the
                # corresponding negative indices of 'center'
                #
                χ = CanonicalMPS(Ψ, center=center-Ψ.size)
                for i in range(Ψ.size):
                    self.assertTrue(similar(ξ[i], χ[i]))
        test_over_random_mps(ok)

    def test_canonical_mps_normalization(self):
        #
        # We verify CanonicalMPS(...,normalize=True) normalizes the
        # vector without really changing it.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ1 = CanonicalMPS(Ψ, center=center, normalize=False)
                ξ2 = CanonicalMPS(Ψ, center=center, normalize=True)
                self.assertAlmostEqual(ξ2.norm2(), 1.0)
                self.assertTrue(similar(ξ1.tovector()/np.sqrt(ξ1.norm2()),
                                        ξ2.tovector()))
        test_over_random_mps(ok)

    def test_canonical_mps_copy(self):
        #
        # Copying a class does not invoke _canonicalize and does not
        # change the tensors in any way
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ψ = CanonicalMPS(Ψ, center=center, normalize=True)
                ξ = ψ.copy()
                self.assertEqual(ξ.size, ψ.size)
                self.assertEqual(ξ.center, ψ.center)
                for i in range(ξ.size):
                    self.assertTrue(np.all(np.equal(ξ[i], ψ[i])))
        test_over_random_mps(ok)
        
    def test_local_update_canonical_2site(self):
        #
        # We verify that _update_in_canonical_form_2site() leaves 
        # a tensor that is an approximate isometry.
        #
        def ok(Ψ):
            for i in range(Ψ.size-1):
                ξ = Ψ.copy()
                AA = np.einsum("ijk,klm -> ijlm",  ξ[i],  ξ[i+1])
                _update_in_canonical_form_2site(ξ, AA, i, +1,
                                          DEFAULT_TOLERANCE)
                self.assertTrue(approximateIsometry(ξ[i], +1))                
            for i in range(1, Ψ.size):
                ξ = Ψ.copy()
                AA = np.einsum("ijk,klm -> ijlm",  ξ[i-1],  ξ[i])
                _update_in_canonical_form_2site(ξ, AA, i, -1,
                                          DEFAULT_TOLERANCE)
                self.assertTrue(approximateIsometry(ξ[i], -1))

        test_over_random_mps(ok)
    
