import unittest
import mps.state
from mps.tools import similar


def approximateIsometry(A, direction, places=7):
    if direction > 0:
        a, i, b = A.shape
        A = np.reshape(A, (a*i, b))
        C = A.T.conj() @ A
    else:
        b, i, a = A.shape
        A = np.reshape(A, (b, i*a))
        C = A @ A.T.conj()
    return np.all(np.isclose(C, np.eye(b), atol=10**(-places)))


class TestCanonicalForm(unittest.TestCase):

    def test_local_update_canonical(self):
        #
        # We verify that _update_in_canonical_form() leaves a tensor that
        # is an approximate isometry.
        #
        for nqubits in range(2, 10):
            for _ in range(20):
                Ψ = mps.state.random(2, nqubits, 10)

                for i in range(Ψ.size-1):
                    ξ = Ψ.copy()
                    _update_in_canonical_form(ξ, ξ[i], i, +1)
                    self.assertTrue(approximateIsometry(ξ[i], +1))

                for i in range(1, Ψ.size):
                    ξ = Ψ.copy()
                    _update_in_canonical_form(ξ, ξ[i], i, -1)
                    self.assertTrue(approximateIsometry(ξ[i], -1))

    def test_canonicalize(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        for nqubits in range(2, 10):
            for _ in range(20):
                Ψ = mps.state.random(2, nqubits, 10)

                for center in range(Ψ.size):
                    ξ = Ψ.copy()
                    _canonicalize(ξ, center)
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

    def test_canonical_mps(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        for nqubits in range(2, 8):
            for _ in range(20):
                Ψ = mps.state.random(2, nqubits, 10)

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
                    self.assertAlmostEqual(ξ.norm2(), Ψ.norm2())
                    #
                    # Local observables give the same
                    #
                    O = np.array([[0, 0],[0, 1]])
                    
                    self.assertAlmostEqual(ξ.expectationAtCenter(O),
                                           Ψ.expectation1(O, center))
