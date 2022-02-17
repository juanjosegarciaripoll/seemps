import unittest
from mps.state import CanonicalMPS
from mps.test.tools import *
from mps.expectation import *

def bit2state(b):
    if b:
        return [0,1]
    else:
        return [1,0]

class TestExpectation(unittest.TestCase):
    
    def test_scprod_basis(self):
        #
        # Test that scprod() can be used to project onto basis states
        for nbits in range(1,8):
            # We create a random MPS
            ψmps = mps.state.random(2, nbits, 2)
            ψwave = ψmps.tovector()
            
            # We then create the basis of all states with well defined
            # values of the qubits
            conf = np.arange(0, 2**nbits, dtype=np.uint8)
            conf = np.reshape(conf, (2**nbits, 1))
            conf = np.unpackbits(conf, axis=1)
            
            # Finally, we loop over the basis states, verifying that the
            # scalar product is the projection onto the state
            for (n, bits) in enumerate(conf):
                proj = ψwave[n]
                ϕmps = mps.state.product(map(bit2state, bits[-nbits:]))
                self.assertEqual(proj, scprod(ϕmps, ψmps))

    def test_norm_standard(self):
        #
        # Test the norm on our sample states
        for nbits in range(1, 8):
            self.assertAlmostEqual(mps.state.GHZ(nbits).norm2(),
                                   1.0, places=10)
            self.assertAlmostEqual(mps.state.W(nbits).norm2(),
                                   1.0, places=10)
            if nbits > 1:
                self.assertAlmostEqual(mps.state.AKLT(nbits).norm2(),
                                       1.0, places=10)
                self.assertAlmostEqual(mps.state.graph(nbits).norm2(),
                                       1.0, places=10)
        
    def test_norm_random(self):
        # Test that the norm works on random states
        for nbits in range(1,8):
            for _ in range(10):
                # We create a random MPS
                ψmps = mps.state.random(2, nbits, 2)
                ψwave = ψmps.tovector()
                self.assertAlmostEqual(ψmps.norm2(), np.vdot(ψwave,ψwave))

    def test_expected1_standard(self):
        O = np.array([[0,0],[0,1]])      
        for nbits in range(1,8):
            ψGHZ = mps.state.GHZ(nbits)
            ψW = mps.state.W(nbits)
            for i in range(nbits):
                self.assertAlmostEqual(ψGHZ.expectation1(O,i), 0.5)
                self.assertAlmostEqual(ψW.expectation1(O,i), 1/nbits)

    def test_expected1(self):
        O1 = np.array([[0.3,1.0+0.2j],[1.0-0.2j,0.5]])
        def expected1_ok(ϕ, canonical=False):
            if canonical:
                for i in range(ϕ.size):
                    expected1_ok(CanonicalMPS(ϕ, center=i), canonical=False)
            else:
                nrm2 = ϕ.norm2()
                for n in range(ϕ.size):
                    ψ = ϕ.copy()
                    ψ[n] = np.einsum('ij,kjl->kil', O1, ψ[n])
                    desired= np.vdot(ϕ.tovector(), ψ.tovector())
                    self.assertAlmostEqual(desired/nrm2, expectation1(ϕ, O1, n)/nrm2)
                    self.assertAlmostEqual(desired/nrm2, ϕ.expectation1(O1, n)/nrm2)
        test_over_random_mps(expected1_ok)
        test_over_random_mps(lambda ϕ: expected1_ok(ϕ, canonical=True))
    
    def test_expected1_density(self):
        def random_wavefunction(n):
            ψ = np.random.rand(n) - 0.5
            return ψ / np.linalg.norm(ψ)
        #
        # When we create a spin wave, 'O' detects the density of the
        # wave with the right magnitude
        O = np.array([[0,0],[0,1]])
        for nbits in range(2,14):
            for _ in range(10):
                # We create a random MPS
                ψwave = random_wavefunction(nbits)
                ψmps = mps.state.wavepacket(ψwave)
                ni = all_expectation1(ψmps, O)
                for i in range(nbits):
                    si = expectation1(ψmps, O, i)
                    self.assertAlmostEqual(si, ψwave[i]**2)
                    xi = ψmps.expectation1(O, i)
                    self.assertEqual(si, xi)
                    self.assertAlmostEqual(ni[i], si)

    def test_expected2_GHZ(self):
        σz = np.array([[1,0],[0,-1]])      
        for nbits in range(2,8):
            ψGHZ = mps.state.GHZ(nbits)
            for i in range(nbits-1):
                self.assertAlmostEqual(expectation2(ψGHZ,σz,σz,i), 1)
                self.assertAlmostEqual(ψGHZ.expectation2(σz,σz,i), 1)

    def test_expected2(self):
        O1 = np.array([[0.3,1.0+0.2j],[1.0-0.2j,0.5]])
        O2 = np.array([[0.34,0.4-0.7j],[0.4+0.7j,-0.6]])
        def expected2_ok(ϕ, canonical=False):
            if canonical:
                for i in range(ϕ.size):
                    CanonicalMPS(ϕ, center=i)
            nrm2 = ϕ.norm2()
            for n in range(1, ϕ.size):
                ψ = ϕ.copy()
                ψ[n-1] = np.einsum('ij,kjl->kil', O1, ψ[n-1])
                ψ[n]   = np.einsum('ij,kjl->kil', O2, ψ[n])
                desired= mps.expectation.scprod(ϕ, ψ)
                self.assertAlmostEqual(desired/nrm2, expectation2(ϕ, O1, O2, n-1)/nrm2)
                self.assertAlmostEqual(desired/nrm2, ϕ.expectation2(O1, O2, n-1)/nrm2)
        test_over_random_mps(expected2_ok)
        test_over_random_mps(lambda ϕ: expected2_ok(ϕ, canonical=True))
