
import unittest
import mps.state
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

    def test_norm_random(self):
        #
        # Test the norm on our sample states
        for nbits in range(1, 8):
            self.assertAlmostEqual(mps.state.GHZ(nbits).norm2(),
                                   1.0, places=10)
            self.assertAlmostEqual(mps.state.W(nbits).norm2(),
                                   1.0, places=10)
            self.assertAlmostEqual(mps.state.graph(nbits).norm2(),
                                   1.0, places=10)
            self.assertAlmostEqual(mps.state.AKLT(nbits).norm2(),
                                   1.0, places=10)
        
        # Test that the norm works on random states
        for nbits in range(1,8):
            for _ in range(10):
                # We create a random MPS
                ψmps = mps.state.random(2, nbits, 2)
                ψwave = ψmps.tovector()
                self.assertAlmostEqual(ψmps.norm2(), np.vdot(ψwave,ψwave))
    
    def test_expected1_density(self):
        def random_wavefunction(n):
            ψ = np.random.rand(n) - 0.5
            return ψ / np.linalg.norm(ψ)
        
        O = np.array([[0,0],[0,1]])
        
        for nbits in range(1,8):
            ψGHZ = mps.state.GHZ(nbits)
            ψW = mps.state.W(nbits)
            for i in range(nbits):
                self.assertAlmostEqual(ψGHZ.expectation1(O,i), 0.5)
                self.assertAlmostEqual(ψW.expectation1(O,i), 1/nbits)

        #
        # When we create a spin wave, 'O' detects the density of the
        # wave with the right magnitude
        for nbits in range(2,14):
            for _ in range(10):
                # We create a random MPS
                ψwave = random_wavefunction(nbits)
                ψmps = mps.state.wavepacket(ψwave)
                ni = all_expectation1_non_canonical(ψmps, O)
                for i in range(nbits):
                    si = expectation1_non_canonical(ψmps, O, i)
                    self.assertAlmostEqual(si, ψwave[i]**2)
                    xi = ψmps.expectation1(O, i)
                    self.assertEqual(si, xi)
                    self.assertAlmostEqual(ni[i], si)
