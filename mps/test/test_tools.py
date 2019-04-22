
import unittest
from mps.tools import *
from mps.test.tools import *

class TestTools(unittest.TestCase):
    
    def test_random_isometry(self):
        for N in range(1, 10):
            for M in range(1, 10):
                A = mps.tools.random_isometry(N, M)
                self.assertTrue(almostIsometry(A))

    def test_random_Pauli(self):
        for N in range(100):
            σ = random_Pauli()
            self.assertTrue(almostIdentity(σ @ σ))
            self.assertTrue(np.sum(np.abs(σ.T.conj() - σ)) == 0)
