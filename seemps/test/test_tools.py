import unittest
import numpy as np
from . import tools


class TestTools(unittest.TestCase):
    def test_random_isometry(self):
        for N in range(1, 10):
            for M in range(1, 10):
                A = tools.random_isometry(N, M)
                self.assertTrue(tools.almostIsometry(A))

    def test_random_Pauli(self):
        for N in range(100):
            σ = tools.random_Pauli()
            self.assertTrue(tools.almostIdentity(σ @ σ))
            self.assertTrue(np.sum(np.abs(σ.T.conj() - σ)) == 0)
