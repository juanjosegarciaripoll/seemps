import unittest
import numpy as np
from mps.test.tools import *
import scipy.sparse as sp

class TestAlgebraic(unittest.TestCase):
    
    P1 = sp.diags([0.,1.],0)
    i2 = sp.eye(2, dtype=np.float64)
    
    @classmethod
    def projector(self, i, L):
        return sp.kron(sp.eye(2**i), sp.kron(self.P1, sp.eye(2**(L-i-1))))
    
    @classmethod
    def linear_operator(self, h):
        L = len(h)
        return sum(hi * self.projector(i, L) for i,hi in enumerate(h) if hi)
    
    @classmethod
    def quadratic_operator(self, J):
        L = len(J)
        return sum(J[i,j] * (self.projector(i,L) @ self.projector(j,L))
                   for i in range(L) for j in range(L) if J[i,j])
    
    def test_qubo_magnetic_field(self):
        np.random.seed(1022)
        for N in range(1, 10):
            h = np.random.rand(N) - 0.5
            self.assertTrue(similar(qubo_mpo(h=h).tomatrix(),
                                    self.linear_operator(h)))
    
    def test_qubo_quadratic(self):
        np.random.seed(1022)
        for N in range(1, 10):
            J = np.random.rand(N,N) - 0.5
            self.assertTrue(similar(qubo_mpo(J=J).tomatrix(),
                                    self.quadratic_operator(J)))
