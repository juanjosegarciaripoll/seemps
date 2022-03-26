import unittest
import numpy as np
from tools import *
from mps.state import MPS, CanonicalMPS
from mps.register import * 
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

    def test_product(self):
        np.random.seed(1034)
        for N in range(1, 10):
            ψ = np.random.rand(2**N,2)-0.5
            ψ = ψ[:,0] + 1j*ψ[:,1]
            ψ /= np.linalg.norm(ψ)
            ψmps = MPS.fromvector(ψ,[2]*N)
            ψ = ψmps.tovector()
            
            ξ = np.random.rand(2**N,2)-0.5
            ξ = ξ[:,0] + 1j*ξ[:,1]
            ξ /= np.linalg.norm(ξ)
            ξmps = MPS.fromvector(ξ,[2]*N)
            ξ = ξmps.tovector()
            
            ψξ = wavefunction_product(ψmps, ξmps, simplify=True, normalize=False).tovector()
            self.assertTrue(similar(ψξ, ψ*ξ))
            
            ψcξ = wavefunction_product(ψmps, ξmps, conjugate=True, simplify=False, normalize=False).tovector()
            self.assertTrue(similar(ψcξ, ψ.conj() * ξ))
