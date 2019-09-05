from mps.qft import *

import unittest
import numpy as np
import numpy.fft
from mps.test.tools import *
from mps.state import MPS
import scipy.sparse as sp
import matplotlib.pyplot as plt

class TestQFT(unittest.TestCase):
    
    @staticmethod
    def gaussian_mps(N):
        x = np.linspace(-4,4,2**N+1)[:-1]
        ψ = np.exp(-(x**2)/2.);
        ψ /= np.linalg.norm(ψ)
        return MPS.fromvector(ψ, [2]*N), ψ
    
    def test_qft_is_fft(self):
        np.random.seed(1022)
        for N in range(4, 10):
            ψmps, ψ = self.gaussian_mps(N)
            self.assertTrue(similar(qft_flip(qft(ψmps)).tovector(),
                                    numpy.fft.fft(ψ, norm='ortho')))
    
    def test_iqft_is_fft(self):
        np.random.seed(1022)
        for N in range(4, 10):
            ψmps, ψ = self.gaussian_mps(N)
            self.assertTrue(similar(qft_flip(iqft(ψmps)).tovector(),
                                    numpy.fft.ifft(ψ, norm='ortho')))
    
    def test_qft_nd_is_qft(self):
        np.random.seed(1022)
        for N in range(4, 10):
            ψmps, _ = self.gaussian_mps(N)
            self.assertTrue(similar(qft(ψmps).tovector(),
                                    qft_nd_mpo(np.arange(N), N).apply(ψmps)))
