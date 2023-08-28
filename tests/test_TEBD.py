import unittest
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from seemps.state import CanonicalMPS, DEFAULT_STRATEGY
from seemps.tools import *
from .tools import *
from seemps.evolution import *
from seemps.hamiltonians import make_ti_Hamiltonian


def random_wavefunction(n):
    ψ = np.random.rand(n) - 0.5
    return ψ / np.linalg.norm(ψ)


class TestTEBD_sweep(unittest.TestCase):
    @staticmethod
    def hopping_model(N, t, ω):
        a = annihilation(2)
        ad = creation(2)
        return make_ti_Hamiltonian(N, [t * a, t * ad], [ad, a], local_term=ω * (ad @ a))

    @staticmethod
    def hopping_model_Trotter_matrix(N, t, ω):
        #
        # Hamiltonian that generates the evolution of the odd hoppings
        # and local frequencies
        return sp.diags(
            [[t, 0] * (N // 2), [ω] + [ω / 2] * (N - 2) + [ω], [t, 0] * (N // 2)],
            offsets=[-1, 0, +1],
            shape=(N, N),
            dtype=np.float64,
        )

    @staticmethod
    def hopping_model_matrix(N, t, ω):
        return sp.diags([[t] * (N), ω, [t] * (N)], offsets=[-1, 0, +1], shape=(N, N))

    def inactive_test_apply_pairwise_unitaries(self):
        N = 2
        tt = -np.pi / 2
        ω = np.pi
        dt = 0.1
        #
        # Numerically exact solution using Scipy's exponentiation routine
        ψwave = random_wavefunction(N)
        print(seemps.state.wavepacket(ψwave).to_vector())
        HMat = self.hopping_model_Trotter_matrix(N, tt, ω)
        ψwave_final = expm_multiply(+1j * dt * HMat, ψwave)
        print(seemps.state.wavepacket(ψwave_final).to_vector())
        print(HMat.todense())
        #
        # Evolution using Trrotter
        H = self.hopping_model(N, tt, ω)
        U = pairwise_unitaries(H, dt)
        ψ = CanonicalMPS(seemps.state.wavepacket(ψwave))
        start = 0
        direction = 1
        apply_pairwise_unitaries(U, ψ, start, direction, truncation=DEFAULT_STRATEGY)
        print(ψ.to_vector())
        print(np.abs(seemps.state.wavepacket(ψwave_final).to_vector() - ψ.to_vector()))

        self.assertTrue(
            similar(
                abs(seemps.state.wavepacket(ψwave_final).to_vector()),
                abs(ψ.to_vector()),
            )
        )

    def test_TEBD_evolution_first_order(self):
        #
        #
        #
        N = 19
        t = -np.pi / 2
        ω = np.pi
        dt = 1e-6
        Nt = int(1000)
        # ψwave = random_wavefunction(N)
        xx = np.arange(N)
        x0 = int(N // 2)
        w0 = 5
        k0 = np.pi / 2
        #
        # Approximate evolution of a wavepacket in a tight-binding model
        ψwave = np.exp(-((xx - x0) ** 2) / w0**2 + 1j * k0 * xx)
        ψwave = ψwave / np.linalg.norm(ψwave)
        Hmat = self.hopping_model_matrix(N, t, ω)
        ψwave_final = expm_multiply(-1j * dt * Nt * Hmat, ψwave)
        #
        # Trotter solution
        ψmps = CanonicalMPS(seemps.state.wavepacket(ψwave))
        H = self.hopping_model(N, t, ω)
        ψmps = TEBD_evolution(
            ψmps, H, dt, timesteps=Nt, order=1, truncation=DEFAULT_STRATEGY
        ).evolve()

        self.assertTrue(
            similar(
                abs(seemps.state.wavepacket(ψwave_final).to_vector()),
                abs(ψmps.to_vector()),
            )
        )

    def test_TEBD_evolution_second_order(self):
        #
        #
        #
        N = 21
        t = 0.1
        ω = 0.5
        dt = 1e-6
        Nt = int(1000)
        # ψwave = random_wavefunction(N)
        xx = np.arange(N)
        x0 = int(N // 2)
        w0 = 5
        k0 = np.pi / 2
        #
        # Approximate evolution of a wavepacket in a tight-binding model
        ψwave = np.exp(-((xx - x0) ** 2) / w0**2 + 1j * k0 * xx)
        ψwave = ψwave / np.linalg.norm(ψwave)
        Hmat = self.hopping_model_matrix(N, t, ω)
        ψwave_final = expm_multiply(-1j * dt * Nt * Hmat, ψwave)
        #
        # Trotter evolution
        H = self.hopping_model(N, t, ω)
        ψmps = CanonicalMPS(seemps.state.wavepacket(ψwave))
        ψmps = TEBD_evolution(
            ψmps, H, dt, timesteps=Nt, order=2, truncation=DEFAULT_STRATEGY
        ).evolve()
        self.assertTrue(
            similar(
                abs(seemps.state.wavepacket(ψwave_final).to_vector()),
                abs(ψmps.to_vector()),
            )
        )
