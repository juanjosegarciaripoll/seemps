from abc import abstractmethod
from typing import Union
import numpy as np
import math
import scipy.linalg  # type: ignore

from seemps.hamiltonians import NNHamiltonian  # type: ignore
from .state import Strategy, DEFAULT_STRATEGY, MPS, CanonicalMPS

Unitary = np.ndarray


def _contract_U_A_B(U: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    #
    # Assuming U[n*r,j*l], A[i,j,k] and B[k,l,m]
    # Implements np.einsum('ijk,klm,nrjl -> inrm', A, B, U)
    # See tests.test_contractions for other implementations and timing
    #
    a, d, b = A.shape
    b, e, c = B.shape
    return np.matmul(
        U, np.matmul(A.reshape(-1, b), B.reshape(b, -1)).reshape(a, -1, c)
    ).reshape(a, d, e, c)


class PairwiseUnitaries:
    U: list[Unitary]

    def __init__(self, H: NNHamiltonian, dt: float, strategy: Strategy):
        self.U = [
            scipy.linalg.expm((-1j * dt) * H.interaction_term(k))
            for k in range(H.size - 1)
        ]
        self.strategy = strategy

    def apply(self, ψ: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        return self.apply_inplace(ψ.copy() if isinstance(ψ, CanonicalMPS) else ψ)

    def apply_inplace(self, ψ: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        strategy = self.strategy
        if not isinstance(ψ, CanonicalMPS):
            ψ = CanonicalMPS(ψ, center=0, strategy=strategy)
        L = ψ.size
        U = self.U
        center = ψ.center
        if center < L // 2:
            if center > 1:
                ψ.recenter(1)
            for j in range(L - 1):
                ## AA = np.einsum("ijk,klm,nrjl -> inrm", ψ[j], ψ[j + 1], U[j])
                ψ.update_2site_right(_contract_U_A_B(U[j], ψ[j], ψ[j + 1]), j, strategy)
        else:
            if center < L - 2:
                ψ.recenter(L - 2)
            for j in range(L - 2, -1, -1):
                ## AA = np.einsum("ijk,klm,nrjl -> inrm", ψ[j], ψ[j + 1], U[j])
                ψ.update_2site_left(_contract_U_A_B(U[j], ψ[j], ψ[j + 1]), j, strategy)
        return ψ


class Trotter:
    @abstractmethod
    def apply(self, state: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        pass

    def __matmul__(self, state: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        return self.apply(state)


class Trotter2ndOrder(Trotter):
    U: PairwiseUnitaries
    strategy: Strategy

    def __init__(
        self, H: NNHamiltonian, dt: float, strategy: Strategy = DEFAULT_STRATEGY
    ):
        self.U = PairwiseUnitaries(H, 0.5 * dt, strategy)

    def apply(self, state: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        state = self.U.apply(state)
        return self.U.apply_inplace(state)

    def apply_inplace(self, state: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        state = self.U.apply_inplace(state)
        return self.U.apply_inplace(state)


class Trotter3rdOrder(Trotter):
    U: PairwiseUnitaries

    def __init__(
        self,
        H: NNHamiltonian,
        dt: float,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.Umid = PairwiseUnitaries(H, 0.5 * dt, strategy)
        self.U = PairwiseUnitaries(H, 0.25 * dt, strategy)

    def apply(self, state: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        state = self.U.apply(state)
        state = self.Umid.apply_inplace(state)
        return self.U.apply_inplace(state)

    def apply_inplace(self, state: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        state = self.U.apply_inplace(state)
        state = self.Umid.apply_inplace(state)
        return self.U.apply_inplace(state)
