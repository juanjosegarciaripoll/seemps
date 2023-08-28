from abc import abstractmethod
from typing import Union
import numpy as np
import scipy.linalg  # type: ignore

from seemps.hamiltonians import NNHamiltonian  # type: ignore
from .state import Strategy, DEFAULT_STRATEGY, MPS, CanonicalMPS

Unitary = np.ndarray


class PairwiseUnitaries:
    U: list[Unitary]

    def __init__(self, H: NNHamiltonian, dt: float, strategy: Strategy):
        self.U = [
            scipy.linalg.expm((-1j * dt) * H.interaction_term(k)).reshape(
                H.dimension(k), H.dimension(k + 1), H.dimension(k), H.dimension(k + 1)
            )
            for k in range(H.size - 1)
        ]
        self.strategy = strategy

    def apply(self, ψ: Union[MPS, CanonicalMPS]) -> CanonicalMPS:
        strategy = self.strategy
        if not isinstance(ψ, CanonicalMPS):
            ψ = CanonicalMPS(ψ, center=0, strategy=strategy)
        L = ψ.size
        U = self.U
        if ψ.center == 0:
            for j in range(L - 1):
                AA = np.einsum("ijk,klm,nrjl -> inrm", ψ[j], ψ[j + 1], U[j])
                ψ.update_2site(AA, j, +1, strategy)
        else:
            if ψ.center < L - 2:
                ψ.recenter(L - 2)
            for j in range(L - 2, -1, -1):
                # print('Updating sites ({}, {}), center={}, direction={}'.format(j, j+1, ψ.center, direction))
                AA = np.einsum("ijk,klm,nrjl -> inrm", ψ[j], ψ[j + 1], U[j])
                ψ.update_2site(AA, j, -1, strategy)
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
        return self.U.apply(state)


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
        state = self.Umid.apply(state)
        return self.U.apply(state)
