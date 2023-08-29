from abc import abstractmethod
from typing import Union
import numpy as np
import math
import scipy.linalg  # type: ignore

from seemps.hamiltonians import NNHamiltonian  # type: ignore
from .state import Strategy, DEFAULT_STRATEGY, MPS, CanonicalMPS

Unitary = np.ndarray


def _investigate_unitary_contraction():
    import timeit

    A = np.random.randn(10, 2, 13)
    A /= np.linalg.norm(A)
    B = np.random.randn(13, 2, 10)
    B /= np.linalg.norm(B)
    U = np.random.randn(2, 2, 2, 2)

    def method1():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U)

    path = np.einsum_path("ijk,klm,nrjl -> inrm", A, B, U, optimize="optimal")[0]

    def method2():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U, optimize=path)

    def method3():
        a, d, b = A.shape
        b, e, c = B.shape
        D = d * e
        aux = np.tensordot(A, B, (2, 0)).reshape(a, D, c)
        aux = np.tensordot(U.reshape(D, D), aux, (1, 1)).transpose(1, 0, 2)
        return aux.reshape(a, d, e, c)

    def method4():
        a, d, b = A.shape
        b, e, c = B.shape
        D = d * e
        aux = np.tensordot(A, B, (2, 0)).reshape(a, D, c)
        aux = np.tensordot(aux, U.reshape(D, D), (1, 1)).transpose(0, 2, 1)
        return aux.reshape(a, d, e, c)

    repeats = 10000
    t = timeit.timeit(method1, number=repeats)
    t = timeit.timeit(method1, number=repeats)
    print(f"Method1 {t/repeats}s")

    t = timeit.timeit(method2, number=repeats)
    t = timeit.timeit(method2, number=repeats)
    print(f"Method2 {t/repeats}s")

    t = timeit.timeit(method3, number=repeats)
    t = timeit.timeit(method3, number=repeats)
    print(f"Method3 {t/repeats}s")

    t = timeit.timeit(method4, number=repeats)
    t = timeit.timeit(method4, number=repeats)
    print(f"Method4 {t/repeats}s")

    for i, m in enumerate([method1, method2, method3, method4]):
        err = np.linalg.norm(method1() - m())
        print(f"Method{i} error = {err}")


def _contract_U_A_B(U: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Implements np.einsum('ijk,klm,nrjl -> inrm', A, B, U)"""
    a, d, _ = A.shape
    _, e, c = B.shape
    D = d * e
    aux = np.tensordot(A, B, (2, 0)).reshape(a, D, c)
    aux = np.tensordot(U.reshape(D, D), aux, (1, 1)).transpose(1, 0, 2)
    return aux.reshape(a, d, e, c)


class PairwiseUnitaries:
    U: list[Unitary]

    def __init__(self, H: NNHamiltonian, dt: float, strategy: Strategy):
        self.U = [
            scipy.linalg.expm((-1j * dt) * H.interaction_term(k)).reshape(
                H.dimension(k),
                H.dimension(k + 1),
                H.dimension(k),
                H.dimension(k + 1),
            )
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
                AA = _contract_U_A_B(U[j], ψ[j], ψ[j + 1])
                ψ.update_2site(AA, j, +1, strategy)
        else:
            if center < L - 2:
                ψ.recenter(L - 2)
            for j in range(L - 2, -1, -1):
                ## AA = np.einsum("ijk,klm,nrjl -> inrm", ψ[j], ψ[j + 1], U[j])
                AA = _contract_U_A_B(U[j], ψ[j], ψ[j + 1])
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
