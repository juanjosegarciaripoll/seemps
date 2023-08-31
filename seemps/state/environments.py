import numpy as np
from ..typing import *


def begin_environment(χ=1) -> Environment:
    """Initiate the computation of a left environment from two MPSLike. The bond
    dimension χ defaults to 1. Other values are used for states in canonical
    form that we know how to open and close."""
    return np.eye(χ, dtype=np.float64)


def update_left_environment(
    B: Tensor3, A: Tensor3, rho: Environment, operator: Optional[Operator] = None
) -> Environment:
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        A = np.einsum("ji,aib->ajb", operator, A)
    if False:
        rho = np.einsum("li,ijk->ljk", rho, A)
        return np.einsum("lmn,lmk->nk", B.conj(), rho)
    else:
        i, j, k = A.shape
        l, j, n = B.shape
        rho = np.dot(rho, A.reshape(i, j * k))
        rho = np.dot(B.conj().reshape(l * j, n).T, rho.reshape(l * j, k))
        return rho


def update_right_environment(
    B: Tensor3, A: Tensor3, rho: Environment, operator: Optional[Operator] = None
) -> Environment:
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        A = np.einsum("ji,aib->ajb", operator, A)
    if False:
        rho = np.einsum("ijk,kn->ijn", A, rho)
        return np.einsum("ijn,ljn->il", rho, B.conj())
    else:
        # np.einsum("ijk,kn,ljn->il", A, rho, B.conj())
        i, j, k = A.shape
        l, j, n = B.shape
        rho = np.dot(A.reshape(i * j, k), rho)
        return np.dot(rho.reshape(i, j * n), B.reshape(l, j * n).T.conj())


def end_environment(ρ: Environment) -> Weight:
    """Extract the scalar product from the last environment."""
    return ρ[0, 0]


def join_environments(ρL: Environment, ρR: Environment) -> Weight:
    """Join left and right environments to produce a scalar."""
    # np.einsum("ij,ji", ρL, ρR)
    return np.trace(np.dot(ρL, ρR))


def scprod(ϕ: MPSLike, ψ: MPSLike) -> Weight:
    """Compute the scalar product between matrix product states <ϕ|ψ>."""
    ρ: Environment = begin_environment()
    for ϕi, ψi in zip(ϕ, ψ):
        ρ = update_left_environment(ϕi, ψi, ρ)
    return end_environment(ρ)
