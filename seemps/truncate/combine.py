from ..typing import *
import numpy as np
from ..state import MPS, CanonicalMPS, Weight
from ..expectation import scprod
from ..state import Truncation, Strategy, DEFAULT_TOLERANCE
from ..tools import log
from .simplify import AntilinearForm


def multi_norm_squared(α, ψ):
    """Compute the norm-squared of the vector sum(α[i]*ψ[i])"""
    c = 0.0
    for i, αi in enumerate(α):
        for j in range(i):
            c += 2 * (αi.conjugate() * α[j] * scprod(ψ[i], ψ[j])).real
        c += np.abs(αi) ** 2 * scprod(ψ[i], ψ[i]).real
    return c


def guess_combine_state(weights: list[Weight], states: list[MPS]) -> MPS:
    def combine_tensors(A: Tensor3, sumA: Tensor3) -> Tensor3:
        DL, d, DR = sumA.shape
        a, d, b = A.shape
        if DL < a or DR < b:
            # Extend with zeros to accommodate new contribution
            sumA = sumA.reshape(max(DL, a), d, max(DR, b))
        dt = type(A[0, 0, 0] + sumA[0, 0, 0])
        if sumA.dtype != dt:
            sumA = sumA.astype(dt)
        else:
            sumA[:a, :, :b] += A
        return sumA

    guess: MPS = weights[0] * states[0]
    for n, state in enumerate(states[1:]):
        for i, (A, sumA) in enumerate(zip(state, guess)):
            guess[i] = combine_tensors(A if i > 0 else A * weights[n], sumA)
    return guess


def combine(
    weights: list[Weight],
    states: list[MPS],
    guess: Optional[MPS] = None,
    maxsweeps: int = 4,
    direction: int = +1,
    tolerance: float = DEFAULT_TOLERANCE,
    max_bond_dimension: Optional[int] = None,
    normalize: bool = True,
) -> tuple[MPS, float]:
    """Simplify an MPS ψ transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Arguments:
    ----------
    weights   -- N values of α
    states    -- N MPS states
    guess     -- An MPS, defaults to states[0]
    direction -- +1/-1 for the direction of the first sweep
    maxsweeps -- maximum number of sweeps to run
    tolerance -- relative tolerance when splitting the tensors
    max_bond_dimension -- maximum bond dimension (defaults to None, which is ignored)

    Output
    ------
    φ         -- CanonicalMPS approximation to the linear combination state
    error     -- error made in the approximation
    """
    if guess is None:
        guess = guess_combine_state(weights, states)
    base_error = sum(
        np.sqrt(np.abs(α)) * np.sqrt(ψ.error()) for α, ψ in zip(weights, states)
    )
    strategy = Strategy(
        method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
        tolerance=tolerance,
        max_bond_dimension=max_bond_dimension,
        normalize=normalize,
    )
    start = 0 if direction > 0 else guess.size - 1
    φ = CanonicalMPS(guess, center=start, strategy=strategy, normalize=normalize)
    err = norm_ψsqr = multi_norm_squared(weights, states)
    if norm_ψsqr < tolerance:
        return MPS([np.zeros((1, P.shape[1], 1)) for P in φ]), 0
    log(
        f"COMBINE ψ with |ψ|={norm_ψsqr**0.5} for {maxsweeps} sweeps with tolerance {strategy.get_tolerance()}.\nWeights: {weights}"
    )

    size = φ.size
    forms = [AntilinearForm(φ, ψ, center=start) for ψ in states]
    for sweep in range(maxsweeps):
        if direction > 0:
            for n in range(0, size - 1):
                tensor = sum(
                    α * f.tensor2site(direction) for α, f in zip(weights, forms)
                )
                φ.update_2site_right(tensor, n, strategy)
                for f in forms:
                    f.update(direction)
        else:
            for n in reversed(range(0, size - 1)):
                tensor = sum(
                    α * f.tensor2site(direction) for α, f in zip(weights, forms)
                )
                φ.update_2site_left(tensor, n, strategy)
                for f in forms:
                    f.update(direction)
            last = 0
        #
        # We estimate the error
        #
        last = φ.center
        B = φ[last]
        norm_φsqr = np.vdot(B, B).real
        if normalize:
            φ[last] = B / norm_φsqr
            norm_φsqr = 1.0
        C = sum(α * f.tensor1site() for α, f in zip(weights, forms))
        scprod_φψ = np.vdot(B, C)
        old_err = err
        err = 2 * abs(1.0 - scprod_φψ.real / np.sqrt(norm_φsqr * norm_ψsqr))
        log(f"sweep={sweep}, rel.err.={err}, old err.={old_err}, |φ|={norm_φsqr**0.5}")
        if err < tolerance or err > old_err:
            log("Stopping, as tolerance reached")
            break
        direction = -direction
    φ._error = 0.0
    φ.update_error(base_error**2)
    φ.update_error(err)
    return φ, err
