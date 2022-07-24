import numpy as np
import math
import mps
import mps.state
from mps.state.truncation import DEFAULT_TOLERANCE
from mps.tools import log
from .simplify import AntilinearForm

def multi_norm_squared(α, ψ):
    """Compute the norm-squared of the vector sum(α[i]*ψ[i])"""
    c = 0.0
    for (i, αi) in enumerate(α):
        for j in range(i):
            c += 2 * (αi.conjugate() * α[j] * scprod(ψ[i], ψ[j])).real
        c += np.abs(αi) ** 2 * scprod(ψ[i], ψ[i]).real
    return c


def combine(
    weights,
    states,
    guess=None,
    maxsweeps=4,
    direction=+1,
    tolerance=DEFAULT_TOLERANCE,
    normalize=True,
    max_bond_dimension=None,
):
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
    start = 0 if direction > 0 else size - 1

    if guess is None:
        guess = states[0]
    base_error = sum(
        np.sqrt(np.abs(α)) * np.sqrt(ψ.error()) for α, ψ in zip(weights, states)
    )
    φ = mps.state.CanonicalMPS(guess, center=start, tolerance=tolerance)
    err = norm_ψsqr = multi_norm_squared(weights, states)
    if norm_ψsqr < tolerance:
        return mps.state.MPS([np.zeros((1, P.shape[1], 1)) for P in φ]), 0
    log(f"Approximating ψ with |ψ|={norm_ψsqr**0.5} for {maxsweeps} sweeps.")
    log(f"Weights: {weights}")

    L = len(states)
    size = φ.size
    forms = [AntilinearForm(φ, ψ, center=start) for ψ in states]
    for sweep in range(maxsweeps):
        if direction > 0:
            for n in range(0, size - 1):
                log(
                    f"Updating sites ({n},{n+1}) left-to-right, form.center={forms[0].center}, φ.center={φ.center}"
                )
                tensor = sum(
                    α * f.tensor2site(direction) for α, f in zip(weights, forms)
                )
                φ.update_2site(
                    tensor, n, direction, tolerance, normalize, max_bond_dimension
                )
                for f in forms:
                    f.update(direction)
        else:
            for n in reversed(range(0, size - 1)):
                log(
                    f"Updating sites ({n},{n+1}) right-to-left, form.center={forms[0].center}, φ.center={φ.center}"
                )
                tensor = sum(
                    α * f.tensor2site(direction) for α, f in zip(weights, forms)
                )
                φ.update_2site(
                    tensor, n, direction, tolerance, normalize, max_bond_dimension
                )
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
        log(f"rel.err.={err}, old err.={old_err}, |φ|={norm_φsqr**0.5}")
        if err < tolerance:
            log(f"Stopping, as tolerance reached")
            break
        direction = -direction
    φ._error = 0.0
    φ.update_error(base_error**2)
    φ.update_error(err)
    return φ, err


def cgs(A, b, guess=None, maxiter=100, tolerance=DEFAULT_TOLERANCE):
    """Given the MPO `A` and the MPS `b`, estimate another MPS that
    solves the linear system of equations A * ψ = b, using the
    conjugate gradient system.

    Parameters
    ----------
    A         -- Linear MPO
    b         -- Right-hand side of the equation
    maxiter   -- Maximum number of iterations
    tolerance -- Truncation tolerance and also error tolerance
    max_bond_dimension -- None (ignore) or maximum bond dimension

    Output
    ------
    ψ         -- Approximate solution to A ψ = b
    error     -- norm square of the residual, ||r||^2
    """
    ρprev = 0
    normb = scprod(b, b).real
    x = guess
    r = b
    if x is not None:
        r, err = combine(
            [1.0, -1.0], [b, A.apply(x)], tolerance=tolerance, normalize=False
        )
    p = r
    ρ = scprod(r, r).real
    for i in range(maxiter):
        Ap = A.apply(p)
        α = ρ / scprod(p, Ap).real
        if x is not None:
            x, err = combine([1, α], [x, p], tolerance=tolerance, normalize=False)
        else:
            x, err = combine([α], [p], tolerance=tolerance, normalize=False)
        r, err = combine([1, -1], [b, A.apply(x)], tolerance=tolerance, normalize=False)
        ρ, ρold = scprod(r, r).real, ρ
        ρ2 = multi_norm_squared([1.0, -1.0], [b, A.apply(x)])
        if ρ < tolerance * normb:
            log(f"Breaking on convergence")
            break
        p, err = combine([1.0, ρ / ρold], [r, p], tolerance=tolerance, normalize=False)
        log(f"Iteration {i:5}: |r|={ρ:5g}")
    return x, abs(ρ)
