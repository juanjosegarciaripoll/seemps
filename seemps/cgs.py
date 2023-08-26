from .expectation import scprod
from .state import DEFAULT_TOLERANCE
from .truncate.combine import combine
from .tools import log


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
    normb = scprod(b, b).real
    x = guess
    r = b
    if x is not None:
        r, _ = combine(
            [1.0, -1.0], [b, A.apply(x)], tolerance=tolerance, normalize=False
        )
    p = r
    ρ = scprod(r, r).real
    log(f"CGS algorithm for {maxiter} iterations")
    for i in range(maxiter):
        Ap = A.apply(p)
        α = ρ / scprod(p, Ap).real
        if x is not None:
            x, _ = combine([1, α], [x, p], tolerance=tolerance, normalize=False)
        else:
            x, _ = combine([α], [p], tolerance=tolerance, normalize=False)
        r, _ = combine([1, -1], [b, A.apply(x)], tolerance=tolerance, normalize=False)
        ρ, ρold = scprod(r, r).real, ρ
        if ρ < tolerance * normb:
            log("Breaking on convergence")
            break
        p, _ = combine([1.0, ρ / ρold], [r, p], tolerance=tolerance, normalize=False)
        log(f"Iteration {i:5}: |r|={ρ:5g}")
    return x, abs(ρ)
