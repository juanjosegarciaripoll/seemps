from mps.expectation import scprod
from mps.state.truncation import DEFAULT_TOLERANCE
from mps.truncate.combine import combine, multi_norm_squared
from mps.tools import log


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
