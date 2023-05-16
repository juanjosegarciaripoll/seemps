import numpy as np
from .. import state
from ..state.truncation import DEFAULT_TOLERANCE
from ..tools import log, mydot
from ..expectation import (
    begin_environment,
    update_right_environment,
    update_left_environment,
    scprod,
)


class AntilinearForm:
    #
    # This class is an object that formally implements <ϕ|ψ> with an argument
    # ϕ that may change and be updated over time.
    #
    # Given a site 'n' it returns the tensor 'L' such that the contraction
    # between 'L' and ϕ[n] is the result of the linear form."""
    #
    def __init__(self, bra, ket, center=0):
        assert bra.size == ket.size
        #
        # At the beginning, we create the right- and left- environments for
        # all sites to the left and to the right of 'center', which is the
        # focus point of 'LinearForm'.
        #
        size = bra.size
        ρ = begin_environment()
        R = [ρ] * size
        for i in range(size - 1, center, -1):
            R[i - 1] = ρ = update_right_environment(bra[i], ket[i], ρ)

        ρ = begin_environment()
        L = [ρ] * size
        for i in range(0, center):
            L[i + 1] = ρ = update_left_environment(bra[i], ket[i], ρ)

        self.bra = bra
        self.ket = ket
        self.size = size
        self.R = R
        self.L = L
        self.center = center

    def tensor1site(self):
        #
        # Return the tensor that represents the LinearForm at the 'center'
        # site of the MPS
        #
        center = self.center
        L = self.L[center]
        R = self.R[center]
        C = self.ket[center]
        return np.einsum("li,ijk,kn->ljn", L, C, R)

    def tensor2site(self, direction):
        #
        # Return the tensor that represents the LinearForm using 'center'
        # and 'center+/-1'
        #
        if direction > 0:
            i = self.center
            j = i + 1
        else:
            j = self.center
            i = j - 1
        L = self.L[i]
        A = self.ket[i]
        B = self.ket[j]
        R = self.R[j]
        LA = mydot(L, A)  # np.einsum("li,ijk->ljk", L, A)
        BR = mydot(B, R)  # np.einsum("kmn,no->kmo", B, R)
        return mydot(LA, BR)  # np.einsum("ljk,kmo->ljmo", LA, BR)

    def update(self, direction):
        #
        # We have updated 'ϕ' (the bra), which is now centered on a different point.
        # We have to recompute the environments.
        #
        prev = self.center
        if direction > 0:
            nxt = prev + 1
            if nxt < self.size:
                self.L[nxt] = update_left_environment(
                    self.bra[prev], self.ket[prev], self.L[prev]
                )
                self.center = nxt
        else:
            nxt = prev - 1
            if nxt >= 0:
                self.R[nxt] = update_right_environment(
                    self.bra[prev], self.ket[prev], self.R[prev]
                )
                self.center = nxt


def simplify(
    ψ,
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
    ψ         -- state to approximate
    direction -- +1/-1 for the direction of the first sweep
    maxsweeps -- maximum number of sweeps to run
    tolerance -- relative tolerance when splitting the tensors
    max_bond_dimension -- maximum bond dimension (defaults to None, which is ignored)

    Output
    ------
    φ         -- CanonicalMPS approximation to the state ψ
    error     -- error made in the approximation, as $‖φ/‖φ‖ - ψ/‖ψ‖‖^2$
    direction -- direction that the next sweep would be
    """
    size = ψ.size
    start = 0 if direction > 0 else size - 1

    base_error = ψ.error()
    φ = state.CanonicalMPS(ψ, center=start, tolerance=tolerance, normalize=normalize)
    φ.maxsweeps = maxsweeps
    φ.max_bond_dimension = max_bond_dimension
    if max_bond_dimension == 0 and tolerance <= 0:
        return φ

    form = AntilinearForm(φ, ψ, center=start)
    norm_ψsqr = scprod(ψ, ψ).real
    log(
        f"SIMPLIFY ψ with |ψ|={norm_ψsqr**0.5} for {maxsweeps} sweeps, with tolerance {tolerance}."
    )
    for sweep in range(maxsweeps):
        if direction > 0:
            for n in range(0, size - 1):
                φ.update_2site(
                    form.tensor2site(direction),
                    n,
                    direction,
                    tolerance,
                    normalize,
                    max_bond_dimension,
                )
                form.update(direction)
            last = size - 1
        else:
            for n in reversed(range(0, size - 1)):
                φ.update_2site(
                    form.tensor2site(direction),
                    n,
                    direction,
                    tolerance,
                    normalize,
                    max_bond_dimension,
                )
                form.update(direction)
            last = 0
        #
        # We estimate the error
        #
        last = φ.center
        B = φ[last]
        norm_φsqr = np.vdot(B, B).real
        if normalize:
            φ[last] = B = B / norm_φsqr
            norm_φsqr = 1.0
        scprod_φψ = np.vdot(B, form.tensor1site())
        err = 2 * abs(1.0 - scprod_φψ.real / np.sqrt(norm_φsqr * norm_ψsqr))
        log(f"sweep={sweep}, rel.err.={err}, |φ|={norm_φsqr**0.5}")
        if err < tolerance:
            log("Stopping, as tolerance reached")
            break
        direction = -direction
    φ._error = 0.0
    φ.update_error(base_error)
    φ.update_error(err)
    return φ, err, direction
