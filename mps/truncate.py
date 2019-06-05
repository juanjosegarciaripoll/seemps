
import numpy as np
import mps
from mps.state import *

from mps.expectation import \
    begin_environment, \
    update_right_environment, \
    update_left_environment

class AntilinearForm:
    #
    # This class is an object that formally implements <ψ|ϕ> with an argument
    # ϕ that may change and be updated over time.
    #
    # Given a site 'n' it returns the tensor 'L' such that the contraction
    # between 'L' and ϕ[n] is the result of the linear form."""
    #
    def __init__(self, ψ, ϕ, center=0):
        #
        # At the beginning, we create the right- and left- environments for
        # all sites to the left and to the right of 'center', which is the
        # focus point of 'LinearForm'.
        #
        ρ = begin_environment()
        R = [ρ] * ψ.size
        for i in range(ψ.size-1, center, -1):
            R[i-1] = ρ = update_right_environment(ϕ[i], ψ[i], ρ)

        ρ = begin_environment()
        L = [ρ] * ψ.size
        for i in range(0, center):
            L[i+1] = ρ = update_left_environment(ϕ[i], ψ[i], ρ)

        self.ψ = ψ
        self.ϕ = ϕ
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
        C = self.ψ[center]
        
        return np.einsum('li,ijk,kn->ljn', L, C, R)

    def tensor2site(self, direction):
        #
        # Return the tensor that represents the LinearForm using 'center'
        # and 'center+/-1'
        #
        center = self.center
        if direction > 0:
            i = self.center
            j = i + 1
        else:
            j = self.center
            i = j - 1
        L = self.L[i]
        A = self.ψ[i]
        B = self.ψ[j]
        R = self.R[j]
        LA = np.einsum('li,ijk->ljk', L, A)
        BR = np.einsum('kmn,no->kmo', B, R)
        return np.einsum('ljk,kmo->ljmo', LA, BR)

    def update(self, direction):
        #
        # We have updated 'ϕ', which is now centered on a different point.
        # We have to recompute the environments.
        #
        center = self.center
        if direction > 0:
            L = self.L[center]
            if center+1 < ψ.size:
                L = update_left_environment(ψ[center], ϕ[center], L)
                self.L[center+1] = L
                self.center = center+1
        else:
            R = self.R[center]
            if center > 0:
                R = update_right_environment(ψ[center], ϕ[center], L)
                self.R[center-1] = R
                self.center = center-1

def simplify_mps_2site(ψ, dimension=0, sweeps=2, direction=+1,
                       tolerance=DEFAULT_TOLERANCE):
    """Simplify an MPS ψ transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.
    
    Arguments:
    ----------
    sweeps = maximum number of sweeps to run
    tolerance = relative tolerance when splitting the tensors
    dimension = maximum bond dimension, 0 to just truncate to tolerance
    """
    
    if dimension == 0:
        return CanonicalMPS(ψ, center=0, tolerance=tolerance)
