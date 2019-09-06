
import numpy as np
import scipy.linalg
from numbers import Number
import mps.state
import scipy.sparse as sp
from mps.state import _truncate_vector, DEFAULT_TOLERANCE


def pairwise_unitaries(H, δt):
    return [scipy.linalg.expm((-1j * δt) * H.interaction_term(k)).
                              reshape(H.dimension(k), H.dimension(k+1),
                                      H.dimension(k), H.dimension(k+1))
            for k in range(H.size-1)]


def apply_pairwise_unitaries(U, ψ, start, direction, tol=DEFAULT_TOLERANCE):
    """Apply the list of pairwise unitaries U onto an MPS state ψ in
    canonical form. Unitaries are applied onto pairs of sites (i,i+1),
    (i+2,i+3), etc. We start at 'i=start' and move in increasing or
    decreasing order of sites depending on 'direction'
    
    Arguments:
    U         -- List of pairwise unitaries
    ψ         -- State in canonical form
    start     -- First site for applying pairwise unitaries
    direction -- Direction of sweep.
    
    Returns:
    ψ         -- MPS in canonical form"""
    
    #print("Apply pairwise unitarities in direction {} and at starting site {} with center {}".format(direction, start, ψ.center))
    ψ.recenter(start)
    if direction > 0:
        newstart = ψ.size-2
        for j in range(start, ψ.size-1, +2):
            #print('Updating sites ({}, {}), center={}, direction={}'.format(j, j+1, ψ.center, direction))
            AA = np.einsum('ijk,klm,nrjl -> inrm', ψ[j], ψ[j+1], U[j])
            ψ.update_2site(AA, j, +1, tolerance=tol)
            if j < newstart:
                ψ.update_canonical(ψ[j+1], +1, tolerance=tol)
            #print("New center= {}, new direction = {}".format(ψ.center, direction))
        return newstart, -1
    else:
        newstart = 0
        for j in range(start, -1, -2):
            #print('Updating sites ({}, {}), center={}, direction={}'.format(j, j+1, ψ.center, direction))
            AA = np.einsum('ijk,klm,nrjl -> inrm', ψ[j], ψ[j+1], U[j])
            ψ.update_2site(AA, j, -1, tolerance=tol)
            if j > 0:
                ψ.update_canonical(ψ[j], -1, tolerance=tol)
            #print("New center= {}, new direction = {}".format(ψ.center, direction))
        return newstart, +1  

class TEBD_evolution(object):
    """TEBD_evolution is a class that continuously updates a quantum state ψ
    evolving it with a Hamiltonian H over intervals of time dt."""
    
    def __init__(self, ψ, H, dt, timesteps=1, order=1, tol=DEFAULT_TOLERANCE):
        """Create a TEBD algorithm to evolve a quantum state ψ with a fixed
        Hamiltonian H.
        
        Arguments:
        ψ         -- Quantum state to be updated. The class keeps a copy.
        H         -- NNHamiltonian for the evolution
        dt        -- Size of each Trotter step
        timesteps -- How many Trotter steps in each call to evolve()
        order     -- Order of the Trotter approximation (1 or 2)
        tol       -- Tolerance in MPS truncation
        """
        self.H = H
        self.dt = float(dt)
        self.timesteps = timesteps
        self.order = order
        self.tolerance = tol
        self.Udt = pairwise_unitaries(H, dt)
        if order == 2:
            self.Udt2 = pairwise_unitaries(H, dt/2)
        if not isinstance(ψ, mps.state.CanonicalMPS):
            ψ = mps.state.CanonicalMPS(ψ, center=0)
        else:
            ψ = ψ.copy()
        self.ψ = ψ
        if ψ.center <= 1:
            self.start = 0
            self.direction = +1
        else:
            self.start = ψ.size-2
            self.direction = -1

    def evolve(self, timesteps=None):
        """Update the quantum state with `timesteps` repetitions of the
        Trotter algorithms."""
        
        #print("Apply TEBD for {} timesteps in the order {}".format(self.timesteps, self.order))
        
        if timesteps is None:
            timesteps = self.timesteps
        for i in range(self.timesteps):
            #print(i)
            if self.order == 1:
                #print("Sweep in direction {} and at starting site {}".format(self.direction, self.start))
                self.start, self.direction = apply_pairwise_unitaries(self.Udt, self.ψ, self.start, self.direction, tol=self.tolerance)
                #print("Sweep in direction {} and at starting site {}".format(self.direction, self.start))
                self.start, self.direction = apply_pairwise_unitaries(self.Udt, self.ψ, self.start, self.direction, tol=self.tolerance)
            else:
                self.start, self.direction = apply_pairwise_unitaries(self.Udt2, self.ψ, self.start, self.direction, tol=self.tolerance)
                self.start, self.direction = apply_pairwise_unitaries(self.Udt, self.ψ, self.start, self.direction, tol=self.tolerance)
                self.start, self.direction = apply_pairwise_unitaries(self.Udt2, self.ψ, self.start, self.direction, tol=self.tolerance)
        
            #print("New direction = {} and new starting site = {}".format(self.direction, self.start))
        
        return self.ψ

    def state():
        return self.ψ
