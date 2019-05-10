
import numpy as np
import scipy.linalg
from numbers import Number
import mps.state
import scipy.sparse as sp
from mps.state import _truncate_vector

σz = np.diag([1.0,-1.0])
i2 = np.identity(2)
σx = np.array([[0, 1], [1, 0]])
σy = -1j * σz @ σx

def creation_op(d):
    # Returns d dimensional cration operator
    return np.diag(np.sqrt(np.arange(1,d)),-1).astype(complex)

def annihilation_op(d):
    # Returns d dimensional cration operator
    return np.diag(np.sqrt(np.arange(1,d)),1).astype(complex)



class NNHamiltonian(object):
    
    def __init__(self, size):
        #
        # Create a nearest-neighbor interaction Hamiltonian
        # of a given size, initially empty.
        #
        self.size = size
        
    def dimension(self, ndx):
        #
        # Return the dimension of the local Hilbert space
        #
        return 0
    
    def interaction_term(self, ndx, t=0.0):
        #
        # Return the interaction between sites (ndx,ndx+1)
        #
        return 0


class ConstantNNHamiltonian(NNHamiltonian):

    def __init__(self, size, dimension):
        #
        # Create a nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operators acting on each site (can be different for each site)
        #  - int_left, int_right: list of L and R operators (can be different for each site)
        #
        self.size = size
        self.int_left = [[]] * (size-1)
        self.int_right = [[]] * (size-1)
        if isinstance(dimension, Number):
            dimension = [dimension] * size
        self.dimension_ = dimension

    def set_local_term(self, ndx, operator):
        #
        # Set the local term acting on the given site
        #
        if ndx == 0:
            self.add_interaction_term(ndx, operator, np.eye(self.dimension(1)))
        elif ndx == self.size-2:
            self.add_interaction_term(ndx, np.eye(self.dimension(ndx)), operator)
        else:
            self.add_interaction_term(ndx, np.eye(self.dimension(ndx)), 0.5*operator)
            self.add_interaction_term(ndx, 0.5*operator, np.eye(self.dimension(ndx+1)))

    def add_interaction_term(self, ndx, L, R):
        #
        # Add an interaction term $L \otimes R$ acting on sites 'ndx' and 'ndx+1'
        #
        # Add to int_left, int_right
        #
        # Update the self.interactions[ndx] term
        self.int_left[ndx].append(L)
        self.int_right[ndx].append(R)

    def dimension(self, ndx):
        return self.dimension_[ndx]

    def interaction_term(self, ndx, t=0.0):
        #for (L, R) in zip(self.int_left[ndx], self.int_right[ndx]):
            
        return sum([np.kron(L, R) for (L, R) in zip(self.int_left[ndx], self.int_right[ndx])])
            

def make_ti_Hamiltonian(size, intL, intR, local_term=None):
    """Construct a translationally invariant, constant Hamiltonian with open
    boundaries and fixed interactions.
    
    Arguments:
    size        -- Number of sites in the model
    int_left    -- list of L (applied to site ndx) operators
    int_right   -- list of R (applied to site ndx + 1) operators
    local_term  -- operator acting on every site (optional)
    
    Returns:
    H           -- ConstantNNHamiltonian
    """
    if local_term is not None:
        dimension = len(local_term)
    else:
        dimension = len(intL[0])
    
    H = ConstantNNHamiltonian(size, dimension)
    H.local_term = local_term
    H.intL = intL
    H.intR = intR
    for ndx in range(size-1):
        for L,R in zip(H.intL, H.intR):
            H.add_interaction_term(ndx, L, R)
        if local_term is not None:
            H.set_local_term(ndx, local_term)
    return H


class Trotter_unitaries(object):
    """"Create Trotter unitarities from a nearest-neighbor interaction Hamiltonian.

    Attributes:
    H = NNHamiltonian
    δt = Time step
    """

    def __init__(self, H, δt):
        self.H = H
        self.δt = δt

    def twosite_unitary(self, start):
        """Creates twp-site exponentials from interaction H terms"""
        U = scipy.linalg.expm(-1j * self.δt * self.H.interaction_term(start))
        U = U.reshape(self.H.dimension(start), self.H.dimension(start+1),
                      self.H.dimension(start), self.H.dimension(start+1))
        return U

def apply_2siteTrotter(U, ψ, start):
    return np.einsum('ijk,klm,prjl -> iprm', ψ[start], ψ[start+1], U)


def TEBD_sweep(H, ψ, δt, dr, evenodd, tol=0):
    #
    # Apply a TEBD sweep by evolving with the pairwise Trotter Hamiltonian
    # starting from left/rightmost site and moving on the 'direction' (>0 right,
    # <0 left) by pairs of sites.
    #
    # - H: NNHamiltonian
    # - ψ: Initial state in CanonicalMPS form (modified destructively)
    # - δt: Time step
    # - evenodd: 0, 1 depending on Trotter step
    # - direction: where to move
    #
    Trotter = Trotter_unitaries(H, δt)        
    def update_two_site(start, nextsite, dr):
        # Apply combined local and interaction exponential and move
        ψ.center = ψ.center + dr
        if start == 0:
            dr = +1
        elif start == (ψ.size-2):
            dr = -1
        AA = apply_2siteTrotter(Trotter.twosite_unitary(start) , ψ, start)
        ψ.update_canonical_2site(AA, start, nextsite, dr, tolerance=tol)

    #
    # Loop over ψ, updating pairs of sites acting with the unitary operator
    # made of the interaction and 0.5 times the local terms
    #
    if dr < 0:
        if ψ.size%2 == evenodd:
            start = ψ.size -1
        else:
            start = ψ.size -2
        for j in range(start, 0, -2):
            update_two_site(j-1, j, dr)
    else:
        start = 0 + evenodd
        for j in range(start, ψ.size-1, +2):
            update_two_site(j, j+1, dr)        
            
    return ψ



class TEBD_evolution(object):
    def __init__(self, H, ψ, dt, timesteps, order=1, tol=0, center=0):
        self.H = H
        self.ψ = mps.state.CanonicalMPS(ψ, center=center)
        self.dt = dt
        self.timesteps = timesteps
        self.order = order
        self.tolerance = tol
        
    def first_order(self, newψ, dr):
        newψ = TEBD_sweep(self.H, newψ, self.dt, dr, 1, tol=self.tolerance)
        newψ = TEBD_sweep(self.H, newψ, self.dt, -dr, 0, tol=self.tolerance)
        return newψ
        
    def second_order(self, newψ, dr):
        newψ = TEBD_sweep(self.H, newψ, self.dt/2., dr, 0, tol=self.tolerance)
        newψ = TEBD_sweep(self.H, newψ, self.dt, -dr, 1, tol=self.tolerance)
        newψ = TEBD_sweep(self.H, newψ, self.dt/2., dr, 0, tol=self.tolerance)
        return newψ
            
    def evolve(self):
        newψ = self.ψ
        dr = 1
        for i in range(self.timesteps):
            if self.order == 1:
                newψ = self.first_order(newψ, dr)
            if self.order == 2:
                newψ = self.second_order(newψ, dr)
                dr = -dr
        return newψ
                
