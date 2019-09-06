
import numpy as np
from numbers import Number
import scipy.sparse as sp

class NNHamiltonian(object):
    
    def __init__(self, size):
        #
        # Create a nearest-neighbor interaction Hamiltonian
        # of a given size, initially empty.
        #
        self.size = size
        self.constant = False

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
    
    def tomatrix(self, t=0.0):
        """Return a sparse matrix representing the NNHamiltonian on the
        full Hilbert space."""
        
        # dleft is the dimension of the Hilbert space of sites 0 to (i-1)
        # both included
        dleft = 1
        # H is the Hamiltonian of sites 0 to i, this site included.
        H = 0 * sp.eye(self.dimension(0))
        for i in range(self.size-1):
            # We extend the existing Hamiltonian to cover site 'i+1'
            H = sp.kron(H, sp.eye(self.dimension(i+1)))
            # We add now the interaction on the sites (i,i+1)
            H += sp.kron(sp.eye(dleft if dleft else 1), self.interaction_term(i,t))
            # We extend the dimension covered
            dleft *= self.dimension(i)

        return H


class ConstantNNHamiltonian(NNHamiltonian):

    def __init__(self, size, dimension):
        #
        # Create a nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operators acting on each site (can be different for each site)
        #  - int_left, int_right: list of L and R operators (can be different for each site)
        #
        super(ConstantNNHamiltonian, self).__init__(size)
        self.constant = True
        self.int_left = [[] for i in range(size-1)]
        self.int_right = [[] for i in range(size-1)]
        self.interactions = [0j]*(size-1)
        if isinstance(dimension, Number):
            dimension = [dimension] * size
        self.dimension_ = dimension

    def add_local_term(self, ndx, operator):
        #
        # Set the local term acting on the given site
        #
        if ndx == 0:
            self.add_interaction_term(ndx, operator, np.eye(self.dimension(1)))
        elif ndx == self.size-1:
            self.add_interaction_term(ndx-1, np.eye(self.dimension(ndx-1)), operator)
        else:
            self.add_interaction_term(ndx-1, np.eye(self.dimension(ndx-1)), 0.5*operator)
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
        self.interactions[ndx] += np.kron(L, R)
        
    def dimension(self, ndx):
        return self.dimension_[ndx]

    def interaction_term(self, ndx, t=0.0):
        #for (L, R) in zip(self.int_left[ndx], self.int_right[ndx]):
        #self.interactions[ndx] = sum([np.kron(L, R) for (L, R) in zip(self.int_left[ndx], self.int_right[ndx])])
        return self.interactions[ndx]
    
    def constant(self):
        return True
            

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
            H.add_local_term(ndx, local_term)
    return H
