
import numpy as np
import scipy.linalg
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
        
    def dimension(self, ndx, t=0.0):
        #
        # Return the dimension of the local Hilbert space
        #
        return 0
    
    def interaction_term(self, ndx, t=0.0):
        #
        # Return the interaction between sites (ndx,ndx+1)
        #
        return 0

def _compute_interaction_term(H, ndx, t=0.0):
    """Computes the interaction term between site ndx and ndx+1, including the local terms 
    for the two sites
    
    Arguments:
    H = NNHamiltonian
    ndx = site index
    """
    if isinstance(H.local_terms[ndx], np.ndarray ):            
        if ndx == 0:
            H.interactions[ndx] +=  np.kron(H.local_terms[ndx],
                                               np.eye(H.dimension(ndx+1)))
        else:
            H.interactions[ndx] +=  0.5 * np.kron(H.local_terms[ndx],
                                                     np.eye(H.dimension(ndx+1)))
    if isinstance(H.local_terms[ndx+1], np.ndarray ):            
        if ndx == H.size-2:
            H.interactions[ndx] +=  np.kron(np.eye(H.dimension(ndx)),
                                               H.local_terms[ndx+1])
        else:
            H.interactions[ndx] +=  0.5 * np.kron(np.eye(H.dimension(ndx)),
                                                     H.local_terms[ndx+1])

    return H.interactions[ndx]

class ConstantNNHamiltonian(NNHamiltonian):
    
    def __init__(self, size):
        #
        # Create a nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operators acting on each site (can be different for each site)
        #  - int_left, int_right: list of L and R operators (can be different for each site)
        #
        self.size = size
        self.local_terms = [0] * size
        self.int_left = [[]] * (size-1)
        self.int_right = [[]] * (size-1)
        self.interactions = [0] * (size-1)

    def set_local_term(self, ndx, operator):
        #
        # Set the local term acting on the given site
        #
        self.local_terms[ndx] = operator

    def add_interaction_term(self, ndx, L, R):
        #
        # Add an interaction term $L \otimes R$ acting on sites 'ndx' and 'ndx+1'
        #
        # Add to int_left, int_right
        #
        # Update the self.interactions[ndx] term
        self.int_left[ndx].append(L)
        self.int_right[ndx].append(R)
        self.interactions[ndx] += np.kron(L,R)
        
    def dimension(self, ndx, t=0.0):
        #
        # Return the dimension of the local Hilbert space
        #
        if ndx == self.size -1:
            return self.int_right[ndx-1][0].shape[0]
        else:
            return self.int_left[ndx][0].shape[0]
    
    #def interaction_term(self, ndx, t=0.0):
        #
        # Return the interaction between sites (ndx,ndx+1) including the corresponding local terms.
        #
        #return _compute_interaction_term(self, ndx, t=0.0)

class TINNHamiltonian(ConstantNNHamiltonian):
    
    def __init__(self, size, local_term, intL, intR):
        #
        # Create a constant nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operator acting on every site
        #  - int_left: list of L (applied to site ndx) operators
        #  - int_right: list of R (applied to site ndx + 1) operators
        #  - interaction: kronecker product of corresponding L and R pairs
        #
        self.size = size
        self.local_terms = [local_term] * size
        self.int_left = [[]] * (size-1)
        self.int_right = [[]] * (size-1)
        self.intL = intL
        self.intR = intR
        self.interactions = [0] * (size-1)
    
    def interaction_term(self, ndx, t=0.0):
        #
        # Return the interaction between sites (ndx,ndx+1)
        #
        if isinstance(self.interactions[ndx], np.ndarray):
            return self.interactions[ndx]
        
        else:
            for L,R in zip(self.intL, self.intR):
                self.add_interaction_term(ndx, L, R)
               
            return _compute_interaction_term(self, ndx, t=0.0)
        

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
        U = U.reshape(self.H.dimension(start),self.H.dimension(start+1),
                      self.H.dimension(start),self.H.dimension(start+1))
        return U
    
    

def apply_2siteTrotter(U, ψ, start):
    return np.einsum('ijk,klm,prjl -> iprm', ψ[start],
                     ψ[start+1], U)


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
    def update_two_site(start, dr):
        # Apply combined local and interaction exponential and move
        AA = apply_2siteTrotter(Trotter.twosite_unitary(start) , ψ, start)
        ψ.update_canonical_2site(AA, dr, tolerance=tol)

    #
    # Loop over ψ, updating pairs of sites acting with the unitary operator
    # made of the interaction and 0.5 times the local terms
    #
    if dr < 0:
        for j in range(ψ.size-2, -1, -2):
            print(ψ.center)
            update_two_site(j, dr)
    else:
        for j in range(0, ψ.size-1, +2):
            print(ψ.center)
            update_two_site(j, dr)        
            
    return ψ


