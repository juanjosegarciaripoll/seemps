
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

    def local_term(self, ndx, t=0.0):
        #
        # Return the local term acting on the ndx site.
        #
        return 0
    
    def interaction_term(self, ndx, t=0.0):
        #
        # Return the interaction between sites (ndx,ndx+1)
        #
        return 0

class TINNHamiltonian(NNHamiltonian):
    
    def __init__(self, local_term, intL, intR):
        #
        # Create a nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operator acting on every site
        #  - int_left: list of L (applied to site ndx) operators
        #  - int_right: list of R (applied to site ndx + 1) operators
        #  - interaction: kronecker product of corresponding L and R pairs
        #
        self.local = local_term
        self.int_left = intL
        self.int_right = intR
        self.interaction = np.array([np.kron(L,R) for L,R in zip(intL, intR)]).sum(axis = 0)

    def local_term(self, ndx, t=0.0):
        #
        # Return the local term acting on the ndx site.
        #
        return self.local
    
    def interaction_term(self, ndx, t=0.0):
        #
        # Return the interaction between sites (ndx,ndx+1)
        #
        return self.interaction

class ConstantNNHamiltonian(NNHamiltonian):
    
    def __init__(self, size):
        #
        # Create a nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operators acting on each site (can be different for each site)
        #  - int_left, int_right: list of L and R operators (can be different for each site)
        #
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
        # Add an interaction term $L \otimes R$ acting on site 'ndx'
        #
        # Add to int_left, int_right
        #
        # Update the self.interactions[ndx] term
        self.int_left[ndx].append(L)
        self.int_right[ndx].append(R)
        self.interactions[ndx] += np.kron(L,R)

    def local_term(self, ndx, t=0.0):
        #
        # Return the local term acting on the ndx site.
        #
        return self.local_terms[ndx]
    
    def interaction_term(self, ndx, t=0.0):
        #
        # Return the interaction between sites (ndx,ndx+1)
        #
        return self.interactions[ndx]

class Trotter_unitaries(object):
    """"Create Trotter unitarities from a nearest-neighbor interaction Hamiltonian.
        
    Attributes:
    H = NNHamiltonian
    δt = Time step
    evenodd = 0, 1 depending on Trotter step
    """

    def __init__(self, H, δt):
        self.H = H
        self.δt = δt
        self.tensors = []
        
    def local_unitary(self, start):
        """Creates one-site exponentials from local H terms,
        they are applied to first and last sites depending on evenodd value"""
        Hloc = self.H.local_term(start)
        U = scipy.linalg.expm(-1j * self.δt * 0.5 * Hloc)
        return U
        
    def twosite_unitary(self, start):
        """Creates twp-site exponentials from interaction H terms"""
        Hloc1 = self.H.local_term(start)
        Hloc2 = self.H.local_term(start+1)
        H12 = self.H.interaction_term(start) + \
              0.5 * (np.kron(Hloc1, np.eye(Hloc2.shape[0])) + \
                     np.kron(np.eye(Hloc1.shape[0]), Hloc2))
        U = scipy.linalg.expm(-1j * self.δt * H12)
        U = U.reshape(Hloc1.shape[0],Hloc2.shape[0],Hloc1.shape[0],Hloc2.shape[0])
        return U
    
    

    

def apply_1siteTrotter(U, ψ, start):
    return np.einsum("ijk,mj -> imk ", ψ[start], U)

def apply_2siteTrotter(U, ψ, start):
    return np.einsum('ijk,klm,prjl -> iprm', ψ[start],
                     ψ[start+1], U)


def TEBD_sweep(H, ψ, δt, evenodd, tol=0):
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
    def update_local_site(start, dr):
        # Apply local exponential and move
        A = apply_1siteTrotter(Trotter.local_unitary(start) , ψ, start)
        ψ.update_canonical(A, dr, tolerance=tol)
        
    def update_two_site(start, dr):
        # Apply combined local and interaction exponential and move
        AA = apply_2siteTrotter(Trotter.twosite_unitary(start) , ψ, start)
        ψ.update_canonical_2site(AA, dr, tolerance=tol)


    if ψ.center == 0:
        # Move rightwards
        dr = +1
    else:
        # Move leftwards
        dr = -1
    
    #
    # Loop over ψ, updating pairs of sites acting with the unitary operator
    # made of the interaction and 0.5 times the local terms
    #
    if dr < 0:
        j = ψ.size - 1
        if j%2 == evenodd:
            update_local_site(j, dr)
            j -= 1
        while j > 0:
            update_two_site(j - 1, dr)
            j -= 2
        if j == 0:
            update_local_site(j, dr)
    else:
        j = 0
        if j != evenodd:
            update_local_site(j, dr)
            j += 1
        while j < ψ.size - 1:
            update_two_site(j, dr)
            j += 2
        if j == ψ.size - 1:
            update_local_site(j, dr)            
            
    return ψ


