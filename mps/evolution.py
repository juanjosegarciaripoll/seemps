
import numpy as np
import scipy.linalg
from numbers import Number
import mps.state
import scipy.sparse as sp
from mps.state import _truncate_vector, DEFAULT_TOLERANCE

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
        if isinstance(dimension, Number):
            dimension = [dimension] * size
        self.dimension_ = dimension

    def add_local_term(self, ndx, operator):
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

    if direction > 0:
        ψ.recenter(start)
        for j in range(start, ψ.size-1, +2):
            AA = np.einsum('ijk,klm,prjl -> iprm', ψ[j], ψ[j+1], U[j])
            ψ.update_canonical_2site(AA, j, j+1, +1, tolerance=tol)
        if j == ψ.size-2:
            return ψ.size-3, -1
        else:
            return ψ.size-2, -1
    else:
        ψ.recenter(start)
        for j in range(start, -1, -2):
            AA = np.einsum('ijk,klm,prjl -> iprm', ψ[j], ψ[j+1], U[j])
            ψ.update_canonical_2site(AA, j, j+1, -1, tolerance=tol)
        if j == 0:
            return 1, +1
        else:
            return 0, +1


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
        if timesteps is None:
            timesteps = self.timesteps
        for i in range(self.timesteps):
            #print(i)
            if self.order == 1:
                self.start, self.direction = apply_pairwise_unitaries(self.Udt, self.ψ, self.start, self.direction, tol=self.tolerance)
                self.start, self.direction = apply_pairwise_unitaries(self.Udt, self.ψ, self.start, self.direction, tol=self.tolerance)
            else:
                self.start, self.direction = apply_pairwise_unitaries(self.Udt2, self.ψ, self.start, self.direction, tol=self.tolerance)
                self.start, self.direction = apply_pairwise_unitaries(self.Udt, self.ψ, self.start, self.direction, tol=self.tolerance)
                self.start, self.direction = apply_pairwise_unitaries(self.Udt2, self.ψ, self.start, self.direction, tol=self.tolerance)
        return self.ψ

    def state():
        return self.ψ
