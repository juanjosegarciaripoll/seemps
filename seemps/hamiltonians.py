import numpy as np
from .typing import *
from math import sqrt
from .tools import σx, σy, σz
import scipy.sparse as sp  # type: ignore


class NNHamiltonian(object):
    size: int
    constant: bool

    def __init__(self, size: int):
        #
        # Create a nearest-neighbor interaction Hamiltonian
        # of a given size, initially empty.
        #
        self.size = size
        self.constant = False

    def dimension(self, site: int) -> int:
        #
        # Return the dimension of the local Hilbert space
        #
        return 0

    def interaction_term(self, site: int, t=0.0) -> Operator:
        #
        # Return the interaction between sites (site,site+1)
        #
        return 0

    def tomatrix(self, t: float = 0.0) -> Operator:
        """Return a sparse matrix representing the NNHamiltonian on the
        full Hilbert space."""

        # dleft is the dimension of the Hilbert space of sites 0 to (i-1)
        # both included
        dleft = 1
        # H is the Hamiltonian of sites 0 to i, this site included.
        H = 0 * sp.eye(self.dimension(0))
        for i in range(self.size - 1):
            # We extend the existing Hamiltonian to cover site 'i+1'
            H = sp.kron(H, sp.eye(self.dimension(i + 1)))
            # We add now the interaction on the sites (i,i+1)
            H += sp.kron(sp.eye(dleft if dleft else 1), self.interaction_term(i, t))
            # We extend the dimension covered
            dleft *= self.dimension(i)

        return H


class ConstantNNHamiltonian(NNHamiltonian):
    dimensions: list[int]
    interactions: list[Operator]

    def __init__(self, size: int, dimension: Union[int, list[int]]):
        #
        # Create a nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operators acting on each site (can be different for each site)
        #  - int_left, int_right: list of L and R operators (can be different for each site)
        #
        super(ConstantNNHamiltonian, self).__init__(size)
        self.constant = True
        if isinstance(dimension, list):
            self.dimensions = dimension
        else:
            self.dimensions = [dimension] * size
        self.interactions = [
            np.zeros((si * sj, si * sj))
            for si, sj in zip(self.dimensions[:-1], self.dimensions[1:])
        ]

    def add_local_term(self, site: int, operator: Operator) -> "ConstantNNHamiltonian":
        #
        # Set the local term acting on the given site
        #
        if site < 0 or site >= self.size:
            raise IndexError("Site {site} out of bounds in add_local_term()")
        if site == 0:
            self.add_interaction_term(site, operator, np.eye(self.dimensions[1]))
        elif site == self.size - 1:
            self.add_interaction_term(
                site - 1, np.eye(self.dimensions[site - 1]), operator
            )
        else:
            self.add_interaction_term(
                site - 1, np.eye(self.dimensions[site - 1]), 0.5 * operator
            )
            self.add_interaction_term(
                site, 0.5 * operator, np.eye(self.dimensions[site + 1])
            )
        return self

    def add_interaction_term(
        self, site, op1: Operator, op2: Optional[Operator] = None
    ) -> "ConstantNNHamiltonian":
        """Add an interaction term to this Hamiltonian, acting in 'site' and 'site+1'.
        If 'op2' is None, then 'op1' is interpreted as an operator acting on both
        sites in matrix form. If 'op1' and 'op2' are both provided, the operator
        is np.kron(op1, op2)."""
        if site < 0 or site >= self.size - 1:
            raise IndexError("Site {site} out of bounds in add_interaction_term()")
        H12 = op1 if op2 is None else sp.kron(op1, op2)
        if (
            H12.ndim != 2
            or H12.shape[0] != H12.shape[1]
            or H12.shape[1] != self.dimension(site) * self.dimension(site + 1)
        ):
            raise Exception(f"Invalid operators supplied to add_interaction_term()")
        self.interactions[site] = self.interactions[site] + H12
        return self

    def dimension(self, site) -> int:
        return self.dimensions[site]

    def interaction_term(self, site: int, t: float = 0.0) -> Operator:
        return self.interactions[site]


class ConstantTIHamiltonian(ConstantNNHamiltonian):
    """Translationally invariant Hamiltonian with constant nearest-neighbor
    interactions"""

    def __init__(
        self,
        size: int,
        interaction: Optional[Operator] = None,
        local_term: Optional[Operator] = None,
    ):
        """Construct a translationally invariant, constant Hamiltonian with open
        boundaries and fixed interactions.

        Arguments:
        size        -- Number of sites in the model
        interaction -- Two-body operator in matrix form
        local_term  -- operator acting on every site (optional)

        Returns:
        H           -- ConstantNNHamiltonian
        """
        if local_term is not None:
            dimension = len(local_term)
        elif interaction is not None:
            dimension = round(sqrt(interaction.shape[0]))
        else:
            raise Exception("Either interactions or local term must be supplied")

        super().__init__(size, dimension)
        for site in range(size - 1):
            if interaction is not None:
                self.add_interaction_term(site, interaction)
            if local_term is not None:
                self.add_local_term(site, local_term)


class HeisenbergHamiltonian(ConstantTIHamiltonian):
    """Nearest-neighbor Hamiltonian with constant Heisenberg interactions
    over 'size' S=1/2 spins."""

    def __init__(self, size: int):
        return super().__init__(
            size, 0.25 * (sp.kron(σx, σx) + sp.kron(σy, σy) + sp.kron(σz, σz)).real
        )
