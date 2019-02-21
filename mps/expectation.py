
import numpy as np
import mps.state

def begin_environment():
    """Initiate the computation of a left environment from two MPS."""
    return np.ones((1,1), dtype=np.float64)

def close_environment(ρ):
    """Extract the scalar product from the last environment."""
    return ρ[0,0]

def update_left_environment(B, A, rho, operator=None):
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        A = np.einsum("ji,aib->ajb", operator, A)
    rho = np.einsum("li,ijk->ljk", rho, A)
    return np.einsum("lmn,lmk->nk", B.conj(), rho)

def udpate_right_environment(B, A, rho, operator=None):
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        A = np.einsum("ji,aib->ajb", operator, A)
    rho = np.einsum("ijk,kn->ijn", A, rho)
    return np.einsum("imn,lmn->il", rho, B)

def scprod(ϕ, ψ):
    """Compute the scalar product between matrix product states <ϕ|ψ>."""
    rho = begin_environment()
    for i in range(ψ.size):
        rho = update_left_environment(ϕ[i], ψ[i], rho)
    return close_environment(rho)

def expectation1_non_canonical(ψ, O, site):
    """Compute the expectation value <ψ|O|ψ> of an operator O acting on 'site'"""
    ρL = begin_environment()
    for i in range(0, ψ.size):
        A = ψ[i]
        if i == site:
            OL = update_left_environment(A, A, ρL, operator=O)
        elif i > site:
            OL = update_left_environment(A, A, OL)
        ρL = update_left_environment(A, A, ρL)
    return close_environment(OL)/close_environment(ρL)

def get_operator(O, i):
    #
    # This utility function is used to guess the operator acting on site 'i'
    # If O is a list, it corresponds to the 'i'-th element. Otherwise, we
    # use operator 'O' everywhere.
    #
    if type(O) == list:
        return O[i]
    else:
        return O

def all_expectation1_non_canonical(ψ, O, tol=0):
    """Return all expectation values of operator O acting on ψ. If O is a list
    of operators, a different one is used for each site."""
    
    Oenv = []
    ρL = begin_environment()
    for i in range(ψ.size):
        A = ψ[i]
        Oenv = [update_left_environment(A, A, ρO) for ρO in Oenv] + \
               [update_left_environment(A, A, ρ, operator=get_operator(O,i))]
        ρL = update_left_environment(A, A, ρ)
    return np.array(map(end_environment, Oenv))/end_environment(ρL)

def product_expectation(ψ, operator_list):
    rho = begin_environment(ρ)
    
    for i in range(ψ.size):
        rho = update_left_environment(ψ[i].conj(), ψ[i], rho, operator = operator_list[i])
    
    return close_environment(ρ)
