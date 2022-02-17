
import numpy as np
import mps.state

def begin_environment(χ=1):
    """Initiate the computation of a left environment from two MPS. The bond
    dimension χ defaults to 1. Other values are used for states in canonical
    form that we know how to open and close."""
    return np.eye(χ, dtype=np.float64)

def update_left_environment(B, A, rho, operator=None):
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        A = np.einsum("ji,aib->ajb", operator, A)
    rho = np.einsum("li,ijk->ljk", rho, A)
    return np.einsum("lmn,lmk->nk", B.conj(), rho)

def update_right_environment(B, A, rho, operator=None):
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if operator is not None:
        A = np.einsum("ji,aib->ajb", operator, A)
    rho = np.einsum("ijk,kn->ijn", A, rho)
    return np.einsum("ijn,ljn->il", rho, B.conj())

def end_environment(ρ):
    """Extract the scalar product from the last environment."""
    return ρ[0,0]

def join_environments(ρL, ρR):
    """Join left and right environments to produce a scalar."""
    return np.einsum('ij,ji', ρL, ρR)

def scprod(ϕ, ψ):
    """Compute the scalar product between matrix product states <ϕ|ψ>."""
    ρ = begin_environment()
    for i in range(ψ.size):
        ρ = update_left_environment(ϕ[i], ψ[i], ρ)
    return end_environment(ρ)

def expectation1(ψ, O, site):
    """Compute the expectation value <ψ|O|ψ> of an operator O acting on 'site'"""
    ρL = ψ.left_environment(site)
    A = ψ[site]
    OL = update_left_environment(A, A, ρL, operator=O)
    ρR = ψ.right_environment(site)
    return join_environments(OL, ρR)

def expectation2(ψ, O, Q, i, j=None):
    """Compute the expectation value <ψ|O_i Q_j|ψ> of an operator O acting on 
    sites 'i' and 'j', with 'j' defaulting to 'i+1'"""
    if j is None:
        j = i+1
    elif j == i:
        return expectation1(ψ, O @ Q, i)
    elif j < i:
        i, j = j,i 
    OQL = ψ.left_environment(i)
    for ndx in range(i,j+1):
        A = ψ[ndx]
        if ndx == i:
            OQL = update_left_environment(A, A, OQL, operator=O)
        elif ndx == j:
            OQL = update_left_environment(A, A, OQL, operator=Q)
        else:
            OQL = update_left_environment(A, A, OQL)
    return join_environments(OQL, ψ.right_environment(j))

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

def all_expectation1(ψ, O, tol=0):
    """Return all expectation values of operator O acting on ψ. If O is a list
    of operators, a different one is used for each site."""
    
    Oenv = []
    ρ = begin_environment()
    allρR = [ρ] * ψ.size
    for i in range(ψ.size-1,0,-1):
        A = ψ[i]
        ρ = update_right_environment(A, A, ρ)
        allρR[i-1] = ρ

    ρL = begin_environment()
    output = allρR
    for i in range(ψ.size):
        A = ψ[i]
        ρR = allρR[i]
        OρL = update_left_environment(A, A, ρL, operator=get_operator(O,i))
        output[i] = join_environments(OρL, ρR)
        ρL = update_left_environment(A, A, ρL)
    return np.array(output)

def product_expectation(ψ, operator_list):
    rho = begin_environment(ρ)
    
    for i in range(ψ.size):
        rho = update_left_environment(ψ[i].conj(), ψ[i], rho, operator = operator_list[i])
    
    return close_environment(ρ)
