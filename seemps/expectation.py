from .typing import *
from .state.environments import *
from .state.mps import MPS


def expectation1(ψ: MPS, O: Operator, site: int) -> Weight:
    """Compute the expectation value <ψ|O|ψ> of an operator O acting on 'site'"""
    return ψ.expectation1(O, site)


def expectation2(
    ψ: MPS, O: Operator, Q: Operator, i: int, j: Optional[int] = None
) -> Weight:
    """Compute the expectation value <ψ|O_i Q_j|ψ> of an operator O acting on
    sites 'i' and 'j', with 'j' defaulting to 'i+1'"""
    return ψ.expectation2(O, Q, i, j)


def all_expectation1(ψ: MPS, O: Union[list[Operator], Operator]) -> Vector:
    """Return all expectation values of operator O acting on ψ. If O is a list
    of operators, a different one is used for each site."""
    return ψ.all_expectation1(O)


def product_expectation(ψ: MPSLike, operator_list: list[Operator]) -> Weight:
    rho = begin_environment()
    for ψi, opi in zip(ψ, operator_list):
        rho = update_left_environment(ψi.conj(), ψi, rho, operator=opi)
    return end_environment(rho)
