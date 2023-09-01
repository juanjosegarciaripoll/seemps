from math import cos, sin
import numpy as np


class InvalidOperation(TypeError):
    """Exception for operations with invalid or non-matching arguments."""

    def __init__(self, op, *args):
        super().__init__(
            f"Invalid operation {op} between arguments of types {(type(x) for x in args)}"
        )


def take_from_list(O, i):
    if type(O) == list:
        return O[i]
    else:
        return O


DEBUG = 0


# TODO: Find a faster way to do logs. Currently `log` always gets called
# We should find a way to replace calls to log in the code with an if-statement
# that checks `DEBUG`
def log(*args, debug_level=1):
    """Optionally log informative messages to the console.

    Logging is only active when :var:`~seemps.tools.DEBUG` is True or an
    integer above or equal to the given `debug_level`.

    Parameters
    ----------
    *args : str
        Strings to be output
    debug_level : int, default = 1
        Level of messages to log
    """
    if DEBUG and (DEBUG is True or DEBUG >= debug_level):
        print(*args)


def random_isometry(N, M=None):
    """Returns a random isometry with size `(M, N)`.

    Parameters
    ----------
    N, M : int
        Size of the isometry, with `N` defaulting to `M`.

    Returns
    -------
    Operator
        A dense matrix for the isometry.
    """
    if M is None:
        M = N
    U = np.random.rand(N, M)
    U, _, V = np.linalg.svd(U, full_matrices=False)
    if M <= N:
        return U
    else:
        return V


σx = np.array([[0.0, 1.0], [1.0, 0.0]])
σz = np.array([[1.0, 0.0], [0.0, -1.0]])
σy = -1j * σz @ σx


def random_Pauli():
    """Random rotation generated by Pauli matrices."""
    r = np.random.rand(2)
    θ = (2 * r[0] - 1) * np.pi
    ϕ = r[1] * np.pi
    return cos(ϕ) * (cos(θ) * σx + sin(θ) * σy) + sin(ϕ) * σz


def creation(d):
    """Bosonic creation operator for a Hilbert space with occupations 0 to `d-1`."""
    return np.diag(np.sqrt(np.arange(1, d)), -1).astype(complex)


def annihilation(d):
    """Bosonic annihilation operator for a Hilbert space with occupations 0 to `d-1`."""
    return np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)


def mydot(a, b):
    """Contract last index of a with first index of b."""
    lefta = a.shape[:-1]
    rightb = b.shape[1:]
    return np.dot(a, b.reshape(b.shape[0], -1)).reshape(lefta + rightb)
