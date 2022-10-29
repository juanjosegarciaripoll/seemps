from math import cos, sin
import numpy as np


def take_from_list(O, i):
    if type(O) == list:
        return O[i]
    else:
        return O


DEBUG = 0


def log(*args, debug_level=1):
    if DEBUG and (DEBUG is True or DEBUG >= debug_level):
        print(*args)


def random_isometry(N, M=None):
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
    r = np.random.rand(2)
    θ = (2 * r[0] - 1) * np.pi
    ϕ = r[1] * np.pi
    return cos(ϕ) * (cos(θ) * σx + sin(θ) * σy) + sin(ϕ) * σz


def creation(d):
    """Returns d dimensional bosonic creation operator"""
    return np.diag(np.sqrt(np.arange(1, d)), -1).astype(complex)


def annihilation(d):
    """Returns d dimensional bosonic annihilation operator"""
    return np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)


def mydot(a, b):
    """Contract last index of a with first index of b"""
    lefta = a.shape[:-1]
    rightb = b.shape[1:]
    return np.dot(a, b.reshape(b.shape[0], -1)).reshape(lefta + rightb)
