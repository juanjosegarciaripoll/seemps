import numpy as np
import scipy.sparse as sp
from seemps.state import MPS
import seemps.state


def similar(A, B, **kwdargs):
    if sp.issparse(A):
        A = A.todense()
    elif isinstance(A, MPS):
        A = A.to_vector()
    if sp.issparse(B):
        B = B.todense()
    elif isinstance(B, MPS):
        B = B.to_vector()
    return (A.shape == B.shape) & np.all(np.isclose(A, B, **kwdargs))


def almostIdentity(L, places=7):
    return np.all(np.isclose(L, np.eye(L.shape[0]), atol=10 ** (-places)))


def almostIsometry(A, places=7):
    N, M = A.shape
    if M < N:
        A = A.T.conj() @ A
    else:
        A = A @ A.T.conj()
    return almostIdentity(A, places=places)


def approximateIsometry(A, direction, places=7):
    if direction > 0:
        a, i, b = A.shape
        A = np.reshape(A, (a * i, b))
        C = A.T.conj() @ A
    else:
        b, i, a = A.shape
        A = np.reshape(A, (b, i * a))
        C = A @ A.T.conj()
    return almostIdentity(C)


def contain_different_objects(A, B):
    return all(a is not b for a, b in zip(A, B))


def contain_same_objects(A, B):
    return all(a is b for a, b in zip(A, B))


def contain_similar_tensors(A, B):
    return all(similar(a, b) for a, b in zip(A, B))


def run_over_random_mps(function, d=2, N=10, D=10, repeats=10):
    for nqubits in range(1, N + 1):
        for _ in range(repeats):
            function(seemps.state.random(d, N, D))
