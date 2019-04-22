import numpy as np


def similar(A, B, **kwdargs):
    return (A.shape == B.shape) & np.all(np.isclose(A, B, **kwdargs))


def almostIdentity(L, places=7):
    return np.all(np.isclose(L, np.eye(L.shape[0]), atol=10**(-places)))


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
        A = np.reshape(A, (a*i, b))
        C = A.T.conj() @ A
    else:
        b, i, a = A.shape
        A = np.reshape(A, (b, i*a))
        C = A @ A.T.conj()
    return almostIdentity(C)
import mps.state


def test_over_random_mps(function, d=2, N=10, D=10, repeats=10):
    for nqubits in range(1, N+1):
        for _ in range(repeats):
            function(mps.state.random(d, N, D))
