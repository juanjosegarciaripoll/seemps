import numpy as np
import scipy.sparse as sp
from math import cos, sin, pi

def take_from_list(O, i):
    if type(O) == list:
        return O[i]
    else:
        return O

DEBUG = False

def log(*args):
    if DEBUG:
        print(*args)

def random_isometry(N, M=None):
    if M is None:
        M = N
    U = np.random.rand(N, M)
    U, s, V = np.linalg.svd(U, full_matrices=False)
    if M <= N:
        return U
    else:
        return V       
    


σx = np.array([[0.0, 1.0], [1.0, 0.0]])
σz = np.array([[1.0, 0.0], [0.0, -1.0]])
σy = -1j * σz @ σx


def random_Pauli():
    r = np.random.rand(2)
    θ = (2*r[0]-1) * np.pi
    ϕ = r[1] * np.pi
    return cos(ϕ) * (cos(θ) * σx + sin(θ) * σy) + sin(ϕ) * σz

def creation(d):
    # Returns d dimensional cration operator
    return np.diag(np.sqrt(np.arange(1,d)),-1).astype(complex)

def annihilation(d):
    # Returns d dimensional cration operator
    return np.diag(np.sqrt(np.arange(1,d)),1).astype(complex)
