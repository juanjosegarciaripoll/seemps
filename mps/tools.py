import numpy as np


def take_from_list(O, i):
    if type(O) == list:
        return O[i]
    else:
        return O

DEBUG = True


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
    
