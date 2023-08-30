from .tools import *


def investigate_unitary_contraction():
    import timeit

    A = np.random.randn(10, 2, 13)
    A /= np.linalg.norm(A)
    B = np.random.randn(13, 2, 10)
    B /= np.linalg.norm(B)
    U = np.random.randn(2, 2, 2, 2)
    U2 = U.reshape(4, 4)

    def method1():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U)

    path = np.einsum_path("ijk,klm,nrjl -> inrm", A, B, U, optimize="optimal")[0]

    def method2():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U, optimize=path)

    def method3():
        a, d, b = A.shape
        b, e, c = B.shape
        D = d * e
        aux = np.tensordot(A, B, (2, 0)).reshape(a, D, c)
        aux = np.tensordot(U.reshape(D, D), aux, (1, 1)).transpose(1, 0, 2)
        return aux.reshape(a, d, e, c)

    def method4():
        a, d, b = A.shape
        b, e, c = B.shape
        return np.matmul(
            U2, np.matmul(A.reshape(-1, b), B.reshape(b, -1)).reshape(a, -1, c)
        ).reshape(a, d, e, c)

    repeats = 10000
    t = timeit.timeit(method1, number=repeats)
    t = timeit.timeit(method1, number=repeats)
    print("\n----------")
    print(f"Method1 {t/repeats}s")

    t = timeit.timeit(method2, number=repeats)
    t = timeit.timeit(method2, number=repeats)
    print(f"Method2 {t/repeats}s")

    t = timeit.timeit(method3, number=repeats)
    t = timeit.timeit(method3, number=repeats)
    print(f"Method3 {t/repeats}s")

    U2 = U.reshape(4, 4)
    t = timeit.timeit(method4, number=repeats)
    t = timeit.timeit(method4, number=repeats)
    print(f"Method4 {t/repeats}s")

    for i, m in enumerate([method2, method3, method4]):
        err = np.linalg.norm(method1() - m())
        print(f"Method{i+2} error = {err}")


class TestTwoSiteEvolutionFold(TestCase):
    def test_contract_U_A_B(self):
        investigate_unitary_contraction()

        A = self.rng.normal(size=(10, 2, 15))
        B = self.rng.normal(size=(15, 3, 13))
        U = self.rng.normal(size=(2 * 3, 2 * 3))

        exact_contraction = np.einsum(
            "ijk,klm,nrjl -> inrm", A, B, U.reshape(2, 3, 2, 3)
        )
        fast_contraction = seemps.evolution._contract_U_A_B(U, A, B)
        self.assertSimilar(exact_contraction, fast_contraction)
