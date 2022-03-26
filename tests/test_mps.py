from ctypes.wintypes import PSIZE
from random import betavariate
from sre_parse import State
import unittest
import numpy as np
from mps.state import TensorArray, MPS
from mps.state.mps import MPSList
from tools import *


class StatesFixture(unittest.TestCase):
    def setUp(self):
        self.product_state = [
            np.reshape([1.0, 2.0], (1, 2, 1)),
            np.reshape([3.0, 5.0], (1, 2, 1)),
            np.reshape([7.0, 11.0], (1, 2, 1)),
        ]
        self.product_state_dimension = 2 ** 3
        self.product_state_wavefunction = np.kron(
            [1.0, 2.0], np.kron([3.0, 5.0], [7.0, 11.0])
        )

        self.inhomogeneous_state = [
            np.reshape([1.0, 2.0], (1, 2, 1)),
            np.reshape([3.0, 5.0, -1.0], (1, 3, 1)),
            np.reshape([7.0, -5.0, -9.0, 11.0], (1, 4, 1)),
        ]
        self.inhomogeneous_state_dimension = 2 * 3 * 4
        self.inhomogeneous_state_wavefunction = np.kron(
            [1.0, 2.0], np.kron([3.0, 5.0, -1.0], [7.0, -5.0, -9.0, 11.0])
        )

        self.other_tensor = np.reshape([13, 15], (1, 2, 1))


class TestTensorArray(StatesFixture):
    def test_initial_data_is_copied(self):
        data = self.product_state.copy()
        A = TensorArray(data)
        A[::] = self.other_tensor
        self.assertTrue(contain_different_objects(A, data))

    def test_copy_creates_independent_object(self):
        A = TensorArray(self.product_state.copy())
        B = A.copy()
        A[::] = self.other_tensor
        self.assertTrue(contain_different_objects(A, B))

    def test_copy_references_same_tensors(self):
        A = TensorArray(self.product_state)
        self.assertTrue(contain_same_objects(A, self.product_state))


class TestMPS(StatesFixture):
    def test_initial_data_is_copied(self):
        data = self.product_state.copy()
        A = MPS(data)
        A[::] = self.other_tensor
        self.assertTrue(contain_different_objects(A, data))

    def test_copy_creates_independent_object(self):
        A = MPS(self.product_state.copy())
        B = A.copy()
        A[::] = self.other_tensor
        self.assertTrue(contain_different_objects(A, B))

    def test_copy_references_same_tensors(self):
        A = MPS(self.product_state)
        self.assertTrue(contain_same_objects(A, self.product_state))

    def test_total_dimension_is_product_of_physical_dimensions(self):
        A = MPS(self.inhomogeneous_state)
        self.assertEqual(A.dimension(), self.inhomogeneous_state_dimension)

    def test_to_vector_creates_correct_wavefunction(self):
        A = MPS(self.inhomogeneous_state)
        ψ = A.to_vector()
        self.assertEqual(ψ.shape, (self.inhomogeneous_state_dimension,))
        self.assertTrue(similar(ψ, self.inhomogeneous_state_wavefunction))

    def test_from_vector_recreates_product_states(self):
        A = MPS.from_vector(
            self.inhomogeneous_state_wavefunction, [2, 3, 4], normalize=False
        )
        self.assertTrue(
            all(a.shape == b.shape for a, b in zip(A, self.inhomogeneous_state))
        )
        ψA = A.to_vector()
        print(ψA)
        print(self.inhomogeneous_state_wavefunction)
        self.assertTrue(similar(ψA, self.inhomogeneous_state_wavefunction))


class TestMPSOperations(StatesFixture):
    def test_norm2_returns_wavefunction_norm(self):
        self.assertAlmostEqual(
            MPS(self.inhomogeneous_state).norm2(),
            np.linalg.norm(self.inhomogeneous_state_wavefunction),
        )

    def test_adding_mps_creates_mps_list(self):
        A = MPS(self.inhomogeneous_state)
        B = MPS(self.inhomogeneous_state)
        C = A + B
        self.assertTrue(isinstance(C, MPSList))

    def test_subtracting_mps_creates_mps_list(self):
        A = MPS(self.inhomogeneous_state)
        B = MPS(self.inhomogeneous_state)
        C = A - B
        self.assertTrue(isinstance(C, MPSList))

    def test_scaling_mps_creates_new_object(self):
        A = MPS(self.inhomogeneous_state)
        B = 3.0 * A
        self.assertTrue(B is not A)
        self.assertTrue(contain_different_objects(B, A))


if __name__ == "__main__":
    unittest.main()
