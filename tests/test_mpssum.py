import numpy as np
from seemps.state import MPS, MPSSum
from .fixture_mps_states import MPSStatesFixture


class TestMPSSum(MPSStatesFixture):
    def make_simple_sum(self):
        A = MPS(self.product_state.copy())
        B = MPS(self.product_state.copy())
        return MPSSum(weights=[1, 2], states=[A, B])

    def test_simple_sums(self):
        A = MPS(self.product_state.copy())
        B = MPS(self.product_state.copy())
        C = MPSSum(weights=[1, 2], states=[A, B])
        self.assertTrue(C.weights == [1, 2])
        self.assertTrue(C.states == [A, B])

    def test_scalar_multiplication_only_changes_weights(self):
        A = self.make_simple_sum()
        B = A * 0.5
        self.assertIsInstance(B, MPSSum)
        self.assertTrue(all(wb == 0.5 * wa for wa, wb in zip(A.weights, B.weights)))
        self.assertEqual(A.states, B.states)

        C = 0.5 * A
        self.assertIsInstance(C, MPSSum)
        self.assertEqual(B.weights, C.weights)
        self.assertEqual(A.states, C.states)

    def test_addition_mpssum_and_mps(self):
        A = self.make_simple_sum()
        B = MPSSum(weights=[0.5], states=[A])
        C = MPS(self.inhomogeneous_state.copy())
        D = B + C
        self.assertEqual(D.weights, [0.5, 1])
        self.assertEqual(D.states, [A, C])

    def test_subtraction_mpssum_and_mps(self):
        A = self.make_simple_sum()
        B = MPSSum(weights=[0.5], states=[A])
        C = MPS(self.inhomogeneous_state.copy())
        D = B - C
        self.assertEqual(D.weights, [0.5, -1])
        self.assertEqual(D.states, [A, C])
