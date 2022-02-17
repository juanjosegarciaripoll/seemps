
import unittest
import numpy as np
from mps.state import TensorArray

class TestTensorArray(unittest.TestCase):

    def setUp(self):
        self.product_state = [np.reshape([1.0, 2.0], (1, 2, 1)),
                              np.reshape([3.0, 5.0], (1, 2, 1)),
                              np.reshape([7.0, 11.0], (1, 2, 1))]

    def test_independence(self):
        #
        # If we create a TestArray, it can be destructively modified without
        # affecting it original list.
        #
        data = self.product_state.copy()
        A = TensorArray(data)
        for i in range(A.size):
            A[i] = np.reshape([13, 15], (1, 2, 1))
            self.assertTrue(np.all(A[i] != data[i]))
            self.assertTrue(np.all(data[i] == self.product_state[i]))

    def test_copy_independence(self):
        #
        # If we clone a TestArray, it can be destructively modified without
        # affecting its sibling.
        #
        A = TensorArray(self.product_state.copy())
        B = A.copy()
        for i in range(A.size):
            A[i] = np.reshape([13, 15], (1, 2, 1))
            self.assertTrue(np.all(A[i] != B[i]))
            self.assertTrue(np.all(B[i] == self.product_state[i]))

    def test_sharing(self):
        #
        # The clone of a TensorArray shares the same tensors
        #
        data = [x.copy() for x in self.product_state]
        A = TensorArray(data)
        B = A.copy()
        for i in range(A.size):
            A[i][0,0,0] = 17.0
            self.assertTrue(np.all(A[i] == B[i]))
