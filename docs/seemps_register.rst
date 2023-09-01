.. current_module:: seemps

.. _seemps_register:

*****************
Quantum registers
*****************

The MPS is a convenient representation to store, manipulate and interrogate a
quantum register of qubits. We now list some algorithms imported from the world
of quantum computing, which have been reimplemented using MPO's and MPS's.

.. _mps_qft:

Fourier transforms
------------------

.. autosummary::
   :toctree: generated/

   ~seemps.qft.qft
   ~seemps.qft.iqft
   ~seemps.qft.qft_mpo
   ~seemps.qft.iqft_mpo
   ~seemps.qft.qft_flip
   ~seemps.qft.qft_nd_mpo
   ~seemps.qft.iqft_nd_mpo

.. _mps_register_transformations:

Other transformations
---------------------

.. automodule:: seemps.register
   :members:
