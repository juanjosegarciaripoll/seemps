.. current_module:: seemps

.. _seemps_register:

*****************
Quantum registers
*****************

The MPS is a convenient representation to store, manipulate and interrogate a
quantum register of qubits. We now list some algorithms imported from the world
of quantum computing, which have been reimplemented using MPO's and MPS's.

.. _mps_register_transformations:

Other transformations
---------------------

.. autosummary::
   :toctree: generated/

   ~seemps.register.twoscomplement
   ~seemps.register.qubo_mpo
   ~seemps.register.qubo_exponential_mpo
   ~seemps.register.wavefunction_product

.. _mps_qft:

Fourier transforms
------------------

SeeMPS also provides matrix-product operators and functions that implement
the quantum Fourier transform. In some cases the functions and MPO's act
over the whole quantum register (:func:`qft`, :func:`qft_mpo`,...) and in
other cases you can specify a subset of quantum systems (:func:`qft_nd_mpo`, etc).

.. autosummary::
   :toctree: generated/

   ~seemps.qft.qft
   ~seemps.qft.iqft
   ~seemps.qft.qft_mpo
   ~seemps.qft.iqft_mpo
   ~seemps.qft.qft_flip
   ~seemps.qft.qft_nd_mpo
   ~seemps.qft.iqft_nd_mpo
