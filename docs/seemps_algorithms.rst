.. _seemps_algorithms:

**************
MPS algorithms
**************

.. _mps_expectation:

Expectation values
------------------

The following are the main routines to compute scalar products and expectation values. The classes MPS and CanonicalMPS provide additional convenience methods with similar names.

.. automodule:: seemps.expectation
   :members: scprod, expectation1, expectation2, all_expectation1


.. _mps_truncate:

MPS simplification
------------------

.. autofunction:: seemps.truncate.simplify

.. autofunction:: seemps.truncate.combine


