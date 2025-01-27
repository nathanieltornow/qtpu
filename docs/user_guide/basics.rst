Basic Usage
==========

This guide shows how to use qTPU for basic circuit evaluation.

Circuit Evaluation
----------------

The following example demonstrates how to evaluate a quantum circuit using qTPU:

.. literalinclude:: ../../examples/basics.py
   :language: python
   :start-after: def run_circuit_qtpu
   :end-before: def run_comparison

Comparing Results
---------------

You can compare the results with standard Qiskit execution:

.. literalinclude:: ../../examples/basics.py
   :language: python
   :start-after: def run_comparison
   :end-before: def main
