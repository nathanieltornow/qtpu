Sampling
========

qTPU supports sampling from quantum circuits using tensor networks.

Basic Sampling
------------

Here's how to sample from a quantum circuit:

.. literalinclude:: ../../examples/sampling.py
   :language: python
   :start-after: def sample_circuit_qtpu
   :end-before: def run_comparison

The sampling process:

1. Cut the circuit into smaller pieces
2. Convert to hybrid tensor network
3. Evaluate using the SamplerEvaluator
4. Sample from the resulting probability distribution
