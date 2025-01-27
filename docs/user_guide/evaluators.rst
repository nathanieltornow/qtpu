Evaluators
==========

The qTPU framework provides several evaluators for computing sampling probabilities and expectation values of quantum tensors. These evaluators are designed to work with different backends and samplers.

BackendEvaluator
----------------

The `BackendEvaluator` class is used to compute sampling probabilities of quantum tensors using a Qiskit Backend.

Example usage:

.. code-block:: python

    from qiskit.circuit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qtpu.evaluators import BackendEvaluator
    import qtpu

    # Define a simple quantum circuit
    circuit = QuantumCircuit(...)   

    # Cut the circuit into two halves
    cut_circ = qtpu.cut(circuit, num_qubits=frag_size)

    # Convert the circuit into a hybrid tensor network
    hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)

    backend = AerSimulator()

    # Evaluate the hybrid tensor network to a classical tensor network using sampling
    evaluator = BackendEvaluator(backend=backend)
    tn = qtpu.evaluate(hybrid_tn, evaluator)

    # Sample from the classical tensor network
    sample_results = qtpu.sample(tn, num_samples=10000)
    print(dict(Counter(sample_results)))

SamplerEvaluator
----------------

The `SamplerEvaluator` class is used to compute sampling probabilities of quantum tensors using a Qiskit SamplerV2 primitive.

Example usage:

.. code-block:: python

    from qiskit.circuit import QuantumCircuit
    from qtpu.evaluators import SamplerEvaluator
    import qtpu

    # Define a simple quantum circuit
    circuit = QuantumCircuit(...)   

    # Cut the circuit into two halves
    cut_circ = qtpu.cut(circuit, num_qubits=frag_size)

    # Convert the circuit into a hybrid tensor network
    hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)

    # Evaluate the hybrid tensor network to a classical tensor network using sampling
    evaluator = SamplerEvaluator()
    tn = qtpu.evaluate(hybrid_tn, evaluator)

    # Sample from the classical tensor network
    sample_results = qtpu.sample(tn, num_samples=10000)
    print(dict(Counter(sample_results)))

For more examples, see the `examples/sampler_evaluator.py` file in the examples folder.

ExpvalEvaluator
------------------

The `ExpvalEvaluator` class is used to compute expectation values of quantum tensors using a Qiskit EstimatorV2 primitive.

.. code-block:: python
    
    from qiskit.circuit import QuantumCircuit
    from qtpu.evaluators import ExpvalEvaluator
    import qtpu

    # Define a simple quantum circuit
    circuit = QuantumCircuit(...)   

    # Cut the circuit into two halves
    cut_circ = qtpu.cut(circuit, num_qubits=frag_size)

    # Convert the circuit into a hybrid tensor network
    hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)

    # Evaluate the hybrid tensor network to a classical tensor network
    evaluator = ExpvalEvaluator()
    tn = qtpu.evaluate(hybrid_tn, evaluator)

    # We contract the tensor network to get the expectation value
    res = tn.contract(all, optimize="auto-hq", output_inds=[])

