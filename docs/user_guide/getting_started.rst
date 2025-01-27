Getting Started
=============

qTPU allows you to process quantum circuits using a hybrid quantum-classical approach based on tensor networks.

Basic Workflow
------------

1. Cut a quantum circuit into smaller pieces
2. Convert the circuit into a hybrid tensor network
3. Evaluate the hybrid tensor network
4. Process the results

Basic Example
-----------

.. code-block:: python

    from qiskit import QuantumCircuit
    import qtpu

    # generate some quantumcircuit
    circuit = QuantumCircuit(...)

    # cut the circuit into two halves
    cut_circ = qtpu.cut(circuit, num_qubits=circuit.num_qubits // 2)

    # convert the circuit into a hybrid tensor network
    hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)

    # evaluate the hybrid tensor network to a classical tensor network
    tn = qtpu.evaluate(hybrid_tn)

    # contract the classical tensor network
    res = tn.contract(all, optimize="auto-hq", output_inds=[])
