import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2

import qtpu

from _helper import simple_circuit


def run_circuit_qtpu(circuit: QuantumCircuit):
    """
    Runs the given circuit using QTPU.
    Returns the expectation value of the circuit of the
    Z operator applied to all qubits that are measured.
    """
    # cut the circuit into two halves
    cut_circ = qtpu.cut(circuit, num_qubits=circuit.num_qubits // 2)

    # convert the circuit into a hybrid tensor network
    hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)

    for i, subcirc in enumerate(hybrid_tn.subcircuits):
        print(f"Subcircuit {i}:")
        print(subcirc)
        print("--------------------")

    # evaluate the hybrid tensor network to a classical tensor network
    tn = qtpu.evaluate(hybrid_tn)

    # contract the classical tensor network
    res = tn.contract(all, optimize="auto-hq", output_inds=[])
    return res


def run_comparison(circuit: QuantumCircuit):
    circuit = circuit.remove_final_measurements(inplace=False)
    return (
        EstimatorV2()
        .run([(circuit, "Z" * circuit.num_qubits)], precision=0.0000)
        .result()[0]
        .data.evs
    )


def main():
    circuit = simple_circuit(4)
    qtpu_res = run_circuit_qtpu(circuit)
    qiskit_res = run_comparison(circuit)
    print(f"QTPU result: {qtpu_res}")
    print(f"Qiskit result: {qiskit_res}")
    print(f"Error: {round(abs(qtpu_res - qiskit_res), 5)}")


if __name__ == "__main__":
    main()
