import numpy as np
from fid import calculate_fidelity
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator

import qvm

from qvm.runtime.runners import SimRunner
from qvm.runtime.runner import sample_fragments, expval_from_counts
from qvm.runtime.virtualizer import build_tensornetwork


def main():
    circuit = QuantumCircuit(8)
    # circuit.h(range(8))
    # circuit.rzz(3* np.pi/2, range(7), range(1, 8))
    # circuit.cx(range(7), range(1, 8))
    circuit = TwoLocal(
        8,
        rotation_blocks=["rz", "ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=1,
    ).decompose()
    circuit.measure_all()
    circuit = circuit.assign_parameters(
        {param: np.random.randn() / 2 for param in circuit.parameters}
    )

    comp = qvm.CutterCompiler(size_to_reach=4)
    virtual_circuit = comp.run(circuit, budget=2)

    results = sample_fragments(virtual_circuit, SimRunner(), shots=100000)

    tn = build_tensornetwork(virtual_circuit, results)
    # tn.draw(color=["frag_result", "coeff"])

    counts = AerSimulator().run(circuit, shots=100000).result().get_counts()
    print(abs(tn.contract() - expval_from_counts(counts)))


if __name__ == "__main__":
    main()
