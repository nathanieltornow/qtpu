import numpy as np
from fid import calculate_fidelity
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

import qvm


def main():
    circuit = TwoLocal(
        8,
        rotation_blocks=["rz", "ry", "rx"],
        entanglement_blocks="rzz",
        entanglement="linear",
        reps=2,
    ).decompose()
    circuit.measure_all()
    circuit = circuit.bind_parameters(
        {param: np.random.randn() / 2 for param in circuit.parameters}
    )

    print(circuit)
    comp = qvm.CutterCompiler(size_to_reach=4)
    virtual_circuit = comp.run(circuit, budget=2)
    for frag in virtual_circuit.fragment_circuits.values():
        print(frag)
    result, times = qvm.run(virtual_circuit, shots=10000)
    print(calculate_fidelity(circuit, result))
    print(times)


if __name__ == "__main__":
    main()
