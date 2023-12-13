import numpy as np
from fid import calculate_fidelity
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

import qvm


def main():
    circuit = QuantumCircuit(8)
    circuit.h(0)
    circuit.cx(range(7), range(1, 8))
    # circuit = TwoLocal(
    #     8,
    #     rotation_blocks=["rz", "ry"],
    #     entanglement_blocks="cz",
    #     entanglement="linear",
    #     reps=2,
    # ).decompose()
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
    print(result)
    print(calculate_fidelity(circuit, result))
    print(times)


if __name__ == "__main__":
    main()
