import numpy as np
from fid import calculate_fidelity
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator

from qvm.virtual_circuit import VirtualCircuit
from qvm.compiler.virtualization.wire_decomp import OptimalWireCutter

from qvm.runtime.runners import SimRunner
from qvm.runtime.runner import sample_fragments, expval_from_counts
from qvm.runtime.virtualizer import build_tensornetwork


def main():
    circuit = EfficientSU2(6, reps=2).decompose()
    clreg = ClassicalRegister(6)
    circuit.add_register(clreg)
    circuit.measure(range(6), range(6))
    circuit = circuit.bind_parameters(
        {param: np.random.randn() / 2 for param in circuit.parameters}
    )
    print(circuit.draw())

    cp = circuit.copy()
    comp_pass = OptimalWireCutter(4)
    cut_circuit = comp_pass.run(circuit, 2)
    print(cut_circuit.draw())

    virt_circ = VirtualCircuit(cut_circuit)

    results = sample_fragments(virt_circ, SimRunner(), shots=100000)

    tn = build_tensornetwork(virt_circ, results)
    # tn.draw(color=["frag_result", "coeff"])

    counts = AerSimulator().run(circuit, shots=100000).result().get_counts()
    print(abs(tn.contract() - expval_from_counts(counts)))


if __name__ == "__main__":
    main()
