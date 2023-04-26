from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister

from .optimal import cut_optimal


def cut_wires_optimal(
    circuit: QuantumCircuit,
    num_fragments: int = 2,
    max_cuts: int = 4,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    return cut_optimal(
        circuit,
        num_fragments=num_fragments,
        max_wire_cuts=max_cuts,
        max_gate_cuts=0,
        max_fragment_size=max_fragment_size,
    )
