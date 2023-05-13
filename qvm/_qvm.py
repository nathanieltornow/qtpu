import logging

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from qvm.virtual_gates import VirtualBinaryGate

from qvm.cutting.gate_cutting import bisect, cut_gates_optimal
from qvm.cutting.optimal import cut_optimal
from qvm.cutting.wire_cutting import cut_wires_optimal

from qvm.vroute import vroute_perfect

logger = logging.getLogger("qvm")


def cut(
    circuit: QuantumCircuit,
    technique: str,
    num_fragments: int = 2,
    max_gate_cuts: int = 2,
    max_wire_cuts: int = 2,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    """
    Cuts a circuit into fragments by inserting virtual operations
    according to the given technique.

    Args:
        circuit (QuantumCircuit): The circuit to cut.
        technique (str): The technique to use.

    Returns:
        QuantumCircuit: The cut circuit.
    """
    logger.debug(f"Cutting circuit with technique {technique}.")
    if max_fragment_size is None:
        max_fragment_size = circuit.num_qubits // num_fragments + 1
    if max_fragment_size <= 1:
        raise ValueError("max_fragment_size must be greater than 1.")

    logger.debug(f"max_fragment_size: {max_fragment_size}")

    if technique == "gate_bisection":
        cut_circ = bisect(
            circuit,
            num_fragments=num_fragments,
            max_fragment_size=max_fragment_size,
            max_gate_cuts=max_gate_cuts,
        )

    elif technique == "gate_optimal":
        cut_circ = cut_gates_optimal(
            circuit,
            num_fragments=num_fragments,
            max_wire_cuts=max_wire_cuts,
            max_fragment_size=max_fragment_size,
        )

    elif technique == "wire_optimal":
        cut_circ = cut_wires_optimal(
            circuit,
            num_fragments=num_fragments,
            max_wire_cuts=max_gate_cuts,
            max_fragment_size=max_fragment_size,
        )

    elif technique == "optimal":
        cut_circ = cut_optimal(
            circuit,
            num_fragments=num_fragments,
            max_wire_cuts=max_wire_cuts,
            max_gate_cuts=max_gate_cuts,
            max_fragment_size=max_fragment_size,
        )

    else:
        raise ValueError(f"Unknown cutting technique: {technique}")

    return cut_circ


def vroute(
    circuit: QuantumCircuit,
    technique: str,
    coupling_map: CouplingMap,
    initial_layout: list[int],
    max_gate_cuts: int = 4,
):
    logger.debug(f"vrouting circuit with technique {technique}.")

    if technique == "perfect":
        circuit = vroute_perfect(circuit, coupling_map, initial_layout)

    elif technique == "furthest_qubits":
        raise NotImplementedError

    else:
        raise ValueError(f"Unknown vroute technique: {technique}")

    if (
        sum(1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate))
        > max_gate_cuts
    ):
        raise ValueError(f"Number of swaps exceeds max_gate_cuts ({max_gate_cuts}).")

    return circuit
