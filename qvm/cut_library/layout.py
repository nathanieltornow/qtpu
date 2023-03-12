from collections import namedtuple
from typing import Optional

from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import BasicSwap

from qvm.cut_library.util import cut_qubit_connections

QubitPairData = namedtuple("QubitPairData", ("distance_on_map", "gate_count"))


def fit_to_coupling_basic_virtualization(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    max_virtualization: Optional[int] = None,
    minimum_qubit_distance: Optional[int] = 2,
) -> QuantumCircuit:
    """
    Fits a circuit to a coupling map using a minimum-effort logic
    analogous to that of qiskit.transpiler.passes.routing.BasicSwap,
    with the exception of considering gate virtualizations as a solution,
    and not only SWAPs, for two-qubit gates on distant qubits.

    This technique virtualizes gates with a higher priority given to more
    distant qubits with a more sparse interaction among them. For example,
    if two qubits are 3 jumps apart on the coupling, but have only 1 gate
    between them, this gate will be virtualized first in comparison to
    another pair of qubits with 4 gates applied on them.

    Args:
        circuit (QuantumCircuit): The logical circuit to map.
        coupling_map (CouplingMap): The coupling map of the physical backend.
        max_virtualization (int): The maximum number of gate virtualizations allowed.
            If None, all gates will be virtualized.
        minimum_qubit_distance (int): The minimum distance between two distant qubits
            before virtualizing. Distances below this threshold will be resolved using
            SWAPs instead.

    Returns:
        QuantumCircuit: The circuit with the virtualizations and the SWAPs incorporated.
    """
    circuit_dag = circuit_to_dag(circuit)

    if len(circuit_dag.qregs) != 1 or circuit_dag.qregs.get("q", None) is None:
        raise TranspilerError("Basic swap runs on physical circuits only")

    canonical_register = circuit_dag.qregs["q"]
    trivial_layout = Layout.generate_trivial_layout(canonical_register).copy()

    qubit_pair_to_distance_sparsity_map: dict[frozenset[Qubit], QubitPairData] = {}

    # Analysis step
    for gate in circuit_dag.two_qubit_ops():
        physical_q0 = trivial_layout[gate.qargs[0]]
        physical_q1 = trivial_layout[gate.qargs[1]]

        distance_in_map = coupling_map.distance(physical_q0, physical_q1)
        if distance_in_map == 1:
            continue

        dict_key = frozenset((gate.qargs[0], gate.qargs[1]))

        if dict_key not in qubit_pair_to_distance_sparsity_map:
            # We count negatively here, since we want our gate count
            # sorted in reverse (fewer first)
            qubit_pair_to_distance_sparsity_map[dict_key] = QubitPairData(
                distance_in_map, -1
            )
        else:
            old_tuple = qubit_pair_to_distance_sparsity_map[dict_key]

            qubit_pair_to_distance_sparsity_map[dict_key] = old_tuple._replace(
                gate_count=old_tuple.gate_count - 1
            )

    # Sort according to the priorities, after filtering too short gates
    pairs_surpassing_threshold = filter(
        lambda x: x[1].distance_on_map >= minimum_qubit_distance,
        qubit_pair_to_distance_sparsity_map.items(),
    )
    criteria_passing_pairs = dict(
        sorted(
            pairs_surpassing_threshold,
            key=lambda x: (x[1].distance_on_map, x[1].gate_count),
            reverse=True,
        )
    )

    # Virtualization step
    virt_credit = (
        max_virtualization
        if max_virtualization is not None
        else len(circuit_dag.two_qubit_ops())
    )
    for pair_idx, pair_data in criteria_passing_pairs.items():
        if virt_credit <= 0:
            break
        available_credit = min(virt_credit, -pair_data.gate_count)
        circuit = cut_qubit_connections(
            circuit=circuit,
            qubit_cons={(*pair_idx,)},
            max_cuts=available_credit,
        )
        virt_credit -= available_credit

    # SWAP Insertion Step
    virt_circuit_dag = circuit_to_dag(circuit)
    swapped_virt_circuit_dag = BasicSwap(coupling_map).run(virt_circuit_dag)

    return dag_to_circuit(swapped_virt_circuit_dag)


def fit_to_coupling_map(
    circuit: QuantumCircuit, coupling_map: CouplingMap
) -> QuantumCircuit:
    if circuit.num_qubits > coupling_map.size():
        raise ValueError("The circuit has more qubits than the coupling map.")

    def _qubit_index(qubit: Qubit):
        return circuit.find_bit(qubit).index

    for cinstr in circuit.data:
        if len(cinstr.qubits) == 1:
            continue
        if len(cinstr.qubits) > 2:
            raise ValueError("Only 1- and 2-qubit gates are supported.")
        qubit_index1, qubit_index2 = [
            circuit.find_bit(qubit).index for qubit in cinstr.qubits
        ]
        if (qubit_index1, qubit_index2) not in coupling_map.get_edges():
            pass
