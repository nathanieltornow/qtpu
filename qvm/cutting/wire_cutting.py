from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister

from qvm.util import fold_circuit, unfold_circuit, decompose_qubits
from qvm.virtual_gates import WireCut, VirtualSWAP


def cut_wires_optimal(
    circuit: QuantumCircuit,
    num_fragments: int = 2,
    max_cuts: int = 4,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    from clingo.control import Control
    import importlib.resources

    two_qubit_circ, one_qubit_circs = fold_circuit(circuit)

    asp = _circuit_to_asp(two_qubit_circ)

    with importlib.resources.path("qvm", "asp") as path:
        asp_file = path / "dag_partition.lp"
        asp += asp_file.read_text()

    print(asp)
    print(two_qubit_circ)
    # control = Control()
    # control.configuration.solve.models = 0  # type: ignore
    # control.add("base", [], asp)
    # control.ground([("base", [])])
    # solve_result = control.solve(yield_=True)  # type: ignore
    # opt_model = None
    # for model in solve_result:  # type: ignore
    #     opt_model = model

    # if opt_model is None:
    #     raise ValueError("No solution found.")

    # qubits_sets: list[set[Qubit]] = [set() for _ in range(num_fragments)]
    # for symbol in opt_model.symbols(shown=True):
    #     if symbol != "partition" and len(symbol.arguments) != 2:
    #         continue
    #     qubit_idx, partition = (
    #         symbol.arguments[0].number,
    #         symbol.arguments[1].number,
    #     )
    #     qubits_sets[partition].add(circuit.qubits[qubit_idx])

    # return decompose_qubits(circuit, qubits_sets)


def _wirecuts_to_vswaps(circuit: QuantumCircuit) -> QuantumCircuit:
    if sum(1 for instr in circuit if isinstance(instr, VirtualSWAP)) > 0:
        raise ValueError("Circuit already contains virtual SWAP gates.")
    num_wire_cuts = sum(1 for instr in circuit if isinstance(instr, WireCut))
    if num_wire_cuts == 0:
        return circuit.copy()

    wire_cut_register = QuantumRegister(num_wire_cuts, "wire_cut")

    new_circuit = QuantumCircuit(
        *circuit.qregs,
        wire_cut_register * circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    qubit_map: dict[Qubit, Qubit] = {}
    cut_ctr = 0
    for instr in circuit:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        qubits = [qubit_map.get(qubit, qubit) for qubit in qubits]
        if isinstance(op, WireCut):
            qubit_map[qubits[0]] = wire_cut_register[cut_ctr]
            op = VirtualSWAP()
            qubits = [qubits[0], wire_cut_register[cut_ctr]]
            cut_ctr += 1
        new_circuit.append(op, qubits, clbits)
    return new_circuit


def _circuit_to_asp(circuit: QuantumCircuit) -> str:
    def _next_index_on_qubit(instr_index: int, qubit: Qubit) -> int:
        for i, instr in enumerate(circuit[instr_index + 1 :]):
            if qubit in instr.qubits:
                return i + instr_index + 1
        return -1

    asp = ""
    for i, instr in enumerate(circuit):
        qubits = instr.qubits
        if len(qubits) != 2:
            raise ValueError("Only 2-qubit gates are supported.")
        q0idx, q1idx = circuit.qubits.index(qubits[0]), circuit.qubits.index(qubits[1])
        asp += f"gate({i}, {q0idx}, {q1idx}).\n"
        for qubit in qubits:
            next_instr_index = _next_index_on_qubit(i, qubit)
            if next_instr_index > 0:
                asp += (
                    f"wire({i}, {next_instr_index}, {circuit.qubits.index(qubit)}).\n"
                )
    return asp
