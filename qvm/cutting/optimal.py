from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister

from qvm.util import fold_circuit, unfold_circuit, fragment_circuit
from qvm.virtual_gates import WireCut, VirtualSWAP, VIRTUAL_GATE_TYPES


def cut_optimal(
    circuit: QuantumCircuit,
    num_fragments: int = 2,
    max_wire_cuts: int = 4,
    max_gate_cuts: int = 2,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    from clingo.control import Control
    import importlib.resources

    two_qubit_circ, one_qubit_circs = fold_circuit(circuit)

    asp = _circuit_to_asp(two_qubit_circ)

    with importlib.resources.path("qvm", "asp") as path:
        asp_file = path / "dag_partition.lp"
        asp += asp_file.read_text()

    if max_fragment_size is None:
        max_fragment_size = circuit.num_qubits // num_fragments + 1

    asp += f"#const num_fragments = {num_fragments}.\n"
    asp += f":- num_wires_cut(N), N > {max_wire_cuts}.\n"
    asp += f":- num_gates_cut(N), N > {max_gate_cuts}.\n"
    asp += f":- fragment_size(F, S), S > {max_fragment_size}.\n"

    control = Control()
    control.configuration.solve.models = 0  # type: ignore
    control.add("base", [], asp)
    control.ground([("base", [])])
    solve_result = control.solve(yield_=True)  # type: ignore
    opt_model = None
    for model in solve_result:  # type: ignore
        opt_model = model

    if opt_model is None:
        raise ValueError("No solution found.")
    op_to_fragment: dict[int, int] = {}
    for symbol in opt_model.symbols(shown=True):
        if symbol.name != "gate_in_frag":
            continue
        op_idx, frag_idx = symbol.arguments[0].number, symbol.arguments[1].number
        op_to_fragment[op_idx] = frag_idx

    cut_circuit = QuantumCircuit(*two_qubit_circ.qregs, *two_qubit_circ.cregs)

    for i, instr in enumerate(two_qubit_circ):
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
        if op_to_fragment[i] == 0:
            op = VIRTUAL_GATE_TYPES[op.name](op)
            cut_circuit.append(op, qubits, clbits)
            continue

        cut_circuit.append(op, qubits, clbits)
        for qubit in qubits:
            next_op_idx = _next_index_on_qubit(two_qubit_circ, i, qubit)
            if (
                next_op_idx > 0
                and op_to_fragment[next_op_idx] != op_to_fragment[i]
                and op_to_fragment[next_op_idx] != 0
            ):
                wc_circ = QuantumCircuit(1)
                wc_circ.append(WireCut(), [0], [])
                one_qubit_circs[next_op_idx] = one_qubit_circs[next_op_idx].compose(
                    wc_circ, qubit, front=True
                )
    cut_circuit = unfold_circuit(cut_circuit, one_qubit_circs)
    cut_circuit = _wirecuts_to_vswaps(cut_circuit)
    return fragment_circuit(cut_circuit)


def _wirecuts_to_vswaps(circuit: QuantumCircuit) -> QuantumCircuit:
    if sum(1 for instr in circuit if isinstance(instr, VirtualSWAP)) > 0:
        raise ValueError("Circuit already contains virtual SWAP gates.")
    num_wire_cuts = sum(1 for instr in circuit if isinstance(instr.operation, WireCut))
    if num_wire_cuts == 0:
        return circuit.copy()

    wire_cut_register = QuantumRegister(num_wire_cuts, "wire_cut")

    new_circuit = QuantumCircuit(
        *circuit.qregs,
        wire_cut_register,
        *circuit.cregs,
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


def _next_index_on_qubit(
    circuit: QuantumCircuit, instr_index: int, qubit: Qubit
) -> int:
    for i, instr in enumerate(circuit[instr_index + 1 :]):
        if qubit in instr.qubits:
            return i + instr_index + 1
    return -1


def _circuit_to_asp(circuit: QuantumCircuit) -> str:
    asp = ""
    for i, instr in enumerate(circuit):
        qubits = instr.qubits
        if len(qubits) != 2:
            raise ValueError("Only 2-qubit gates are supported.")
        q0idx, q1idx = circuit.qubits.index(qubits[0]), circuit.qubits.index(qubits[1])
        asp += f"gate({i}, {q0idx}, {q1idx}).\n"
        for qubit in qubits:
            next_instr_index = _next_index_on_qubit(circuit, i, qubit)
            if next_instr_index > 0:
                asp += (
                    f"wire({i}, {next_instr_index}, {circuit.qubits.index(qubit)}).\n"
                )
    return asp
