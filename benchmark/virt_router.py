import itertools
from multiprocessing.pool import Pool

from qiskit.circuit import (ClassicalRegister, QuantumCircuit, QuantumRegister,
                            Qubit)
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit.transpiler import CouplingMap

from qvm.cut_library.util import (VIRTUAL_GATE_TYPES, circuit_to_qcg,
                                  cut_qubit_connections)
from qvm.quasi_distr import QuasiDistr
from qvm.virtual_gates import VirtualBinaryGate


def _circuit_instance(
    circuit: QuantumCircuit, inst_label: tuple[int, ...]
) -> QuantumCircuit:
    conf_reg = ClassicalRegister(len(inst_label), "zconf")
    inst_ctr = 0
    inst_circuit = QuantumCircuit(*circuit.qregs, *(circuit.cregs + [conf_reg]), name=circuit.name, global_phase=circuit.global_phase, metadata=circuit.metadata)
    for cinstr in circuit.data:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if isinstance(op, VirtualBinaryGate):
            assert inst_label[inst_ctr] != -1
            vgate_instance = op.instantiate(inst_label[inst_ctr]).to_instruction()
            op = vgate_instance
            clbits = [conf_reg[inst_ctr]]
            inst_ctr += 1
        inst_circuit.append(op, qubits, clbits)
    return inst_circuit.decompose()


def instantiate(circuit: QuantumCircuit) -> list[QuantumCircuit]:
    virtual_gates = [
        instr.operation
        for instr in circuit.data
        if isinstance(instr.operation, VirtualBinaryGate)
    ]
    inst_list = [range(len(vg._instantiations())) for vg in virtual_gates]
    instances = []
    for inst_label in itertools.product(*inst_list):
        instances.append(_circuit_instance(circuit, inst_label))
    return instances


def _chunk(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _knit_vgate(
    results: list[QuasiDistr], vgate: VirtualBinaryGate, pool: Pool | None = None
) -> list[QuasiDistr]:
    n = vgate.num_instantiations
    chunks = _chunk(results, n)
    if pool is None:
        return list(map(vgate.knit, chunks))
    else:
        return pool.map(vgate.knit, chunks)


def knit(
    circuit: QuantumCircuit, results: list[QuasiDistr], pool: Pool | None = None
) -> QuasiDistr:
    vgates = [
        instr.operation
        for instr in circuit.data
        if isinstance(instr.operation, VirtualBinaryGate)
    ]    
    while len(vgates) > 0:
        vgate = vgates.pop(-1)
        results = _knit_vgate(results, vgate, pool)
    return results[0]


def route_circuit_trivial(
    circuit: QuantumCircuit,
    init_layout: list[int],
    coupling_map: CouplingMap,
    max_vgates: int,
) -> QuantumCircuit:
    assert circuit.num_qubits == len(init_layout)
    vcirc = QuantumCircuit(
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata,
    )
    init_mapping = {circuit.qubits[i]: init_layout[i] for i in range(len(init_layout))}

    for instr in circuit.data:
        op, qubits, clbits = instr.operation, instr.qubits, instr.clbits

        if len(qubits) == 2 and not op.name == "barrier":
            qubit1, qubit2 = qubits
            if (
                coupling_map.distance(init_mapping[qubit1], init_mapping[qubit2]) > 1
                and max_vgates > 0
            ):
                op = VIRTUAL_GATE_TYPES[op.name](op)
                max_vgates -= 1
        vcirc.append(op, qubits, clbits)
    return vcirc


def virt_furthest_qubits(
    circuit: QuantumCircuit,
    init_layout: list[int],
    coupling_map: CouplingMap,
    num_cuts: int,
) -> QuantumCircuit:
    assert circuit.num_qubits == len(init_layout)
    qubit_map = {circuit.qubits[i]: init_layout[i] for i in range(len(init_layout))}

    qubits_distance: dict[tuple[Qubit, Qubit], int] = {}

    qcg = circuit_to_qcg(circuit)
    for qubit1, qubit2 in itertools.combinations(circuit.qubits, 2):
        if coupling_map.distance(qubit_map[qubit1], qubit_map[qubit2]) > 1:
            if (
                (qubit1, qubit2) not in qubits_distance
                and (
                    qubit2,
                    qubit1,
                )
                not in qubits_distance
                and (qcg.has_edge(qubit1, qubit2) or qcg.has_edge(qubit2, qubit1))
            ):
                qubits_distance[(qubit1, qubit2)] = coupling_map.distance(
                    qubit_map[qubit1], qubit_map[qubit2]
                )
    qubits_distance_list = sorted(
        qubits_distance.items(), key=lambda item: item[1], reverse=True
    )
    # print(set(qubits for qubits, _ in qubits_distance_list))
    return cut_qubit_connections(
        circuit, set(qubits for qubits, _ in qubits_distance_list), num_cuts
    )


# def run(
#     circuit: QuantumCircuit, backend: FakeBackendV2, shots: int, max_vgates: int = 4
# ) -> QuasiDistr:
#     trans_qc = transpile(circuit, backend, optimization_level=3)
#     small_qc = mm.deflate_circuit(trans_qc)
#     init_layout, _, _ = mm.best_overall_layout(small_qc, [backend])

#     vcirc = route_circuit_trivial(
#         small_qc, init_layout, backend.coupling_map, max_vgates
#     )

#     instances = instantiate(vcirc)
#     counts = backend.run(instances, shots=shots).result().get_counts()
#     counts = [counts] if isinstance(counts, dict) else counts
#     distrs = [QuasiDistr.from_counts(c, shots) for c in counts]
#     with Pool() as pool:
#         res = knit(vcirc, distrs, pool)
#     return res
