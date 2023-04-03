import itertools
from multiprocessing.pool import Pool

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister, Qubit

from qvm.quasi_distr import QuasiDistr
from qvm.virtual_gates import VirtualBinaryGate


def _circuit_instance(
    self, fragment: QuantumRegister, inst_label: tuple[int, ...]
) -> QuantumCircuit:
    assert len(inst_label) == len(self._virtual_gates())
    conf_reg = ClassicalRegister(len(inst_label), "conf")
    inst_ctr = 0
    inst_circuit = QuantumCircuit(fragment, *(self._circuit.cregs + [conf_reg]))
    for cinstr in self._circuit.data:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if isinstance(op, VirtualBinaryGate):
            assert inst_label[inst_ctr] != -1
            vgate_instance = op.instantiate(inst_label[inst_ctr])
            inst_circuit = inst_circuit.compose(vgate_instance, qubits)
            inst_ctr += 1
        else:
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
    n = len(vgate.num_instantiations)
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
    if len(vgates) == 0:
        return results[0]

    while len(vgates) > 0:
        vgate = vgates.pop(-1)
        results = _knit_vgate(results, vgate, pool)
    return results[0]
