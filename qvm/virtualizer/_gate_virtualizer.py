import itertools

from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qvm.virtual_gates import VirtualBinaryGate
from qvm.quasi_distr import QuasiDistr


class GateVirtualizer:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit.copy()

    def _virtual_gates(self) -> list[VirtualBinaryGate]:
        return [
            instr.operation
            for instr in self._circuit.data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]

    def _inst_ids(self) -> list[tuple[int, ...]]:
        inst_l = [range(len(vg._instantiations())) for vg in self._virtual_gates()]
        return list(itertools.product(*inst_l))

    def _circuit_instance(self, inst_id: tuple[int, ...]) -> QuantumCircuit:
        assert len(inst_id) == len(self._virtual_gates())
        conf_reg = ClassicalRegister(len(inst_id), "conf")
        inst_circuit = QuantumCircuit(
            *self._circuit.qregs, *self._circuit.cregs, conf_reg
        )
        inst_ctr = 0
        for cinstr in self._circuit.data:
            op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
            if isinstance(op, VirtualBinaryGate):
                op = op.instantiate(inst_id[inst_ctr]).to_instruction(label=f"{op.name}({inst_id[inst_ctr]})")
                clbits = [conf_reg[inst_ctr]]
                inst_ctr += 1
            inst_circuit.append(op, qubits, clbits)
        return inst_circuit

    def instantiations(self) -> list[QuantumCircuit]:
        return [self._circuit_instance(inst_id) for inst_id in self._inst_ids()]

    def knit(self, results: list[QuasiDistr]) -> QuasiDistr:
        def _chunk(lst: list, n: int) -> list[list]:
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        vgates = self._virtual_gates()
        while len(vgates) > 0:
            vg = vgates.pop(-1)
            chunks = _chunk(results, len(vg._instantiations()))
            results = [vg._knit(chunk) for chunk in chunks]
        return results[0]
