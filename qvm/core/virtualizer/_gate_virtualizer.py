import itertools
import logging
from multiprocessing.pool import Pool
from time import perf_counter

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qvm.core.quasi_distr import QuasiDistr
from qvm.core.types import Argument, Fragment, PlaceholderGate
from qvm.core.util import fragment_circuit
from qvm.core.virtual_gates import VirtualBinaryGate

from ._virtualizer import Virtualizer


logger = logging.getLogger("qvm")


class GateVirtualizer(Virtualizer):
    def __init__(self, circuit: QuantumCircuit) -> None:
        circuit = fragment_circuit(circuit)
        super().__init__(circuit)
        self._vgates = [
            instr.operation
            for instr in circuit
            if isinstance(instr.operation, VirtualBinaryGate)
        ]
        self._circuit = _insert_placeholders_for_vgates(self._circuit)

    def _knit_distrs(self, distrs: list[QuasiDistr], pool: Pool) -> QuasiDistr:
        num_overhead = 1
        vgates = self._vgates.copy()
        for vgate in vgates:
            num_overhead *= vgate.num_instantiations
        now = perf_counter()
        logger.info(f"Knitting {num_overhead} distributions.")
        assert (
            len(distrs) == num_overhead
        ), f"Number of distributions ({len(distrs)}) does not match the number of instantiations ({num_overhead})."
        while len(vgates) > 0:
            vgate = vgates.pop(-1)
            chunks = _chunk(distrs, vgate.num_instantiations)
            distrs = pool.map(vgate.knit, chunks)
        logger.info(f"Knitting took {perf_counter() - now} seconds.")
        return distrs[0]


class OneFragmentGateVirtualizer(GateVirtualizer):
    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)
        if len(self._circuit.qregs) != 1:
            raise ValueError("Circuit must have exactly one fragment.")

    def instantiate(self) -> dict[Fragment, list[Argument]]:
        arguments = _generate_all_arguments(self._vgates)
        return {self._circuit.qregs[0]: arguments}

    def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
        frag = self._circuit.qregs[0]
        distrs = results[frag]
        return self._knit_distrs(distrs, pool)


class TwoFragmentGateVirtualizer(GateVirtualizer):
    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)
        if len(self._circuit.qregs) != 2:
            raise ValueError("Circuit must have exactly two fragments.")

    @staticmethod
    def _merge_two_distrs(distrs: tuple[QuasiDistr, QuasiDistr]) -> QuasiDistr:
        return distrs[0] * distrs[1]

    @staticmethod
    def _binary_merge(
        distrs1: list[QuasiDistr], distrs2: list[QuasiDistr], pool: Pool
    ) -> list[QuasiDistr]:
        logger.info(f"Binary-Merging {len(distrs1)} distributions.")
        now = perf_counter()
        if len(distrs1) != len(distrs2):
            raise ValueError("Distributions must have the same length.")
        distrs = pool.map(
            TwoFragmentGateVirtualizer._merge_two_distrs, zip(distrs1, distrs2)
        )
        logger.info(f"Binary-Merging took {perf_counter() - now} seconds.")
        return distrs

    def instantiate(self) -> dict[Fragment, list[Argument]]:
        arguments = _generate_all_arguments(self._vgates)
        return {self._circuit.qregs[0]: arguments, self._circuit.qregs[1]: arguments}

    def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
        distrs1 = results[self._circuit.qregs[0]]
        distrs2 = results[self._circuit.qregs[1]]
        merged = TwoFragmentGateVirtualizer._binary_merge(distrs1, distrs2, pool)
        return self._knit_distrs(merged, pool)


def _insert_placeholders_for_vgates(circuit: QuantumCircuit) -> QuantumCircuit:
    num_vgates = sum(
        1 for instr in circuit if isinstance(instr.operation, VirtualBinaryGate)
    )
    conf_reg: ClassicalRegister = ClassicalRegister(num_vgates, "conf")
    vgate_index = 0
    res_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs, conf_reg)
    for cinstr in circuit:
        op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
        if isinstance(op, VirtualBinaryGate):
            for i in range(2):
                res_circuit.append(
                    PlaceholderGate(
                        f"vg_{vgate_index}_{i}", clbit=conf_reg[vgate_index]
                    ),
                    [qubits[i]],
                    [],
                )
            vgate_index += 1
        else:
            res_circuit.append(op, qubits, clbits)
    return res_circuit


def _generate_all_arguments(virtual_gates: list[VirtualBinaryGate]) -> list[Argument]:
    vgate_list = [range(vgate.num_instantiations) for vgate in virtual_gates]
    args = []
    for inst_label in itertools.product(*vgate_list):
        q_arg = Argument()
        for vgate_index, inst_id in enumerate(inst_label):
            vgate = virtual_gates[vgate_index]
            vgate_instance = vgate.instantiate(inst_id)
            for i in range(2):
                inst = _circuit_on_index(vgate_instance, i)
                q_arg[f"vg_{vgate_index}_{i}"] = inst
        args.append(q_arg)
    return args


def _circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
    qreg = QuantumRegister(1)
    new_circuit = QuantumCircuit(qreg, *circuit.cregs)
    qubit = circuit.qubits[index]
    for instr in circuit.data:
        if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
            new_circuit.append(instr.operation, (new_circuit.qubits[0],), instr.clbits)
    return new_circuit


def _chunk(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]
