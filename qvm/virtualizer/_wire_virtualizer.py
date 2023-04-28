from multiprocessing.pool import Pool

from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qvm.types import Fragment, Argument, PlaceholderGate
from qvm.quasi_distr import QuasiDistr
from qvm.virtual_gates import VirtualSWAP
from qvm.util import fragment_circuit

from ._virtualizer import Virtualizer


class SingleWireVirtualizer(Virtualizer):
    """
    A virtualizer which can virtualize a wire in a circuit that
    has only one single wire cut.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        circuit = fragment_circuit(circuit)
        super().__init__(circuit)
        self._circuit = fragment_circuit(circuit)
        self._circuit = _insert_placeholders_for_vswaps(self._circuit)
        vswap_instrs = [
            instr for instr in circuit if isinstance(instr.operation, VirtualSWAP)
        ]
        if len(self._circuit.qregs) != 2 or len(vswap_instrs) != 1:
            raise ValueError("Circuit must have exactly one wire cut.")

        Oqubit, rqubit = vswap_instrs[0].qubits[0], vswap_instrs[0].qubits[1]
        self._Oreg = self._circuit.find_bit(Oqubit).registers[0][0]
        self._rreg = self._circuit.find_bit(rqubit).registers[0][0]
        print(self._Oreg, self._rreg)

    def instantiate(self) -> dict[Fragment, list[Argument]]:
        O_args = [
            {"vswap_0_O": QuantumCircuit(1)},
            {"vswap_0_O": _x_circuit()},
            {"vswap_0_O": _y_circuit()},
        ]
        r_args = [
            {"vswap_0_r": QuantumCircuit(1)},
            {"vswap_0_r": _one_circuit()},
            {"vswap_0_r": _plus_circuit()},
            {"vswap_0_r": _i_circuit()},
        ]
        return {self._Oreg: O_args, self._rreg: r_args}

    def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
        resultsO = results[self._Oreg]
        resultsr = results[self._rreg]

        pZ0, pZ1 = resultsO[0].divide_by_first_bit()
        pZ0 *= 2
        pZ1 *= 2
        pX0, pX1 = resultsO[1].divide_by_first_bit()
        pX = pX0 - pX1
        pY0, pY1 = resultsO[2].divide_by_first_bit()
        pY = pY0 - pY1

        p0, _ = resultsr[0].divide_by_first_bit()
        p1, _ = resultsr[1].divide_by_first_bit()
        pp, _ = resultsr[2].divide_by_first_bit()
        pp = 2 * pp - p0 - p1
        pi, _ = resultsr[3].divide_by_first_bit()
        pi = 2 * pi - p0 - p1

        return 0.5 * (pZ0 * p0 + pZ1 * p1 + pX * pp + pY * pi)


def _x_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(1)
    circ.h(0)
    return circ


def _y_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(1)
    circ.sdg(0)
    circ.h(0)
    return circ


def _one_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(1)
    circ.x(0)
    return circ


def _plus_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(1)
    circ.h(0)
    return circ


def _i_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(1)
    circ.s(0)
    circ.h(0)
    return circ


def _insert_placeholders_for_vswaps(circuit: QuantumCircuit) -> QuantumCircuit:
    num_vswaps = sum(1 for instr in circuit if isinstance(instr.operation, VirtualSWAP))
    conf_reg: ClassicalRegister = ClassicalRegister(num_vswaps, "conf_wire")
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs, conf_reg)
    vswap_ctr = 0
    for instr in circuit:
        op, qubits = instr.operation, instr.qubits
        if isinstance(op, VirtualSWAP):
            new_circuit.append(PlaceholderGate(f"vswap_{vswap_ctr}_O"), [qubits[0]], [])
            new_circuit.measure(qubits[0], conf_reg[vswap_ctr])
            new_circuit.append(PlaceholderGate(f"vswap_{vswap_ctr}_r"), [qubits[1]], [])
            vswap_ctr += 1
            continue
        new_circuit.append(instr)
    return new_circuit
