from dataclasses import dataclass

from qiskit.circuit import Barrier, QuantumCircuit
from qiskit.circuit import QuantumRegister as Fragment

from qvm.frag_meta import FragmentMetadata
from qvm.virtual_gates import VirtualBinaryGate, VirtualGateEndpoint


@dataclass
class VirtualGateInformation:
    vgate: VirtualBinaryGate
    frag1: Fragment
    frag2: Fragment
    frag1_vgate_idx: int
    frag2_vgate_idx: int


class VirtualCircuit:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._vgates = [
            instr.operation
            for instr in circuit
            if isinstance(instr.operation, VirtualBinaryGate)
        ]
        self._orig_circuit = circuit.copy()

        divided_circuit, vgate_infos = self._replace_vgates_with_endpoints(circuit)
        self._vgate_infos = vgate_infos
        self._frag_circs = {
            qreg: self._circuit_on_fragment(divided_circuit, qreg)
            for qreg in circuit.qregs
        }

        self._metadata = {frag: FragmentMetadata() for frag in circuit.qregs}

    @property
    def virtual_gates(self) -> list[VirtualBinaryGate]:
        return self._vgates

    @property
    def virtual_gate_information(self) -> list[VirtualGateInformation]:
        return self._vgate_infos

    @property
    def circuit(self) -> QuantumCircuit:
        return self._orig_circuit

    @property
    def fragment_circuits(self) -> dict[Fragment, QuantumCircuit]:
        return self._frag_circs

    @property
    def metadata(self) -> dict[Fragment, FragmentMetadata]:
        return self._metadata

    def virtual_gates_in_fragment(self, fragment: Fragment) -> list[VirtualBinaryGate]:
        return [
            vgate_info.vgate
            for vgate_info in self._vgate_infos
            if vgate_info.frag1 == fragment or vgate_info.frag2 == fragment
        ]

    @staticmethod
    def _replace_vgates_with_endpoints(
        circuit: QuantumCircuit,
    ) -> tuple[QuantumCircuit, list[VirtualGateInformation]]:
        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        vgate_index = 0

        frag_to_vgate_idx: dict[Fragment, int] = {frag: 0 for frag in circuit.qregs}
        vgate_infos: list[VirtualGateInformation] = []

        for instr in circuit:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if isinstance(op, VirtualBinaryGate):
                for i in range(2):
                    new_circuit.append(
                        VirtualGateEndpoint(op, vgate_idx=vgate_index, qubit_idx=i),
                        [qubits[i]],
                        [],
                    )
                vgate_index += 1

                frag1 = circuit.find_bit(qubits[0]).registers[0][0]
                frag2 = circuit.find_bit(qubits[1]).registers[0][0]
                vgate_infos.append(
                    VirtualGateInformation(
                        op,
                        frag1,
                        frag2,
                        frag_to_vgate_idx[frag1],
                        frag_to_vgate_idx[frag2],
                    )
                )
                frag_to_vgate_idx[frag1] += 1
                frag_to_vgate_idx[frag2] += 1

                continue
            new_circuit.append(op, qubits, clbits)
        return new_circuit, vgate_infos

    @staticmethod
    def _circuit_on_fragment(
        circuit: QuantumCircuit, fragment: Fragment
    ) -> QuantumCircuit:
        new_circuit = QuantumCircuit(fragment, *circuit.cregs)
        for instr in circuit.data:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if set(qubits) <= set(fragment):
                new_circuit.append(op, qubits, clbits)
                continue
            elif isinstance(op, Barrier):
                continue
            elif set(qubits) & set(fragment):
                raise ValueError(
                    f"Circuit contains gates that act on multiple fragments. {op}"
                )
        return new_circuit
