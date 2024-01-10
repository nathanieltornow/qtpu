import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qiskit.circuit import (
    Barrier,
    ClassicalRegister,
    Instruction,
    Parameter,
    QuantumCircuit,
)
from qiskit.circuit import QuantumRegister as Fragment

from qvm.instructions import InstantiableInstruction, VirtualBinaryGate


@dataclass
class VirtualGateInfo:
    vgate: VirtualBinaryGate
    frag1: Fragment
    frag2: Fragment
    frag1_index: int
    frag2_index: int


class VirtualCircuit:
    def __init__(self, circuit: QuantumCircuit) -> None:
        divided_circuit, vgate_infos, inst_ops_per_frag = self._build(circuit)

        self._orig_circuit = circuit
        self._frag_circs = {
            frag: self._circuit_on_fragment(divided_circuit, frag)
            for frag in circuit.qregs
        }
        self._vgate_infos = vgate_infos
        self._inst_ops_per_frag = inst_ops_per_frag

    @property
    def virtual_gate_infos(self) -> list[VirtualGateInfo]:
        return self._vgate_infos

    def instance_operations(self, fragment: Fragment) -> list[InstantiableInstruction]:
        return self._inst_ops_per_frag[fragment]

    @property
    def virtual_gates(self) -> list[VirtualBinaryGate]:
        return [info.vgate for info in self._vgate_infos]
    
    @property
    def num_clbits(self) -> int:
        return self._orig_circuit.num_clbits

    # @property
    # def circuit(self) -> QuantumCircuit:
    #     return self._orig_circuit

    @property
    def fragment_circuits(self) -> dict[Fragment, QuantumCircuit]:
        return self._frag_circs

    @property
    def fragments(self) -> list[Fragment]:
        return self._orig_circuit.qregs

    def num_instantiations(self) -> int:
        inst_per_fragment = {
            frag: np.prod(
                [op.num_instantiations for op in self._inst_ops_per_frag[frag]]
            )
            for frag in self._orig_circuit.qregs
        }
        return sum(list(inst_per_fragment.values()))

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

    def _build(
        self, circuit: QuantumCircuit
    ) -> tuple[
        QuantumCircuit,
        list[VirtualGateInfo],
        dict[Fragment, list[InstantiableInstruction]],
    ]:
        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        vgate_infos: list[VirtualGateInfo] = []
        inst_ops_per_frag: dict[Fragment, list[InstantiableInstruction]] = {
            frag: [] for frag in circuit.qregs
        }

        vgate_index = 0

        for instr in circuit:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits

            if isinstance(op, VirtualBinaryGate):
                frag1, frag2 = (
                    circuit.find_bit(qubits[0]).registers[0][0],
                    circuit.find_bit(qubits[1]).registers[0][0],
                )
                vgate_infos.append(
                    VirtualGateInfo(
                        vgate=op,
                        frag1=frag1,
                        frag2=frag2,
                        frag1_index=len(inst_ops_per_frag[frag1]),
                        frag2_index=len(inst_ops_per_frag[frag2]),
                    )
                )

                clreg = ClassicalRegister(1, f"vgate_{vgate_index}")
                new_circuit.add_register(clreg)
                clbits = [clreg[0]]

                if frag1 == frag2:  # the virtual gate is in one fragment
                    inst_op = InstantiableInstruction.from_virtual_gate(op, vgate_index)
                    new_circuit.append(
                        inst_op,
                        qubits,
                        clbits,
                    )
                    inst_ops_per_frag[frag1].append(inst_op)

                else:  # the virtual gate is in two fragments
                    frags = [frag1, frag2]
                    inst_ops = InstantiableInstruction.from_virtual_gate_divided(
                        op, vgate_index
                    )
                    for i in range(2):
                        new_circuit.append(
                            inst_ops[i],
                            [qubits[i]],
                            clbits,
                        )
                        inst_ops_per_frag[frags[i]].append(inst_ops[i])

                vgate_index += 1
                continue

            new_circuit.append(op, qubits, clbits)

        return new_circuit, vgate_infos, inst_ops_per_frag
