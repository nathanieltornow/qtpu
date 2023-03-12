import itertools
from multiprocessing.pool import Pool

from qiskit.circuit import (ClassicalRegister, QuantumCircuit, QuantumRegister,
                            Qubit)

from qvm.quasi_distr import QuasiDistr
from qvm.virtual_gates import VirtualBinaryGate


class Virtualizer:
    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:
        self._circuit = circuit.copy()
        self._results: dict[QuantumRegister, dict[tuple[int, ...], QuasiDistr]] = {}

    def _virtual_gates(self) -> list[tuple[VirtualBinaryGate, list[Qubit]]]:
        """
        Returns a list of virtual gates and the qubits they act on
        in the circuit.

        Returns:
            list[tuple[VirtualBinaryGate, set[Qubit]]]: The list
                of virtual gates and the qubits they act on.
        """
        return [
            (instr.operation, list(instr.qubits))
            for instr in self._circuit.data
            if isinstance(instr.operation, VirtualBinaryGate)
        ]

    def _global_inst_labels(self) -> list[tuple[int, ...]]:
        """
        Returns all possible instantiation labels in order for the virtual gates.

        Returns:
            list[tuple[int, ...]]: The list of instantiation labels.
        """
        inst_l = [range(len(vg._instantiations())) for vg, _ in self._virtual_gates()]
        return list(itertools.product(*inst_l))

    def _frag_inst_labels(self, fragment: QuantumRegister) -> list[tuple[int, ...]]:
        """
        Returns all possible instantiation labels in order for the virtual gates
        that act on the given fragment.

        Args:
            fragment (QuantumRegister): The fragment.

        Returns:
            list[tuple[int, ...]]: The list of instantiation labels.
        """
        inst_l = [
            tuple(range(len(vg._instantiations())))
            if set(qubits) & set(fragment)
            else (-1,)
            for vg, qubits in self._virtual_gates()
        ]
        return list(itertools.product(*inst_l))

    @staticmethod
    def _circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
        qreg = QuantumRegister(1)
        new_circuit = QuantumCircuit(qreg, *circuit.cregs)
        qubit = circuit.qubits[index]
        for instr in circuit.data:
            if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
                new_circuit.append(
                    instr.operation, (new_circuit.qubits[0],), instr.clbits
                )
        return new_circuit

    def _fragment_instance(
        self, fragment: QuantumRegister, inst_label: tuple[int, ...]
    ) -> QuantumCircuit:
        """
        Returns the circuit with the virtual gates instantiated with the given
        instantiation label.

        Args:
            inst_label (tuple[int, ...]): The instantiation label.

        Returns:
            QuantumCircuit: The circuit with the virtual gates instantiated.
        """
        assert len(inst_label) == len(self._virtual_gates())
        conf_reg = ClassicalRegister(len(inst_label), "conf")
        inst_ctr = 0
        inst_circuit = QuantumCircuit(fragment, *(self._circuit.cregs + [conf_reg]))
        for cinstr in self._circuit.data:
            op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
            if isinstance(op, VirtualBinaryGate) and set(qubits) & set(fragment):
                # Virtual gates that act on the fragment are instantiated.
                assert inst_label[inst_ctr] != -1
                vgate_instance = op.instantiate(inst_label[inst_ctr])
                for i, qubit in enumerate(qubits):
                    if qubit in fragment:
                        inst = self._circuit_on_index(vgate_instance, i).to_instruction(
                            label=f"{op.name}({inst_label[inst_ctr]})"
                        )
                        inst_circuit.append(inst, [qubit], [conf_reg[inst_ctr]])
                inst_ctr += 1
            elif isinstance(op, VirtualBinaryGate):
                # Virtual gates that do not act on the fragment are ignored.
                assert inst_label[inst_ctr] == -1
                inst_ctr += 1
            elif set(qubits) <= set(fragment):
                # Non-virtual gates that are fully in the fragment are copied.
                inst_circuit.append(op, qubits, clbits)
        return inst_circuit.decompose()

    def instantiations(self) -> dict[QuantumRegister, list[QuantumCircuit]]:
        return {
            fragment: [
                self._fragment_instance(fragment, inst_label)
                for inst_label in self._frag_inst_labels(fragment)
            ]
            for fragment in self._circuit.qregs
        }

    def put_results(self, fragment: QuantumRegister, results: list[QuasiDistr]) -> None:
        self._results[fragment] = {}
        for inst_label, result in zip(self._frag_inst_labels(fragment), results):
            self._results[fragment][inst_label] = result

    def _global_to_fragment_inst_label(
        self, fragment: QuantumRegister, global_inst_label: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Returns the instantiation label for the given fragment.

        Args:
            fragment (QuantumRegister): The fragment.
            inst_label (tuple[int, ...]): The instantiation label.

        Returns:
            tuple[int, ...]: The instantiation label for the fragment.
        """
        frag_inst_label = []
        for i, (_, qubits) in enumerate(self._virtual_gates()):
            if set(qubits) & set(fragment):
                frag_inst_label.append(global_inst_label[i])
            else:
                frag_inst_label.append(-1)
        return tuple(frag_inst_label)

    def _merge(self, pool: Pool | None = None) -> list[QuasiDistr]:
        """
        Merges the results from the fragments to all O(6^k) circuit instantiations.

        Returns:
            list[QuasiDistr]: The merged results.
        """
        global_inst_labels = self._global_inst_labels()
        if pool is None:
            results = list(map(self._merge_global_inst_label, global_inst_labels))
        else:
            results = pool.map(self._merge_global_inst_label, global_inst_labels)
        return results

    def _merge_global_inst_label(self, global_inst_label: tuple[int, ...]):
        fragments = list(self._results.keys())
        frag_label = self._global_to_fragment_inst_label(
            fragments[0], global_inst_label
        )
        merged_res = self._results[fragments[0]][frag_label]
        for frag in fragments[1:]:
            frag_label = self._global_to_fragment_inst_label(frag, global_inst_label)
            merged_res = merged_res.merge(self._results[frag][frag_label])
        return merged_res

    def knit(self, pool: Pool | None = None) -> QuasiDistr:
        def _chunk(lst: list, n: int) -> list[list]:
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        if not len(self._results) == len(self._circuit.qregs):
            raise ValueError(
                "Not all fragments have been evaluated. "
                "Please evaluate all fragments first."
            )
        results = self._merge(pool)
        vgates, _ = zip(*self._virtual_gates())
        vgates = list(vgates)
        while len(vgates) > 0:
            vg = vgates.pop(-1)
            chunks = _chunk(results, len(vg._instantiations()))
            if pool is None:
                results = list(map(vg.knit, chunks))
            else:
                results = pool.map(vg.knit, chunks)
        return results[0]
