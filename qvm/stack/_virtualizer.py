import itertools
from multiprocessing.pool import Pool

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qvm.cut_library.util import fragment_circuit
from qvm.quasi_distr import QuasiDistr
from qvm.stack._types import PlaceholderGate, QernelArgument
from qvm.virtual_gates import VirtualBinaryGate


class Virtualizer:
    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:
        self._circuit = fragment_circuit(circuit)
        self._results: dict[QuantumRegister, dict[tuple[int, ...], QuasiDistr]] = {}
        self._virtual_gates = []
        num_vgates = sum(
            1
            for instr in self._circuit.data
            if isinstance(instr.operation, VirtualBinaryGate)
        )

        conf_reg: ClassicalRegister = ClassicalRegister(num_vgates, "c_dec")

        vgate_index = 0
        self._qernel = QuantumCircuit(
            *self._circuit.qregs, *self._circuit.cregs, conf_reg
        )
        for cinstr in self._circuit.data:
            op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
            if isinstance(op, VirtualBinaryGate):
                self._virtual_gates.append((op, list(qubits)))
                for i in range(2):
                    self._qernel.append(
                        PlaceholderGate(
                            f"dec_{vgate_index}_{i}", clbit=conf_reg[vgate_index]
                        ),
                        [qubits[i]],
                        [],
                    )
                vgate_index += 1
            else:
                self._qernel.append(op, qubits, clbits)

        sub_qernels: dict[QuantumRegister, QuantumCircuit] = {
            qreg: QuantumCircuit(qreg, *self._qernel.cregs)
            for qreg in self._circuit.qregs
        }
        for cinstr in self._qernel.data:
            op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
            appended = False
            for qreg in self._qernel.qregs:
                if set(qubits) <= set(qreg):
                    sub_qernels[qreg].append(op, qubits, clbits)
                    appended = True
                    break
            assert appended or op.name == "barrier"
        self._sub_qernels = sub_qernels

    def sub_qernels(self) -> dict[QuantumRegister, QuantumCircuit]:
        return self._sub_qernels.copy()

    def _global_inst_labels(self) -> list[tuple[int, ...]]:
        """
        Returns all possible instantiation labels in order for the virtual gates.

        Returns:
            list[tuple[int, ...]]: The list of instantiation labels.
        """
        inst_l = [range(len(vg._instantiations())) for vg, _ in self._virtual_gates]
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
            for vg, qubits in self._virtual_gates
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
    ) -> QernelArgument:
        assert len(inst_label) == len(self._virtual_gates)

        q_arg = QernelArgument()
        for vgate_index, inst_id in enumerate(inst_label):
            if inst_id == -1:
                continue
            vgate, qubits = self._virtual_gates[vgate_index]
            assert set(qubits) & set(fragment)
            vgate_instance = vgate.instantiate(inst_label[vgate_index])
            for i, qubit in enumerate(qubits):
                if qubit in fragment:
                    inst = self._circuit_on_index(vgate_instance, i)
                    q_arg.insertions[f"dec_{vgate_index}_{i}"] = inst
        return q_arg

    def instantiations(self, fragment: QuantumRegister) -> list[QernelArgument]:
        return [
            self._fragment_instance(fragment, inst_label)
            for inst_label in self._frag_inst_labels(fragment)
        ]

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
        for i, (_, qubits) in enumerate(self._virtual_gates):
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
        vgates, _ = zip(*self._virtual_gates)
        vgates = list(vgates)
        while len(vgates) > 0:
            vg = vgates.pop(-1)
            chunks = _chunk(results, len(vg._instantiations()))
            if pool is None:
                results = list(map(vg.knit, chunks))
            else:
                results = pool.map(vg.knit, chunks)
        print(results[0])
        return results[0]
