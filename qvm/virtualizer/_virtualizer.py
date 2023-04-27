import abc
from multiprocessing.pool import Pool

from qiskit.circuit import QuantumCircuit

from qvm.quasi_distr import QuasiDistr
from qvm.types import Argument, Fragment
from qvm.util import fragment_circuit


class Virtualizer(abc.ABC):
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = fragment_circuit(circuit)

    def fragments(self) -> dict[Fragment, QuantumCircuit]:
        sub_circs: dict[Fragment, QuantumCircuit] = {
            frag: QuantumCircuit(frag, *self._circuit.cregs)
            for frag in self._circuit.qregs
        }
        for cinstr in self._circuit.data:
            op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
            appended = False
            for frag in self._circuit.qregs:
                if set(qubits) <= set(frag):
                    sub_circs[frag].append(op, qubits, clbits)
                    appended = True
                    break
            assert appended or op.name == "barrier"
        return sub_circs

    @abc.abstractmethod
    def instantiate(self) -> dict[Fragment, list[Argument]]:
        ...

    @abc.abstractmethod
    def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
        ...



# class Virtualizer:
#     def __init__(
#         self,
#         circuit: QuantumCircuit,
#     ) -> None:
#         circ = fragment_circuit(circuit)
#         self._vgates = [
#             (instr.operation, instr.qubits)
#             for instr in circ
#             if isinstance(instr.operation, VirtualBinaryGate)
#         ]
#         if len(self._vgates) == 0:
#             raise ValueError("No virtual gates found in the circuit.")

#         self._circuit = _insert_placeholders_for_vgates(circ)

#     def sub_circuits(self) -> dict[Fragment, QuantumCircuit]:
#         sub_circs: dict[Fragment, QuantumCircuit] = {
#             frag: QuantumCircuit(frag, *self._circuit.cregs)
#             for frag in self._circuit.qregs
#         }
#         for cinstr in self._circuit.data:
#             op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
#             appended = False
#             for frag in self._circuit.qregs:
#                 if set(qubits) <= set(frag):
#                     sub_circs[frag].append(op, qubits, clbits)
#                     appended = True
#                     break
#             assert appended or op.name == "barrier"
#         return sub_circs

#     def _global_inst_labels(self) -> list[tuple[int, ...]]:
#         """
#         Returns all possible instantiation labels in order for the virtual gates.

#         Returns:
#             list[tuple[int, ...]]: The list of instantiation labels.
#         """
#         inst_l = [range(len(vg._instantiations())) for vg, _ in self._vgates]
#         return list(itertools.product(*inst_l))

#     def _frag_inst_labels(self, fragment: Fragment) -> list[tuple[int, ...]]:
#         """
#         Returns all possible instantiation labels in order for the virtual gates
#         that act on the given fragment.

#         Args:
#             fragment (QuantumRegister): The fragment.

#         Returns:
#             list[tuple[int, ...]]: The list of instantiation labels.
#         """
#         inst_l = [
#             tuple(range(len(vg._instantiations())))
#             if set(qubits) & set(fragment)
#             else (-1,)
#             for vg, qubits in self._vgates
#         ]
#         return list(itertools.product(*inst_l))

#     @staticmethod
#     def _circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
#         qreg = QuantumRegister(1)
#         new_circuit = QuantumCircuit(qreg, *circuit.cregs)
#         qubit = circuit.qubits[index]
#         for instr in circuit.data:
#             if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
#                 new_circuit.append(
#                     instr.operation, (new_circuit.qubits[0],), instr.clbits
#                 )
#         return new_circuit

#     def _fragment_instance(
#         self, fragment: QuantumRegister, inst_label: tuple[int, ...]
#     ) -> Argument:
#         assert len(inst_label) == len(self._vgates)

#         q_arg = Argument()
#         for vgate_index, inst_id in enumerate(inst_label):
#             if inst_id == -1:
#                 continue
#             vgate, qubits = self._vgates[vgate_index]
#             assert set(qubits) & set(fragment)
#             vgate_instance = vgate.instantiate(inst_label[vgate_index])
#             for i, qubit in enumerate(qubits):
#                 if qubit in fragment:
#                     inst = self._circuit_on_index(vgate_instance, i)
#                     q_arg[f"vg_{vgate_index}_{i}"] = inst
#         return q_arg

#     def instantiations(self, fragment: Fragment) -> list[Argument]:
#         return [
#             self._fragment_instance(fragment, inst_label)
#             for inst_label in self._frag_inst_labels(fragment)
#         ]

#     def _global_to_fragment_inst_label(
#         self, fragment: Fragment, global_inst_label: tuple[int, ...]
#     ) -> tuple[int, ...]:
#         """
#         Returns the instantiation label for the given fragment.

#         Args:
#             fragment (QuantumRegister): The fragment.
#             inst_label (tuple[int, ...]): The instantiation label.

#         Returns:
#             tuple[int, ...]: The instantiation label for the fragment.
#         """
#         frag_inst_label = []
#         for i, (_, qubits) in enumerate(self._vgates):
#             if set(qubits) & set(fragment):
#                 frag_inst_label.append(global_inst_label[i])
#             else:
#                 frag_inst_label.append(-1)
#         return tuple(frag_inst_label)

#     def _merge(self, pool: Pool | None = None) -> list[QuasiDistr]:
#         """
#         Merges the results from the fragments to all O(6^k) circuit instantiations.

#         Returns:
#             list[QuasiDistr]: The merged results.
#         """
#         global_inst_labels = self._global_inst_labels()
#         if pool is None:
#             results = list(map(self._merge_global_inst_label, global_inst_labels))
#         else:
#             results = pool.map(self._merge_global_inst_label, global_inst_labels)
#         return results

#     def _merge_global_inst_label(self, global_inst_label: tuple[int, ...]):
#         fragments = list(self._results.keys())
#         frag_label = self._global_to_fragment_inst_label(
#             fragments[0], global_inst_label
#         )
#         merged_res = self._results[fragments[0]][frag_label]
#         for frag in fragments[1:]:
#             frag_label = self._global_to_fragment_inst_label(frag, global_inst_label)
#             merged_res = merged_res.merge(self._results[frag][frag_label])
#         return merged_res

#     def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
#         pass
#         # def _chunk(lst: list, n: int) -> list[list]:
#         #     return [lst[i : i + n] for i in range(0, len(lst), n)]

#         # if not len(self._results) == len(self._circuit.qregs):
#         #     raise ValueError(
#         #         "Not all fragments have been evaluated. "
#         #         "Please evaluate all fragments first."
#         #     )
#         # with Pool() as pool:
#         #     results = self._merge(pool)
#         #     if len(self._virtual_gates) == 0:
#         #         return results[0]
#         #     print("Merged results")
#         #     vgates, _ = zip(*self._virtual_gates)
#         #     vgates = list(vgates)
#         #     while len(vgates) > 0:
#         #         vg = vgates.pop(-1)
#         #         chunks = _chunk(results, len(vg._instantiations()))
#         #         results = pool.map(vg.knit, chunks)
#         #         print("Knitted vgate")
#         # return results[0]
