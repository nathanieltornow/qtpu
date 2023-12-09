import itertools
import multiprocessing as mp

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import QuantumRegister as Fragment
from tensornetwork.backends.numpy.numpy_backend import NumPyBackend

from qvm.quasi_distr import QuasiDistr, prepare_quasidist
from qvm.virtual_circuit import VirtualCircuit
from qvm.virtual_gates import VirtualGateEndpoint

InstanceLabelType = tuple[int, ...]


class Virtualizer:
    def __init__(self, virtual_circuit: VirtualCircuit) -> None:
        self._virtual_circuit = virtual_circuit

    def instantiations(self) -> dict[Fragment, list[QuantumCircuit]]:
        return {
            fragment: [
                self._instantiate_fragment(fragment, inst_label)
                for inst_label in self._instance_labels(fragment)
            ]
            for fragment in self._virtual_circuit.fragment_circuits.keys()
        }

    def knit(
        self, results: dict[Fragment, list[QuasiDistr]], num_processes: int = 4
    ) -> QuasiDistr:
        num_clbits = self._virtual_circuit.circuit.num_clbits
        prepared_results = {
            fragment: np.array(
                [prepare_quasidist(qd, num_clbits) for qd in qds], dtype=object
            )
            for fragment, qds in results.items()
        }
        fragments = list(self._virtual_circuit.fragment_circuits.keys())
        for frag in fragments:
            shape = tuple(
                inst.num_instantiations
                for inst in self._virtual_circuit.virtual_gates_in_fragment(frag)
            )
            prepared_results[frag] = prepared_results[frag].reshape(shape)

        coefficient_matrix: list[np.array] = [
            vg.coefficients() for vg in self._virtual_circuit.virtual_gates
        ]
        total_dim = np.prod(
            [vg.num_instantiations for vg in self._virtual_circuit.virtual_gates]
        )

        all_coeffiecients = np.array(
            [np.prod(coeffs) for coeffs in itertools.product(*coefficient_matrix)],
            dtype=object,
        )

        full_frag_results = np.ndarray(
            shape=(len(fragments) + 1, total_dim),
            dtype=object,
        )
        for i, (frag, res) in enumerate(prepared_results.items()):
            full_frag_results[i] = np.array(
                [
                    res[self._global_to_local(frag, inst_label)]
                    for inst_label in self._all_instance_labels()
                ]
            )

        full_frag_results[-1] = all_coeffiecients

        chunks = np.array_split(full_frag_results, num_processes, axis=1)
        with mp.Pool(num_processes) as pool:
            knit_results = pool.map(_merge_and_knit, chunks)

        return np.sum(knit_results).nearest_probability_distribution()

    def _global_to_local(
        self, fragment: Fragment, global_instance: InstanceLabelType
    ) -> InstanceLabelType:
        local_instance = []
        for vgate_idx in self._virtual_circuit._frag_to_vgate_ids[fragment]:
            local_instance.append(global_instance[vgate_idx])
        return InstanceLabelType(local_instance)

    def _instantiate_fragment(
        self, fragment: Fragment, inst_label: InstanceLabelType
    ) -> QuantumCircuit:
        frag_circuit = self._virtual_circuit.fragment_circuits[fragment]
        if len(inst_label) == 0:
            return frag_circuit.copy()

        config_register = ClassicalRegister(len(inst_label), "vgate_c")
        new_circuit = QuantumCircuit(
            *frag_circuit.qregs, *(frag_circuit.cregs + [config_register])
        )

        # vgate_idx -> clbit_idx
        applied_vgates: dict[int, int] = {}

        vgate_ctr = 0
        for instr in frag_circuit:
            op, qubits, clbits = instr.operation, instr.qubits, instr.clbits
            if isinstance(op, VirtualGateEndpoint):
                vgate_idx = op.vgate_idx
                if vgate_idx not in applied_vgates:
                    applied_vgates[vgate_idx] = vgate_ctr
                    vgate_ctr += 1

                clbit_idx = applied_vgates[vgate_idx]

                inst_id = inst_label[clbit_idx]
                op = op.instantiate(inst_id).to_instruction()
                clbits = [config_register[clbit_idx]]

            new_circuit.append(op, qubits, clbits)
        return new_circuit.decompose()
        pass

    def _instance_labels(self, fragment: Fragment) -> list[InstanceLabelType]:
        frag_vgates = self._virtual_circuit.virtual_gates_in_fragment(fragment)
        inst_list = [tuple(range(vgate.num_instantiations)) for vgate in frag_vgates]
        return list(itertools.product(*inst_list))

    def _all_instance_labels(self) -> list[InstanceLabelType]:
        inst_list = [
            tuple(range(vgate.num_instantiations))
            for vgate in self._virtual_circuit.virtual_gates
        ]
        return list(itertools.product(*inst_list))


def _merge_and_knit(results: np.ndarray) -> QuasiDistr:
    merged_results = np.prod(results, axis=0)
    return np.sum(merged_results)
