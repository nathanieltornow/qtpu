import itertools

import numpy as np
import tensornetwork as tn
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import QuantumRegister as Fragment
from tensornetwork.backends.numpy.numpy_backend import NumPyBackend

from qvm.quasi_distr import QuasiDistr, prepare_quasidist
from qvm.virtual_circuit import VirtualCircuit
from qvm.virtual_gates import VirtualGateEndpoint

InstanceLabelType = tuple[int, ...]


class _CustomNumpyBackend(NumPyBackend):
    def __init__(self):
        super().__init__()
        self.name = "custom_numpy"

    def tensordot(self, a, b, axes):
        # we need to override this method because the default implementation
        # does not support tensors with dtype=object
        return np.tensordot(a, b, axes=axes)


tn.set_default_backend(_CustomNumpyBackend())


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

    def knit(self, results: dict[Fragment, list[QuasiDistr]]) -> QuasiDistr:
        tensor_network = self._build_tensor_network(results)
        return (
            tn.contractors.auto(tensor_network)
            .tensor.item()
            .nearest_probability_distribution()
        )

    def _build_tensor_network(
        self, results: dict[Fragment, list[QuasiDistr]]
    ) -> list[tn.Node]:
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

        frag_nodes = {
            frag: tn.Node(prepared_results[frag], name=f"frag_{frag.name}")
            for frag in fragments
        }

        coeff_nodes = []
        for vgate_info in self._virtual_circuit.virtual_gate_information:
            if vgate_info.frag1 == vgate_info.frag2:
                assert vgate_info.frag1_vgate_idx == vgate_info.frag2_vgate_idx
                coeff_node = tn.Node(vgate_info.vgate.coefficients())
                coeff_node[0] ^ frag_nodes[vgate_info.frag1][vgate_info.frag1_vgate_idx]
                coeff_nodes.append(coeff_node)
                continue

            coeff_node = tn.Node(np.diag(vgate_info.vgate.coefficients()))
            coeff_node[0] ^ frag_nodes[vgate_info.frag1][vgate_info.frag1_vgate_idx]
            coeff_node[1] ^ frag_nodes[vgate_info.frag2][vgate_info.frag2_vgate_idx]
            coeff_nodes.append(coeff_node)

        return coeff_nodes + list(frag_nodes.values())

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
