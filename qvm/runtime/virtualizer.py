import abc
from typing import Iterable
import itertools

import numpy as np
from numpy.typing import NDArray

import quimb.tensor as qtn

from qiskit.circuit import Parameter, QuantumRegister as Fragment

from qvm.virtual_circuit import VirtualCircuit


def knit(
    virtual_circuit: VirtualCircuit, results: dict[Fragment, NDArray[np.float32]]
) -> NDArray[np.float32]:
    tensor_network = build_tensornetwork(virtual_circuit, results)
    return tensor_network.contract(all)


def generate_instance_parameters(
    virtual_circuit: VirtualCircuit,
) -> dict[Fragment, Iterable[dict[Parameter, float]]]:
    return {
        frag: _gen_inst_parameters_for_fragment(virtual_circuit, frag)
        for frag in virtual_circuit.fragments
    }


def _gen_inst_parameters_for_fragment(
    virtual_circuit: VirtualCircuit, fragment: Fragment
) -> Iterable[dict[Parameter, float]]:
    inst_ops = virtual_circuit.instance_operations(fragment)
    params = [op.param for op in inst_ops]
    n_insts = [op.num_instantiations for op in inst_ops]

    for inst_label in itertools.product(*[range(n) for n in n_insts]):
        yield {param: inst_label[i] for i, param in enumerate(params)}


def build_tensornetwork(
    virtual_circuit: VirtualCircuit, results: dict[Fragment, NDArray[np.float32]]
) -> qtn.TensorNetwork:
    all_tensors = []
    fragment_tensors = {}

    for fragment in virtual_circuit.fragments:
        shape = tuple(
            op.num_instantiations
            for op in virtual_circuit.instance_operations(fragment)
        )
        tens = qtn.Tensor(
            results[fragment].reshape(shape),
            inds=list(range(len(shape))),
            tags=[fragment.name, "frag_result"],
        )
        fragment_tensors[fragment] = tens
        all_tensors.append(fragment_tensors[fragment])

    for vgate_info in virtual_circuit.virtual_gate_infos:
        if vgate_info.frag1 == vgate_info.frag2:
            assert vgate_info.frag1_index == vgate_info.frag2_index

            coeff_tensor = qtn.Tensor(
                vgate_info.vgate.coefficients_1d(),
                inds=[0],
                tags=[f"coeff_{vgate_info.vgate.name}", "coeff"],
            )
            all_tensors.append(coeff_tensor)
            qtn.connect(
                coeff_tensor,
                fragment_tensors[vgate_info.frag1][vgate_info.frag1_index],
                0,
                vgate_info.frag1_index,
            )
            continue

        # frag1 != frag2
        coeff_tensor = qtn.Tensor(
            vgate_info.vgate.coefficients_2d(),
            inds=[0, 1],
            tags=[f"coeff_{vgate_info.vgate.name}", "coeff"],
        )
        all_tensors.append(coeff_tensor)
        qtn.connect(
            coeff_tensor,
            fragment_tensors[vgate_info.frag1],
            0,
            vgate_info.frag1_index,
        )
        qtn.connect(
            coeff_tensor,
            fragment_tensors[vgate_info.frag2],
            1,
            vgate_info.frag2_index,
        )

    return qtn.TensorNetwork(all_tensors)
