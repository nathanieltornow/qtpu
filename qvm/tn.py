import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray
from qiskit.circuit import QuantumRegister as Fragment

from qvm.virtual_circuit import VirtualCircuit
from qvm.virtual_gates import VirtualMove


def build_tensornetwork(
    virtual_circuit: VirtualCircuit, results: dict[Fragment, NDArray]
) -> qtn.TensorNetwork:
    fragment_tensors = _fragment_tensors(virtual_circuit, results)
    all_tensors = list(fragment_tensors.values())

    for i, vgate_info in enumerate(virtual_circuit.virtual_gate_infos):
        if vgate_info.frag1 == vgate_info.frag2:
            assert vgate_info.frag1_index == vgate_info.frag2_index

            coeff_tensor = qtn.Tensor(
                vgate_info.vgate.coefficients_1d(),
                inds=[f"{vgate_info.frag1.name}_{vgate_info.frag1_index}"],
                tags=[f"C_{i}", "C"],
            )
            all_tensors.append(coeff_tensor)

        elif isinstance(vgate_info.vgate, VirtualMove):
            _preprocess_wire_cut(
                fragment_tensors[vgate_info.frag1],
                vgate_info.frag1_index,
                fragment_tensors[vgate_info.frag2],
                vgate_info.frag2_index,
            )
            qtn.connect(
                fragment_tensors[vgate_info.frag1],
                fragment_tensors[vgate_info.frag2],
                vgate_info.frag1_index,
                vgate_info.frag2_index,
            )

        else:
            # virtual gate across two fragments
            coeff_tensor = qtn.Tensor(
                vgate_info.vgate.coefficients_2d(),
                inds=[
                    f"{vgate_info.frag1.name}_{vgate_info.frag1_index}",
                    f"{vgate_info.frag2.name}_{vgate_info.frag2_index}",
                ],
                tags=[f"C_{i}", "C"],
            )

            all_tensors.append(coeff_tensor)

    return qtn.TensorNetwork(all_tensors)


def build_dummy_tensornetwork(virtual_circuit: VirtualCircuit) -> qtn.TensorNetwork:
    inst_per_fragment = {
        frag: np.prod(
            [op.num_instantiations for op in virtual_circuit.instance_operations(frag)]
        )
        for frag in virtual_circuit.fragments
    }
    dummy_results = {
        frag: np.random.random((inst_per_fragment[frag],))
        for frag in virtual_circuit.fragments
    }
    return build_tensornetwork(virtual_circuit, dummy_results)


def _fragment_tensors(vc: VirtualCircuit, results: dict[Fragment, NDArray]):
    frag_shapes = {
        frag: tuple(op.num_instantiations for op in vc.instance_operations(frag))
        for frag in vc.fragments
    }

    return {
        frag: qtn.Tensor(
            results[frag].reshape(frag_shapes[frag]),
            inds=[f"{frag.name}_{j}" for j in range(len(frag_shapes[frag]))],
            tags=[f"F_{i}", "F"],
        )
        for i, frag in enumerate(vc.fragments)
    }


def _preprocess_wire_cut(
    tensor1: qtn.Tensor, axis1: int, tensor2: qtn.Tensor, axis2: int
):
    t1, t2 = tensor1.data, tensor2.data

    t1 = np.moveaxis(t1, axis1, 0)
    t2 = np.moveaxis(t2, axis2, 0)

    t1[0], t1[1] = (t1[0] + t1[1]) / 2, (t1[0] - t1[1]) / 2
    t1[2] /= 2
    t1[3] /= 2

    t2_01 = t2[0] + t2[1]
    t2[2] *= 2
    t2[2] -= t2_01
    t2[3] *= 2
    t2[3] -= t2_01

    t1 = np.moveaxis(t1, 0, axis1)
    t2 = np.moveaxis(t2, 0, axis2)


if __name__ == "__main__":

    A = 1/2 * np.array([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [-1, -1, 2, 0], [-1, -1, 0, 2]])

    tensA = qtn.Tensor(A, inds=['x', 'o1'], tags=['A'])
    tensB = qtn.Tensor(B, inds=['x', 'o2'], tags=['B'])

    res = tensA & tensB

    print(res.contract(all, optimize='auto-hq').data)
