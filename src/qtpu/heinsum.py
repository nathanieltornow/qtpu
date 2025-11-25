from qtpu.tensor import QuantumTensor, TensorSpec, CTensor

import torch

import cotengra as ctg


class HEinusum:

    def __init__(
        self,
        qtensors: list[QuantumTensor],
        ctensors: list[CTensor],
        input_tensors: list[TensorSpec],
        output_inds: tuple[str, ...],
    ):
        self.qtensors = qtensors
        self.ctensors = ctensors
        self.input_tensors = input_tensors
        self.output_inds = output_inds

        ind_to_char = {}
        inputs = ""

        ind_sizes = {}

        next_char = ord("a")
        for tensor in qtensors + ctensors + input_tensors:
            input_entry = ""
            for i, ind in enumerate(tensor.inds):
                if ind not in ind_to_char:
                    ind_to_char[ind] = chr(next_char)
                    ind_sizes[ind] = tensor.shape[i]
                    next_char += 1

                input_entry += ind_to_char[ind]
                if ind_sizes[ind] != tensor.shape[i]:
                    raise ValueError(f"Index {ind} has inconsistent sizes.")
            inputs += input_entry + ","

        outputs = ""
        for ind in output_inds:
            if ind not in ind_to_char:
                raise ValueError(f"Output index {ind} not found in input tensors.")
            outputs += ind_to_char[ind]

        self._size_dict = ind_sizes
        self.einsum_expr = inputs[:-1] + "->" + outputs

    def op_graph(self) -> ctg.ContractionTree:
        opt = ctg.HyperOptimizer()
        inputs, output = ctg.utils.eq_to_inputs_output(self.einsum_expr)    
        return opt.search(inputs=inputs, output=output, size_dict=self._size_dict)

        
