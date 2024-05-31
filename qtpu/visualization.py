import matplotlib.pyplot as plt
import numpy as np


from qtpu.tensor import HybridTensorNetwork


def draw_hybrid_tn(hybrid_tn: HybridTensorNetwork, **kwargs) -> plt.Figure:
    import quimb.tensor as qtn

    tn = qtn.TensorNetwork(
        [
            qtn.Tensor(data=np.zeros(tens.shape), inds=tens.inds, tags=["C"])
            for tens in hybrid_tn.classical_tensors
        ]
        + [
            qtn.Tensor(data=np.zeros(tens.shape), inds=tens.inds, tags=["Q"])
            for tens in hybrid_tn.quantum_tensors
        ],
    )
    return tn.draw(color=["C", "Q"], show_inds='all', return_fig=True, **kwargs)
