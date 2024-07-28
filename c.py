from catalyst import qjit, measure, cond, for_loop, while_loop, grad
import pennylane as qml
from jax import numpy as jnp
from jax.core import ShapedArray

import cotengra as ctg

ctg.ContractionTree().contract

import jax
import quimb.tensor as qtn
import jaxopt
from jax.lax import fori_loop

dev = qml.device("lightning.qubit", wires=1)
qtn.TensorNetwork

@qml.qnode(dev)
def circuit(param):
    qml.Hadamard(0)
    qml.RY(param, wires=0)
    return qml.expval(qml.PauliZ(0))



def kernel(tensor_list):
    """
    Perform a series of tensor contractions on a list of tensors.
    """
    result = tensor_list[0]
    for i in range(1, len(tensor_list)):
        result = jnp.tensordot(result, tensor_list[i], axes=1)
    return result

tensor_list = [jnp.ones((100, 100)) for _ in range(5)]


expr = jax.make_jaxpr(kernel)(tensor_list)
print(expr)
