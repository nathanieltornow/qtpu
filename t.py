import jax
import jax.numpy as jnp

def entire_tensor_network_contraction(tensor_list):
    """
    Perform a series of tensor contractions on a list of tensors.
    """
    result = tensor_list[0]
    for i in range(1, len(tensor_list)):
        result = jnp.tensordot(result, tensor_list[i], axes=1)
    return result

# Generate a list of random tensors
tensor_list = [jnp.ones((100, 100)) for _ in range(5)]

# JIT compile the entire tensor network contraction function
jit_entire_tensor_network_contraction = jax.jit(entire_tensor_network_contraction)

# Execute the JIT-compiled function
result = jit_entire_tensor_network_contraction(tensor_list)

# Print the result
print(result)
