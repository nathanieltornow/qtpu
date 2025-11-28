"""Hybrid einsum API for specifying tensor network contractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import cotengra as ctg

from qtpu.tensor import QuantumTensor, TensorSpec, CTensor, ISwitch

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class HEinsum:
    """High-level API for specifying hybrid tensor network contractions."""

    def __init__(
        self,
        qtensors: list[QuantumTensor],
        ctensors: list[CTensor],
        input_tensors: list[TensorSpec],
        output_inds: tuple[str, ...],
    ):
        """Initialize a hybrid einsum specification.

        Args:
            qtensors: Quantum circuit tensors.
            ctensors: Classical tensors.
            input_tensors: Input tensor specifications (provided at runtime).
            output_inds: Output indices for the contraction result.

        Raises:
            ValueError: If indices are inconsistent or output indices not found.
        """
        self._qtensors = qtensors
        self._ctensors = ctensors
        self._input_tensors = input_tensors
        self._output_inds = output_inds

        # Build einsum expression from tensor specs
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
                else:
                    # Validate consistent sizes
                    if ind_sizes[ind] != tensor.shape[i]:
                        raise ValueError(
                            f"Index '{ind}' has inconsistent sizes: "
                            f"{ind_sizes[ind]} vs {tensor.shape[i]}"
                        )

                input_entry += ind_to_char[ind]
            inputs += input_entry + ","

        outputs = ""
        for ind in output_inds:
            if ind not in ind_to_char:
                raise ValueError(
                    f"Output index '{ind}' not found in input tensors. "
                    f"Available: {set(ind_to_char.keys())}"
                )
            outputs += ind_to_char[ind]

        # Also store char-based size dict for cotengra compatibility
        self._size_dict = {ind_to_char[ind]: size for ind, size in ind_sizes.items()}
        self._einsum_expr = inputs[:-1] + "->" + outputs

    @property
    def einsum_expr(self) -> str:
        """The einsum expression string."""
        return self._einsum_expr

    @property
    def size_dict(self) -> dict[str, int]:
        """Mapping of single-char indices (used in einsum_expr) to their sizes.

        This is for cotengra compatibility where einsum expressions use
        single-character indices.
        """
        return self._size_dict

    @property
    def quantum_tensors(self) -> list[QuantumTensor]:
        """Quantum circuit tensors."""
        return self._qtensors

    @property
    def classical_tensors(self) -> list[CTensor]:
        """Classical tensors."""
        return self._ctensors

    @property
    def input_tensors(self) -> list[TensorSpec]:
        """Input tensor specifications."""
        return self._input_tensors

    @property
    def output_inds(self) -> tuple[str, ...]:
        """Output indices."""
        return self._output_inds

    @staticmethod
    def from_circuit(circuit: QuantumCircuit) -> HEinsum:
        """Create a HEinsum specification from a quantum circuit.

        This method decomposes the circuit into quantum tensors using
        the quantum-pseudo-density (QPD) representation.

        Args:
            circuit: The quantum circuit to convert.

        Returns:
            HEinsum specification representing the circuit.
        """
        return HEinsum(
            qtensors=[QuantumTensor(circuit)],
            ctensors=[],
            input_tensors=[],
            output_inds=(),
        )

    def to_dummy_tn(
        self, seed: int | None = None
    ) -> tuple[ctg.ContractionTree | None, list[np.ndarray]]:
        """Create dummy random arrays and contraction tree for benchmarking.

        This creates random numpy arrays matching the shapes of the HEinsum
        tensors. Useful for measuring classical contraction time without
        running quantum circuits.

        Args:
            optimize: Contraction path optimizer (default: "auto-hq").
            seed: Random seed for reproducibility.

        Returns:
            A tuple of (arrays, tree) where arrays is a list of random numpy
            arrays matching the tensor shapes, and tree is the optimized
            contraction tree from cotengra.
        """

        if seed is not None:
            np.random.seed(seed)

        arrays = []
        inputs = []

        # Collect all tensors (quantum, classical, input)
        all_tensors = (
            list(self._qtensors) + list(self._ctensors) + list(self._input_tensors)
        )

        for tensor in all_tensors:
            if tensor.shape:
                data = np.random.randn(*tensor.shape).astype(np.float64)
            else:
                # Scalar tensor (empty shape)
                data = np.array(np.random.randn(), dtype=np.float64)
            arrays.append(data)

        if len(all_tensors) == 1:
            return None, arrays

        # Get optimized contraction tree (parallel=False to avoid semaphore leaks)
        opt = ctg.HyperOptimizer(parallel=False, progbar=False)
        inputs, outputs = ctg.utils.eq_to_inputs_output(self.einsum_expr)
        tree = opt.search(inputs, outputs, self.size_dict)

        return tree, arrays


# =============================================================================
# Random Hybrid Tensor Network Generators
# =============================================================================


def _create_qnn_circuit(
    num_qubits: int,
    num_layers: int,
    output_inds: list[tuple[str, int]],
    seed: int | None = None,
) -> QuantumCircuit:
    """Create a QNN circuit with ISwitches at the end for specified output indices.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of variational layers.
        output_inds: List of (index_name, dimension) tuples for ISwitches.
        seed: Random seed for parameter initialization.

    Returns:
        A QNN circuit with ISwitches appended for tensor contraction.
    """
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.circuit.library import RYGate, RZGate, CXGate

    if seed is not None:
        np.random.seed(seed)

    qc = QuantumCircuit(num_qubits, num_qubits)

    # Build variational layers
    for layer in range(num_layers):
        # Single-qubit rotation layer
        for q in range(num_qubits):
            theta = np.random.rand() * 2 * np.pi
            phi = np.random.rand() * 2 * np.pi
            qc.ry(theta, q)
            qc.rz(phi, q)

        # Entangling layer (linear connectivity)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    # Add ISwitches at the end for output indices
    # Each ISwitch goes on a separate qubit (cycling if needed)
    for i, (ind_name, dim) in enumerate(output_inds):
        qubit_idx = i % num_qubits
        param = Parameter(ind_name)

        def make_basis_circuit(k: int, dim: int = dim) -> QuantumCircuit:
            """Create measurement basis rotation for index k."""
            basis_qc = QuantumCircuit(1)
            angle = k * np.pi / dim
            basis_qc.ry(angle, 0)
            return basis_qc

        iswitch = ISwitch(param, 1, dim, make_basis_circuit)
        qc.append(iswitch, [qubit_idx])

    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))

    return qc


def rand_hybrid_equation(
    n_quantum: int,
    n_classical: int,
    reg: int = 3,
    n_out: int = 0,
    d_min: int = 2,
    d_max: int = 4,
    qubits_min: int = 2,
    qubits_max: int = 4,
    layers_min: int = 1,
    layers_max: int = 3,
    seed: int | None = None,
) -> HEinsum:
    """Generate a random hybrid tensor network with quantum and classical tensors.

    Similar to cotengra.utils.rand_equation but generates a HEinsum with
    QuantumTensors (QNN circuits with ISwitches) and CTensors.

    Args:
        n_quantum: Number of quantum tensors (QNN circuits).
        n_classical: Number of classical tensors.
        reg: Average number of indices per tensor (total indices ≈ (n_quantum + n_classical) * reg / 2).
        n_out: Number of output (uncontracted) indices.
        d_min: Minimum dimension size for indices.
        d_max: Maximum dimension size for indices.
        qubits_min: Minimum number of qubits per quantum circuit.
        qubits_max: Maximum number of qubits per quantum circuit.
        layers_min: Minimum number of QNN layers.
        layers_max: Maximum number of QNN layers.
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A random hybrid tensor network specification.

    Example:
        >>> heinsum = rand_hybrid_equation(n_quantum=3, n_classical=2, reg=3, seed=42)
        >>> print(heinsum.einsum_expr)
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n_quantum + n_classical
    n_inds = max(1, n_total * reg // 2)

    # Generate index names and sizes
    ind_names = [f"i{i}" for i in range(n_inds)]
    ind_sizes = {name: np.random.randint(d_min, d_max + 1) for name in ind_names}

    # Assign indices to tensors (each index appears in exactly 2 tensors, or 1 if output)
    # First n_out indices are output indices (appear once)
    output_ind_names = ind_names[:n_out]
    contract_ind_names = ind_names[n_out:]

    # Build tensor index assignments
    tensor_inds: list[list[tuple[str, int]]] = [[] for _ in range(n_total)]

    # Assign output indices (each to one random tensor)
    for ind in output_ind_names:
        t = np.random.randint(n_total)
        tensor_inds[t].append((ind, ind_sizes[ind]))

    # Assign contraction indices (each to exactly 2 tensors)
    for ind in contract_ind_names:
        tensors = np.random.choice(n_total, size=2, replace=False)
        for t in tensors:
            tensor_inds[t].append((ind, ind_sizes[ind]))

    # Ensure each tensor has at least one index
    for t in range(n_total):
        if len(tensor_inds[t]) == 0:
            # Add a dummy contracted index
            dummy_ind = f"dummy_{t}"
            ind_sizes[dummy_ind] = np.random.randint(d_min, d_max + 1)
            other_t = (t + 1) % n_total
            tensor_inds[t].append((dummy_ind, ind_sizes[dummy_ind]))
            tensor_inds[other_t].append((dummy_ind, ind_sizes[dummy_ind]))

    # Create quantum tensors (first n_quantum)
    qtensors = []
    for t in range(n_quantum):
        num_qubits = np.random.randint(qubits_min, qubits_max + 1)
        num_layers = np.random.randint(layers_min, layers_max + 1)
        circuit = _create_qnn_circuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            output_inds=tensor_inds[t],
            seed=None,  # Already seeded above
        )
        qtensors.append(QuantumTensor(circuit))

    # Create classical tensors (remaining n_classical)
    ctensors = []
    for t in range(n_quantum, n_total):
        inds = tuple(name for name, _ in tensor_inds[t])
        shape = tuple(size for _, size in tensor_inds[t])
        data = np.random.randn(*shape).astype(np.float64)
        ctensors.append(CTensor(data, inds))

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=tuple(output_ind_names),
    )


def rand_hybrid_chain(
    n_quantum: int,
    n_classical: int = 0,
    bond_dim: int = 4,
    phys_dim: int = 2,
    qubits: int = 2,
    layers: int = 2,
    open_boundaries: bool = True,
    seed: int | None = None,
) -> HEinsum:
    """Generate a hybrid MPS-like (matrix product state) chain tensor network.

    Creates a linear chain where quantum and classical tensors alternate
    or are placed according to specified pattern. Each tensor connects to
    its neighbors via bond indices.

    Args:
        n_quantum: Number of quantum tensors in the chain.
        n_classical: Number of classical tensors (interspersed or at ends).
        bond_dim: Dimension of bond indices connecting tensors.
        phys_dim: Dimension of physical (output) indices.
        qubits: Number of qubits per quantum circuit.
        layers: Number of QNN layers.
        open_boundaries: If True, chain has open ends. If False, periodic.
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A chain-structured hybrid tensor network.

    Example:
        >>> heinsum = rand_hybrid_chain(n_quantum=4, bond_dim=4, seed=42)
        >>> # Creates: Q--Q--Q--Q with bond indices connecting neighbors
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n_quantum + n_classical

    # Determine which positions are quantum vs classical
    # Quantum tensors are placed first, then classical
    is_quantum = [True] * n_quantum + [False] * n_classical
    np.random.shuffle(is_quantum)

    # Build indices for each tensor
    # Each tensor has: left_bond, right_bond, physical
    tensor_inds: list[list[tuple[str, int]]] = []

    for i in range(n_total):
        inds = []

        # Physical index
        inds.append((f"p{i}", phys_dim))

        # Left bond (except first tensor in open boundary)
        if i > 0 or not open_boundaries:
            left_idx = (i - 1) % n_total
            inds.append((f"b{left_idx}_{i}", bond_dim))

        # Right bond (except last tensor in open boundary)
        if i < n_total - 1 or not open_boundaries:
            right_idx = (i + 1) % n_total
            bond_name = f"b{i}_{right_idx}" if i < right_idx else f"b{right_idx}_{i}"
            # Avoid duplicate for periodic
            if not any(name == bond_name for name, _ in inds):
                inds.append((bond_name, bond_dim))

        tensor_inds.append(inds)

    # Create tensors
    qtensors = []
    ctensors = []

    for i in range(n_total):
        if is_quantum[i]:
            circuit = _create_qnn_circuit(
                num_qubits=qubits,
                num_layers=layers,
                output_inds=tensor_inds[i],
                seed=None,
            )
            qtensors.append(QuantumTensor(circuit))
        else:
            inds = tuple(name for name, _ in tensor_inds[i])
            shape = tuple(size for _, size in tensor_inds[i])
            data = np.random.randn(*shape).astype(np.float64)
            ctensors.append(CTensor(data, inds))

    # Output indices are all physical indices
    output_inds = tuple(f"p{i}" for i in range(n_total))

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=output_inds,
    )


def rand_hybrid_tree(
    depth: int,
    branching: int = 2,
    n_quantum_ratio: float = 0.5,
    bond_dim: int = 4,
    phys_dim: int = 2,
    qubits: int = 2,
    layers: int = 2,
    seed: int | None = None,
) -> HEinsum:
    """Generate a hybrid tree tensor network.

    Creates a tree structure where nodes are either quantum or classical
    tensors. Each node connects to its parent and children via bond indices.

    Args:
        depth: Depth of the tree (root is depth 0).
        branching: Number of children per node.
        n_quantum_ratio: Fraction of tensors that should be quantum (0 to 1).
        bond_dim: Dimension of bond indices.
        phys_dim: Dimension of physical (leaf) indices.
        qubits: Number of qubits per quantum circuit.
        layers: Number of QNN layers.
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A tree-structured hybrid tensor network.

    Example:
        >>> heinsum = rand_hybrid_tree(depth=3, branching=2, seed=42)
        >>> # Creates a binary tree with depth 3
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate total nodes in tree
    n_total = sum(branching**d for d in range(depth + 1))
    n_quantum = int(n_total * n_quantum_ratio)

    # Assign quantum vs classical randomly
    is_quantum = [True] * n_quantum + [False] * (n_total - n_quantum)
    np.random.shuffle(is_quantum)

    # Build tree structure using BFS ordering
    # Node i has children at indices: branching*i + 1, branching*i + 2, ..., branching*i + branching
    tensor_inds: list[list[tuple[str, int]]] = []

    for i in range(n_total):
        inds = []

        # Parent bond (except root)
        if i > 0:
            parent = (i - 1) // branching
            inds.append((f"b{parent}_{i}", bond_dim))

        # Child bonds
        for c in range(branching):
            child = branching * i + c + 1
            if child < n_total:
                inds.append((f"b{i}_{child}", bond_dim))

        # Physical index for leaves (no children)
        first_child = branching * i + 1
        if first_child >= n_total:
            inds.append((f"p{i}", phys_dim))

        tensor_inds.append(inds)

    # Create tensors
    qtensors = []
    ctensors = []
    output_inds = []

    for i in range(n_total):
        # Check if leaf
        first_child = branching * i + 1
        if first_child >= n_total:
            output_inds.append(f"p{i}")

        if is_quantum[i]:
            circuit = _create_qnn_circuit(
                num_qubits=qubits,
                num_layers=layers,
                output_inds=tensor_inds[i],
                seed=None,
            )
            qtensors.append(QuantumTensor(circuit))
        else:
            inds = tuple(name for name, _ in tensor_inds[i])
            shape = tuple(size for _, size in tensor_inds[i])
            if shape:
                data = np.random.randn(*shape).astype(np.float64)
            else:
                data = np.array(np.random.randn(), dtype=np.float64)
            ctensors.append(CTensor(data, inds))

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=tuple(output_inds),
    )


def rand_hybrid_grid(
    rows: int,
    cols: int,
    n_quantum_ratio: float = 0.5,
    bond_dim: int = 4,
    phys_dim: int = 2,
    qubits: int = 2,
    layers: int = 2,
    seed: int | None = None,
) -> HEinsum:
    """Generate a hybrid 2D grid (PEPS-like) tensor network.

    Creates a 2D grid where each tensor connects to its neighbors
    (up, down, left, right) and has a physical index.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        n_quantum_ratio: Fraction of tensors that should be quantum (0 to 1).
        bond_dim: Dimension of bond indices.
        phys_dim: Dimension of physical indices.
        qubits: Number of qubits per quantum circuit.
        layers: Number of QNN layers.
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A 2D grid-structured hybrid tensor network.

    Example:
        >>> heinsum = rand_hybrid_grid(rows=3, cols=3, seed=42)
        >>> # Creates a 3x3 grid of tensors
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = rows * cols
    n_quantum = int(n_total * n_quantum_ratio)

    # Assign quantum vs classical randomly
    is_quantum = [True] * n_quantum + [False] * (n_total - n_quantum)
    np.random.shuffle(is_quantum)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    # Build indices for each tensor
    tensor_inds: list[list[tuple[str, int]]] = []

    for r in range(rows):
        for c in range(cols):
            inds = []

            # Physical index
            inds.append((f"p{r}_{c}", phys_dim))

            # Right bond
            if c < cols - 1:
                inds.append((f"h{r}_{c}", bond_dim))

            # Left bond
            if c > 0:
                inds.append((f"h{r}_{c-1}", bond_dim))

            # Down bond
            if r < rows - 1:
                inds.append((f"v{r}_{c}", bond_dim))

            # Up bond
            if r > 0:
                inds.append((f"v{r-1}_{c}", bond_dim))

            tensor_inds.append(inds)

    # Create tensors
    qtensors = []
    ctensors = []

    for i in range(n_total):
        if is_quantum[i]:
            circuit = _create_qnn_circuit(
                num_qubits=qubits,
                num_layers=layers,
                output_inds=tensor_inds[i],
                seed=None,
            )
            qtensors.append(QuantumTensor(circuit))
        else:
            inds = tuple(name for name, _ in tensor_inds[i])
            shape = tuple(size for _, size in tensor_inds[i])
            data = np.random.randn(*shape).astype(np.float64)
            ctensors.append(CTensor(data, inds))

    # Output indices are all physical indices
    output_inds = tuple(f"p{r}_{c}" for r in range(rows) for c in range(cols))

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=output_inds,
    )


def rand_hybrid_star(
    n_arms: int,
    arm_length: int = 2,
    n_quantum_ratio: float = 0.5,
    bond_dim: int = 4,
    phys_dim: int = 2,
    qubits: int = 2,
    layers: int = 2,
    seed: int | None = None,
) -> HEinsum:
    """Generate a hybrid star tensor network.

    Creates a star topology with a central tensor connected to multiple
    arms (chains) radiating outward.

    Args:
        n_arms: Number of arms radiating from center.
        arm_length: Number of tensors in each arm (excluding center).
        n_quantum_ratio: Fraction of tensors that should be quantum (0 to 1).
        bond_dim: Dimension of bond indices.
        phys_dim: Dimension of physical indices.
        qubits: Number of qubits per quantum circuit.
        layers: Number of QNN layers.
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A star-structured hybrid tensor network.

    Example:
        >>> heinsum = rand_hybrid_star(n_arms=4, arm_length=3, seed=42)
        >>> # Creates a star with 4 arms, each 3 tensors long
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = 1 + n_arms * arm_length  # Center + arms
    n_quantum = int(n_total * n_quantum_ratio)

    # Assign quantum vs classical randomly
    is_quantum = [True] * n_quantum + [False] * (n_total - n_quantum)
    np.random.shuffle(is_quantum)

    # Build indices
    # Index 0 is center, then arms are consecutive
    tensor_inds: list[list[tuple[str, int]]] = []

    # Center tensor
    center_inds = [(f"p_center", phys_dim)]
    for arm in range(n_arms):
        center_inds.append((f"b_center_{arm}_0", bond_dim))
    tensor_inds.append(center_inds)

    # Arm tensors
    for arm in range(n_arms):
        for pos in range(arm_length):
            inds = []

            # Physical index at end of arm
            if pos == arm_length - 1:
                inds.append((f"p_{arm}_{pos}", phys_dim))

            # Bond to previous (center or previous in arm)
            if pos == 0:
                inds.append((f"b_center_{arm}_0", bond_dim))
            else:
                inds.append((f"b_{arm}_{pos-1}_{pos}", bond_dim))

            # Bond to next in arm
            if pos < arm_length - 1:
                inds.append((f"b_{arm}_{pos}_{pos+1}", bond_dim))

            tensor_inds.append(inds)

    # Create tensors
    qtensors = []
    ctensors = []

    for i in range(n_total):
        if is_quantum[i]:
            circuit = _create_qnn_circuit(
                num_qubits=qubits,
                num_layers=layers,
                output_inds=tensor_inds[i],
                seed=None,
            )
            qtensors.append(QuantumTensor(circuit))
        else:
            inds = tuple(name for name, _ in tensor_inds[i])
            shape = tuple(size for _, size in tensor_inds[i])
            data = np.random.randn(*shape).astype(np.float64)
            ctensors.append(CTensor(data, inds))

    # Output indices: center + arm endpoints
    output_inds = [f"p_center"]
    for arm in range(n_arms):
        output_inds.append(f"p_{arm}_{arm_length - 1}")

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=tuple(output_inds),
    )


def rand_bounded_heinsum(
    n_quantum: int,
    n_classical: int,
    reg: int = 3,
    n_out: int = 0,
    d_min: int = 2,
    d_max: int = 4,
    max_tensor_size: int = 1000,
    seed: int | None = None,
) -> HEinsum:
    """Generate a random hybrid tensor network with bounded tensor sizes.

    Creates a HEinsum where quantum tensors are constrained to have at most
    `max_tensor_size` total elements (product of dimensions). This ensures
    that quantum circuit sampling remains tractable.

    The algorithm assigns indices to tensors while respecting the size bound
    for quantum tensors. Classical tensors have no size restriction.

    Args:
        n_quantum: Number of quantum tensors.
        n_classical: Number of classical tensors.
        reg: Average number of indices per tensor.
        n_out: Number of output (uncontracted) indices.
        d_min: Minimum dimension size for indices.
        d_max: Maximum dimension size for indices.
        max_tensor_size: Maximum total elements per quantum tensor (default: 1000).
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A random hybrid tensor network with bounded quantum tensor sizes.

    Example:
        >>> heinsum = rand_bounded_heinsum(n_quantum=3, n_classical=2, seed=42)
        >>> # All quantum tensors have <= 1000 elements
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n_quantum + n_classical
    n_inds = max(1, n_total * reg // 2)

    # Generate index names and sizes
    ind_names = [f"i{i}" for i in range(n_inds)]
    ind_sizes = {name: np.random.randint(d_min, d_max + 1) for name in ind_names}

    # Output indices (appear once) vs contraction indices (appear twice)
    output_ind_names = ind_names[:n_out]
    contract_ind_names = ind_names[n_out:]

    # Track tensor indices and current sizes
    # tensor_inds[i] = list of (ind_name, size)
    tensor_inds: list[list[tuple[str, int]]] = [[] for _ in range(n_total)]
    tensor_sizes: list[int] = [1] * n_total  # Current product of dimensions

    def can_add_to_tensor(t: int, dim: int) -> bool:
        """Check if adding dimension to tensor keeps it under size limit."""
        # Only quantum tensors (indices 0 to n_quantum-1) have size limits
        if t >= n_quantum:
            return True
        return tensor_sizes[t] * dim <= max_tensor_size

    def add_to_tensor(t: int, ind: str, dim: int) -> None:
        """Add index to tensor and update size tracking."""
        tensor_inds[t].append((ind, dim))
        tensor_sizes[t] *= dim

    # Assign output indices (each to one random tensor that can accept it)
    for ind in output_ind_names:
        dim = ind_sizes[ind]
        # Find tensors that can accept this index
        candidates = [t for t in range(n_total) if can_add_to_tensor(t, dim)]
        if not candidates:
            # If no quantum tensor can accept, assign to a classical tensor
            candidates = list(range(n_quantum, n_total))
        if candidates:
            t = np.random.choice(candidates)
            add_to_tensor(t, ind, dim)

    # Assign contraction indices (each to exactly 2 tensors)
    for ind in contract_ind_names:
        dim = ind_sizes[ind]

        # Find pairs of tensors that can both accept this index
        candidates = [t for t in range(n_total) if can_add_to_tensor(t, dim)]

        if len(candidates) >= 2:
            tensors = np.random.choice(candidates, size=2, replace=False)
        elif len(candidates) == 1:
            # One tensor can accept, find another (prefer classical)
            t1 = candidates[0]
            classical_opts = [t for t in range(n_quantum, n_total) if t != t1]
            if classical_opts:
                t2 = np.random.choice(classical_opts)
            else:
                # Force assign to a quantum tensor (may exceed limit slightly)
                other_quantum = [t for t in range(n_quantum) if t != t1]
                t2 = np.random.choice(other_quantum) if other_quantum else t1
            tensors = [t1, t2]
        else:
            # No tensor can accept within limit, assign to classical tensors
            classical_tensors = list(range(n_quantum, n_total))
            if len(classical_tensors) >= 2:
                tensors = np.random.choice(classical_tensors, size=2, replace=False)
            else:
                # Fallback: assign to any two different tensors
                tensors = np.random.choice(n_total, size=2, replace=False)

        for t in tensors:
            add_to_tensor(t, ind, dim)

    # Ensure each tensor has at least one index
    for t in range(n_total):
        if len(tensor_inds[t]) == 0:
            # Add a small dummy contracted index
            dummy_ind = f"dummy_{t}"
            dummy_dim = d_min
            ind_sizes[dummy_ind] = dummy_dim

            # Find another tensor to share this index
            other_candidates = [
                o for o in range(n_total) if o != t and can_add_to_tensor(o, dummy_dim)
            ]
            if not other_candidates:
                other_candidates = [o for o in range(n_total) if o != t]

            other_t = np.random.choice(other_candidates)

            add_to_tensor(t, dummy_ind, dummy_dim)
            add_to_tensor(other_t, dummy_ind, dummy_dim)

    # Create quantum tensors (first n_quantum)
    # For quantum tensors, we create simple circuits with ISwitches for each index
    qtensors = []
    for t in range(n_quantum):
        inds = tuple(name for name, _ in tensor_inds[t])
        shape = tuple(size for _, size in tensor_inds[t])

        # Create a minimal quantum tensor with the required shape
        qtensor = QuantumTensor.from_shape(shape, inds)
        qtensors.append(qtensor)

    # Create classical tensors (remaining n_classical)
    ctensors = []
    for t in range(n_quantum, n_total):
        inds = tuple(name for name, _ in tensor_inds[t])
        shape = tuple(size for _, size in tensor_inds[t])
        if shape:
            data = np.random.randn(*shape).astype(np.float64)
        else:
            data = np.array(np.random.randn(), dtype=np.float64)
        ctensors.append(CTensor(data, inds))

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=tuple(output_ind_names),
    )


def rand_regular_heinsum(
    n_quantum: int,
    n_classical: int,
    reg: int = 3,
    q_bond_dim: int = 2,
    c_bond_dim: int = 8,
    seed: int | None = None,
) -> HEinsum:
    """Generate a random n-regular hybrid tensor network.

    Creates an n-regular random graph where each tensor has exactly `reg` bonds.
    Bonds between quantum tensors use small dimensions (q_bond_dim), while
    bonds involving only classical tensors use larger dimensions (c_bond_dim).
    This allows scaling classical contraction complexity without affecting
    quantum tensor sizes.

    Args:
        n_quantum: Number of quantum tensors.
        n_classical: Number of classical tensors.
        reg: Number of bonds per tensor (regularity). Must satisfy
            n_total * reg being even.
        q_bond_dim: Bond dimension for edges touching quantum tensors.
        c_bond_dim: Bond dimension for edges between classical tensors only.
        seed: Random seed for reproducibility.

    Returns:
        HEinsum: A random n-regular hybrid tensor network.

    Example:
        >>> # 5 quantum + 10 classical tensors, 4-regular, small q-bonds, large c-bonds
        >>> h = rand_regular_heinsum(5, 10, reg=4, q_bond_dim=2, c_bond_dim=32)
        >>> # Quantum tensors stay small: 2^4 = 16 elements max
        >>> # Classical contraction cost scales with c_bond_dim
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n_quantum + n_classical

    if (n_total * reg) % 2 != 0:
        raise ValueError(
            f"n_total * reg must be even for regular graph. "
            f"Got {n_total} * {reg} = {n_total * reg}"
        )

    # Generate random regular graph using configuration model
    # Each tensor has `reg` "half-edges" (stubs)
    stubs = []
    for t in range(n_total):
        stubs.extend([t] * reg)

    # Shuffle and pair up stubs to form edges
    np.random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        t1, t2 = stubs[i], stubs[i + 1]
        edges.append((min(t1, t2), max(t1, t2)))

    # Remove self-loops and multi-edges by resampling if needed
    # (Simple rejection: just keep valid edges)
    edge_set = set()
    valid_edges = []
    for t1, t2 in edges:
        if t1 != t2 and (t1, t2) not in edge_set:
            edge_set.add((t1, t2))
            valid_edges.append((t1, t2))

    # Build tensor indices from edges
    tensor_inds: list[list[tuple[str, int]]] = [[] for _ in range(n_total)]

    for idx, (t1, t2) in enumerate(valid_edges):
        ind_name = f"e{idx}"

        # Determine bond dimension based on whether quantum tensors are involved
        if t1 < n_quantum or t2 < n_quantum:
            # At least one quantum tensor - use small bond dim
            dim = q_bond_dim
        else:
            # Both classical - use large bond dim
            dim = c_bond_dim

        tensor_inds[t1].append((ind_name, dim))
        tensor_inds[t2].append((ind_name, dim))

    # Create quantum tensors
    qtensors = []
    for t in range(n_quantum):
        inds = tuple(name for name, _ in tensor_inds[t])
        shape = tuple(size for _, size in tensor_inds[t])

        if not shape:
            # Tensor with no edges - give it a trivial shape
            inds = (f"trivial_q{t}",)
            shape = (1,)

        qtensor = QuantumTensor.from_shape(shape, inds)
        qtensors.append(qtensor)

    # Create classical tensors
    ctensors = []
    for t in range(n_quantum, n_total):
        inds = tuple(name for name, _ in tensor_inds[t])
        shape = tuple(size for _, size in tensor_inds[t])

        if not shape:
            # Tensor with no edges - give it a trivial shape
            inds = (f"trivial_c{t}",)
            shape = (1,)
            data = np.array([1.0], dtype=np.float64)
        else:
            data = np.random.randn(*shape).astype(np.float64)

        ctensors.append(CTensor(data, inds))

    return HEinsum(
        qtensors=qtensors,
        ctensors=ctensors,
        input_tensors=[],
        output_inds=(),
    )
