"""Convert quantum circuits with ISwitches to CUDA-Q kernel code.

This generates CUDA-Q Python kernel code from circuits containing ISwitch
instructions, where ISwitches become conditional statements (if/elif chains)
over the parameter indices.

The generated kernel can be:
1. Executed directly via cudaq.sample() or cudaq.observe()
2. Compiled by CUDA-Q's MLIR-based compiler for GPU acceleration
3. Run on NVIDIA quantum hardware simulators

This bridges QTPU's declarative tensor notation with CUDA-Q's
imperative kernel programming model.

Note: This module uses duck-typing for ISwitch detection to avoid
circular imports with tensor.py. An ISwitch is detected by checking
for the 'iswitch' instruction name.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from qiskit.circuit import QuantumCircuit

from qtpu.core.qtensor import QuantumTensor


@dataclass
class CudaQGate:
    """Represents a gate in the generated CUDA-Q code."""

    name: str
    qubits: list[int]
    params: list[float] | None = None
    ctrl_qubits: list[int] | None = None


@dataclass
class QPDMeasureInfo:
    """Represents a deferred QPD measure that needs an ancilla qubit."""

    qubit: int  # The qubit being "measured"


def extract_gate_info(
    circuit, include_qpd_measures: bool = False
) -> tuple[list[CudaQGate], list[QPDMeasureInfo]]:
    """Extract gate information from a Qiskit circuit.

    Args:
        circuit: Qiskit QuantumCircuit
        include_qpd_measures: If True, track QPD measures for deferred handling
            using ancilla qubits (CNOT to ancilla + measure ancilla pattern).
            If False, QPD measures are skipped entirely.

    Returns:
        Tuple of (gates, qpd_measures) where qpd_measures contains info about
        QPD measures that need ancilla qubits for deferred measurement.
        The qpd_measures list is ordered by appearance in the circuit.
    """
    gates = []
    qpd_measures = []

    for instr in circuit:
        op = instr.operation
        qubits = [circuit.qubits.index(q) for q in instr.qubits]

        name = op.name.lower()
        params = list(op.params) if op.params else None

        # Handle QPD measures specially - these become deferred measurements
        # using an ancilla qubit: CNOT(target, ancilla) + measure(ancilla)
        if name == "qpd_measure":
            if include_qpd_measures:
                qpd_measures.append(QPDMeasureInfo(qubit=qubits[0]))
            # Skip adding as a gate - will be handled separately
            continue

        # Skip reset operations - they are handled implicitly in wire cut decomposition
        # by only measuring ancilla qubits, not the main qubit after reset
        if name == "reset":
            continue

        # Map Qiskit gate names to CUDA-Q
        name_map = {
            "h": "h",
            "x": "x",
            "y": "y",
            "z": "z",
            "s": "s",
            "t": "t",
            "sdg": "s.adj",
            "tdg": "t.adj",
            "sx": "rx",  # sqrt(X) = rx(pi/2)
            "sxdg": "rx",  # sqrt(X)^dag = rx(-pi/2)
            "rx": "rx",
            "ry": "ry",
            "rz": "rz",
            "cx": "x.ctrl",
            "cz": "z.ctrl",
            "swap": "swap",
        }

        if name in name_map:
            cudaq_name = name_map[name]

            # Handle sx and sxdg: they become rx(pi/2) and rx(-pi/2)
            if name == "sx":
                import math
                gates.append(
                    CudaQGate(
                        name="rx",
                        qubits=qubits,
                        params=[math.pi / 2],
                    )
                )
            elif name == "sxdg":
                import math
                gates.append(
                    CudaQGate(
                        name="rx",
                        qubits=qubits,
                        params=[-math.pi / 2],
                    )
                )
            elif name in ("cx", "cz"):
                # Control gate
                gates.append(
                    CudaQGate(
                        name=cudaq_name,
                        qubits=[qubits[1]],  # target
                        ctrl_qubits=[qubits[0]],  # control
                        params=params,
                    )
                )
            else:
                gates.append(
                    CudaQGate(
                        name=cudaq_name,
                        qubits=qubits,
                        params=params,
                    )
                )

    return gates, qpd_measures


def gates_to_cudaq_code(
    gates: list[CudaQGate],
    qubit_var: str = "q",
    param_formatter: callable = None,
) -> list[str]:
    """Convert gate list to CUDA-Q code lines.

    Args:
        gates: List of CudaQGate objects.
        qubit_var: Variable name for qubits.
        param_formatter: Optional function to format parameter values.
            If None, uses str(). Should handle both numeric and Parameter types.
    """
    if param_formatter is None:
        param_formatter = str

    lines = []

    for gate in gates:
        qubits_str = ", ".join(f"{qubit_var}[{q}]" for q in gate.qubits)

        if gate.ctrl_qubits:
            ctrl_str = ", ".join(f"{qubit_var}[{q}]" for q in gate.ctrl_qubits)
            if gate.params:
                params_str = ", ".join(param_formatter(p) for p in gate.params)
                lines.append(f"{gate.name}({params_str}, {ctrl_str}, {qubits_str})")
            else:
                lines.append(f"{gate.name}({ctrl_str}, {qubits_str})")
        else:
            if gate.params:
                params_str = ", ".join(param_formatter(p) for p in gate.params)
                lines.append(f"{gate.name}({params_str}, {qubits_str})")
            else:
                lines.append(f"{gate.name}({qubits_str})")

    return lines


def qpd_measures_to_cudaq_code(
    qpd_measures: list[QPDMeasureInfo],
    qubit_var: str = "q",
    ancilla_start_idx: int = 0,
    qubit_remap: dict[int, int] | None = None,
) -> list[str]:
    """Generate CUDA-Q code for deferred QPD measurements.

    Deferred measurement pattern: Instead of measuring the target qubit directly,
    we CNOT the target to an ancilla qubit and include the ancilla in the observable.
    This preserves the target qubit's state for subsequent operations while still
    capturing the measurement outcome in the expectation value.

    The pattern is:
        1. Allocate ancilla qubit (initialized to |0⟩)
        2. CNOT(target, ancilla) - copies Z-basis information to ancilla
        3. Include ancilla in Z⊗Z⊗...⊗Z observable

    This is equivalent to measuring the target qubit in the Z basis, but deferred
    to the end of the circuit via the principle of deferred measurement.

    Args:
        qpd_measures: List of QPDMeasureInfo objects describing qubits to measure.
        qubit_var: Variable name for qubits.
        ancilla_start_idx: Starting index for ancilla qubits in the qubit register.
        qubit_remap: Optional mapping from sub-circuit qubit indices to main circuit indices.
            If provided, the qubit indices in qpd_measures will be remapped.

    Returns:
        List of CUDA-Q code lines implementing deferred measurements via CNOT to ancilla.
    """
    lines = []

    for i, qpd in enumerate(qpd_measures):
        target_qubit = qpd.qubit
        if qubit_remap is not None:
            target_qubit = qubit_remap.get(target_qubit, target_qubit)

        ancilla_qubit = ancilla_start_idx + i

        # CNOT from target to ancilla implements deferred measurement
        # The ancilla captures Z-basis information and is included in the observable
        lines.append(
            f"x.ctrl({qubit_var}[{target_qubit}], {qubit_var}[{ancilla_qubit}])  # Deferred QPD measure"
        )

    return lines


def _sanitize_param_name(name: str) -> str:
    """Convert a parameter name to a valid Python identifier.

    E.g., 'theta[0]' -> 'theta_0', 'phi-1' -> 'phi_1'
    """
    import re

    # Replace brackets and other invalid chars with underscores
    sanitized = re.sub(r"[\[\]\-\.\s]", "_", name)
    # Remove any remaining invalid characters
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "", sanitized)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def quantum_tensor_to_cudaq(
    circuit: QuantumCircuit,
    shape: tuple[int, ...],
    kernel_name: str = "qtpu_kernel",
    param_values: dict[str, float] | None = None,
) -> str:
    """Generate complete CUDA-Q program that computes the full tensor.

    The key speedup over SimulatorBackend: the kernel is compiled ONCE by CUDA-Q's
    JIT compiler, then executed N times with different integer parameters. This
    avoids regenerating circuit objects for each batch element.

    This generates a program that:
    1. Defines the kernel with if statements for ISwitches (handles arbitrary sub-circuits)
    2. Computes Z⊗Z⊗...⊗Z expectation for all parameter combinations
    3. Returns a numpy array matching the given shape

    The generated code uses cudaq.observe() with spin operators to compute
    expectation values, matching QTPU's behavior.

    Args:
        circuit: The QuantumCircuit containing ISwitch instructions.
        shape: The output tensor shape (from QuantumTensor.shape).
        kernel_name: Name for the generated kernel function.
        param_values: Values for free parameters (non-ISwitch). If None,
            free parameters become kernel arguments that can be passed
            via command line. Keys can use original param names (e.g., 'theta[0]')
            or sanitized names (e.g., 'theta_0').

    Returns:
        String containing complete executable CUDA-Q Python code.
    """
    from qiskit.circuit import Parameter

    n_qubits = circuit.num_qubits
    param_values = param_values or {}

    # Normalize param_values keys to original names
    # (accept both 'theta[0]' and 'theta_0')
    normalized_param_values = {}
    for k, v in param_values.items():
        normalized_param_values[k] = v
        # Also map sanitized version back
        normalized_param_values[_sanitize_param_name(k)] = v

    # Find measured qubits
    measured_qubits = []
    for instr in circuit:
        if instr.operation.name == "measure":
            q = circuit.qubits.index(instr.qubits[0])
            if q not in measured_qubits:
                measured_qubits.append(q)

    if not measured_qubits:
        measured_qubits = list(range(n_qubits))

    # Collect ISwitch parameter info (these become int args)
    # Use duck-typing: check for 'iswitch' instruction name
    def _is_iswitch(op) -> bool:
        return op.name == "iswitch"

    iswitch_params = {}  # param_name -> size (ordered)
    iswitch_instances = []  # Keep track of ISwitches for sub-circuit scanning
    iswitch_qubit_mapping = {}  # iswitch_id -> list of qubit indices in main circuit
    for instr in circuit:
        if _is_iswitch(instr.operation):
            iswitch = instr.operation
            iswitch_instances.append(iswitch)
            qubits = [circuit.qubits.index(q) for q in instr.qubits]
            iswitch_qubit_mapping[id(iswitch)] = qubits
            name = iswitch.param.name
            if name not in iswitch_params:
                iswitch_params[name] = iswitch.size

    # Scan all ISwitch sub-circuits to find max QPD measures needed
    # We need to allocate enough ancilla qubits for the worst case
    max_qpd_measures_per_iswitch = {}  # iswitch_id -> max count across variants
    for iswitch in iswitch_instances:
        max_count = 0
        for i in range(iswitch.size):
            sub_circuit = iswitch._selector(i)
            count = sum(
                1
                for instr in sub_circuit
                if instr.operation.name.lower() == "qpd_measure"
            )
            max_count = max(max_count, count)
        max_qpd_measures_per_iswitch[id(iswitch)] = max_count

    total_ancillas = sum(max_qpd_measures_per_iswitch.values())
    n_total_qubits = n_qubits + total_ancillas

    # Build ancilla offset map: iswitch_id -> starting ancilla index
    ancilla_offset = {}
    offset = n_qubits
    for iswitch in iswitch_instances:
        ancilla_offset[id(iswitch)] = offset
        offset += max_qpd_measures_per_iswitch[id(iswitch)]

    # Track which main qubits should be traced out (I observable)
    # A qubit should use I if it has an ISwitch where ALL variants end with reset
    # This is the wire cut prep side scenario
    traced_out_qubits = set()
    for iswitch in iswitch_instances:
        qubits = iswitch_qubit_mapping[id(iswitch)]
        # Check if ALL variants end with reset
        all_end_with_reset = True
        for i in range(iswitch.size):
            sub_circuit = iswitch._selector(i)
            if sub_circuit.data:
                last_op = sub_circuit.data[-1].operation.name.lower()
                if last_op != "reset":
                    all_end_with_reset = False
                    break
            else:
                all_end_with_reset = False
                break
        if all_end_with_reset:
            traced_out_qubits.update(qubits)

    # Find ALL parameters including those in ISwitch sub-circuits
    all_param_names = {p.name for p in circuit.parameters}

    # Also scan ISwitch sub-circuits for parameters
    for iswitch in iswitch_instances:
        for i in range(iswitch.size):
            sub_circuit = iswitch._selector(i)
            for p in sub_circuit.parameters:
                all_param_names.add(p.name)

    # Free parameters are those not used as ISwitch selectors
    free_param_names_orig = sorted(all_param_names - set(iswitch_params.keys()))

    # Build mapping from original name to sanitized name
    orig_to_sanitized = {
        name: _sanitize_param_name(name) for name in free_param_names_orig
    }

    # Check which free params have values provided (check both original and sanitized names)
    free_params_with_values = {}  # orig_name -> value
    free_params_as_args = []  # list of orig_names that need to be args
    for orig_name in free_param_names_orig:
        sanitized = orig_to_sanitized[orig_name]
        if orig_name in normalized_param_values:
            free_params_with_values[orig_name] = normalized_param_values[orig_name]
        elif sanitized in normalized_param_values:
            free_params_with_values[orig_name] = normalized_param_values[sanitized]
        else:
            free_params_as_args.append(orig_name)

    lines = [
        "#!/usr/bin/env python3",
        '"""CUDA-Q kernel generated from QTPU QuantumTensor.',
        "",
        "This program computes Z expectation values for all parameter combinations,",
        "returning a tensor matching the original QuantumTensor shape.",
        '"""',
        "",
        "import cudaq",
        "import numpy as np",
        "from cudaq import spin",
        "",
    ]

    # Generate kernel arguments: ISwitches as int, free params as float (use sanitized names)
    kernel_args = []
    for name in iswitch_params.keys():
        kernel_args.append(f"{_sanitize_param_name(name)}: int")
    for orig_name in free_params_as_args:
        sanitized = orig_to_sanitized[orig_name]
        kernel_args.append(f"{sanitized}: float")

    param_args = ", ".join(kernel_args)
    lines.append("@cudaq.kernel")
    lines.append(f"def {kernel_name}({param_args}):")
    lines.append(f'    """Quantum kernel with parameterized gates."""')
    lines.append(f"    q = cudaq.qvector({n_total_qubits})")
    if total_ancillas > 0:
        lines.append(
            f"    # Qubits 0-{n_qubits-1}: main circuit, {n_qubits}-{n_total_qubits-1}: ancillas for deferred QPD measures"
        )
    lines.append("")

    def format_param(p):
        """Format a parameter - either as sanitized variable name or literal value."""
        if hasattr(p, "name"):  # It's a Parameter or ParameterVectorElement
            orig_name = p.name
            if orig_name in free_params_with_values:
                return str(free_params_with_values[orig_name])
            # Return sanitized name for use as variable
            return _sanitize_param_name(orig_name)
        return str(p)

    # Process instructions
    for instr in circuit:
        op = instr.operation
        qubits = [circuit.qubits.index(q) for q in instr.qubits]

        if op.name == "measure":
            continue
        elif op.name == "barrier":
            continue
        elif op.name.lower() == "qpd_measure":
            # QPD measures in the main circuit (outside ISwitches) are skipped
            # They should be handled via deferred measurement if needed
            continue
        elif _is_iswitch(op):
            # Generate independent if statements for ISwitch
            # (Using if instead of elif avoids AST recursion depth issues)
            iswitch = op
            param_name = iswitch.param.name
            iswitch_ancilla_start = ancilla_offset[id(iswitch)]

            lines.append(f"    # ISwitch on {param_name} (size={iswitch.size})")

            for i in range(iswitch.size):
                sub_circuit = iswitch._selector(i)
                # Extract gates AND QPD measures from sub-circuit
                gates, qpd_measures = extract_gate_info(
                    sub_circuit, include_qpd_measures=True
                )
                gate_lines = gates_to_cudaq_code(
                    gates, "q", param_formatter=format_param
                )

                # Build qubit remap for this ISwitch: sub-circuit idx -> main circuit idx
                qubit_remap = {
                    sub_q: qubits[sub_q] for sub_q in range(iswitch.num_qubits)
                }

                # Remap qubit indices from sub-circuit to main circuit
                remapped_lines = []
                for line in gate_lines:
                    for sub_q in range(iswitch.num_qubits):
                        line = line.replace(f"q[{sub_q}]", f"q[{qubits[sub_q]}]")
                    remapped_lines.append(line)

                # Generate deferred measurement code for QPD measures
                qpd_lines = qpd_measures_to_cudaq_code(
                    qpd_measures,
                    qubit_var="q",
                    ancilla_start_idx=iswitch_ancilla_start,
                    qubit_remap=qubit_remap,
                )

                lines.append(f"    if {_sanitize_param_name(param_name)} == {i}:")

                if remapped_lines or qpd_lines:
                    for gate_line in remapped_lines:
                        lines.append(f"        {gate_line}")
                    for qpd_line in qpd_lines:
                        lines.append(f"        {qpd_line}")
                else:
                    lines.append("        pass  # Identity")

            lines.append("")
        elif op.name.lower() == "cx":
            lines.append(f"    x.ctrl(q[{qubits[0]}], q[{qubits[1]}])")
        elif op.name.lower() == "cz":
            lines.append(f"    z.ctrl(q[{qubits[0]}], q[{qubits[1]}])")
        elif op.name.lower() in ("h", "x", "y", "z", "s", "t"):
            lines.append(f"    {op.name.lower()}(q[{qubits[0]}])")
        elif op.params:
            params_str = ", ".join(format_param(p) for p in op.params)
            lines.append(f"    {op.name.lower()}({params_str}, q[{qubits[0]}])")

    # Build Z observable for measured qubits AND ancilla qubits
    # Ancilla qubits hold the deferred QPD measurement outcomes
    # Main qubits that are traced out (wire cut prep side) use I observable
    lines.append("")

    # Build observable: Z for measured/active qubits, I for traced out qubits
    observable_qubits = []
    
    # Add main qubits that are NOT traced out
    for q in measured_qubits:
        if q not in traced_out_qubits:
            observable_qubits.append(q)

    # Add ancilla qubits (they hold deferred measurement results from QPD measures)
    if total_ancillas > 0:
        ancilla_qubits = list(range(n_qubits, n_total_qubits))
        observable_qubits.extend(ancilla_qubits)
        
        if traced_out_qubits:
            lines.append(
                f"# Z observable on active qubits + ancillas, I on traced-out qubits {sorted(traced_out_qubits)}"
            )
        else:
            lines.append(
                f"# Z⊗Z⊗...⊗Z observable on measured qubits ({measured_qubits}) and ancillas ({ancilla_qubits})"
            )
    else:
        if traced_out_qubits:
            lines.append(
                f"# Z observable on active qubits, I on traced-out qubits {sorted(traced_out_qubits)}"
            )
        else:
            lines.append("# Z⊗Z⊗...⊗Z observable on measured qubits")

    # Build observable: Z for active qubits, I for traced-out qubits
    # All qubits must be in the observable for CudaQ
    all_obs_qubits = sorted(set(observable_qubits) | traced_out_qubits)
    
    if len(all_obs_qubits) == 1:
        q = all_obs_qubits[0]
        if q in traced_out_qubits:
            lines.append(f"hamiltonian = spin.i({q})")
        else:
            lines.append(f"hamiltonian = spin.z({q})")
    else:
        obs_parts = []
        for q in all_obs_qubits:
            if q in traced_out_qubits:
                obs_parts.append(f"spin.i({q})")
            else:
                obs_parts.append(f"spin.z({q})")
        lines.append(f"hamiltonian = {' * '.join(obs_parts)}")

    lines.append("")

    # Generate compute function
    lines.append("")

    # Get sanitized names for free params that are args
    free_params_sanitized = [orig_to_sanitized[name] for name in free_params_as_args]

    # Function signature includes free params as args (sanitized)
    if free_params_sanitized:
        func_args = ", ".join(f"{name}: float" for name in free_params_sanitized)
        lines.append(f"def compute_tensor({func_args}) -> np.ndarray:")
    else:
        lines.append(f"def compute_tensor() -> np.ndarray:")

    # Use CUDA-Q broadcasting - pass list of parameter values, get all results in one call
    iswitch_param_names_sanitized = [
        _sanitize_param_name(name) for name in iswitch_params.keys()
    ]

    lines.append(f'    """Compute expectation values using CUDA-Q broadcasting."""')
    lines.append(f"    shape = {shape}")
    lines.append("")

    if len(iswitch_params) == 1:
        # Single ISwitch - use broadcasting with list of values
        param_name = iswitch_param_names_sanitized[0]
        size = list(iswitch_params.values())[0]

        lines.append(f"    # Build parameter arrays for broadcasting")
        lines.append(f"    {param_name}_values = list(range({size}))")
        lines.append("")

        # Build args list for observe call
        all_kernel_args = [f"{param_name}_values"] + free_params_sanitized
        args = ", ".join(all_kernel_args)

        lines.append(
            f"    # Use CUDA-Q broadcasting - single call evaluates all batch elements"
        )
        lines.append(f"    results = cudaq.observe({kernel_name}, hamiltonian, {args})")
        lines.append("")
        lines.append(f"    # Extract expectation values")
        lines.append(
            f"    result = np.array([r.expectation() for r in results], dtype=np.float64)"
        )
        lines.append(f"    return result.reshape(shape)")

    elif len(iswitch_params) > 1:
        # Multiple ISwitches - use itertools.product for all combinations
        lines.append(f"    import itertools")
        lines.append("")
        lines.append(f"    # Build parameter arrays for broadcasting")

        range_vars = []
        for orig_name, size in iswitch_params.items():
            sanitized = _sanitize_param_name(orig_name)
            lines.append(f"    {sanitized}_range = list(range({size}))")
            range_vars.append(f"{sanitized}_range")

        lines.append("")
        lines.append(f"    # Generate all parameter combinations")
        ranges_str = ", ".join(range_vars)
        lines.append(f"    all_combos = list(itertools.product({ranges_str}))")
        lines.append("")

        # Create separate lists for each parameter
        for i, orig_name in enumerate(iswitch_params.keys()):
            sanitized = _sanitize_param_name(orig_name)
            lines.append(f"    {sanitized}_values = [c[{i}] for c in all_combos]")

        lines.append("")

        # Build args list for observe call
        param_value_vars = [
            f"{_sanitize_param_name(name)}_values" for name in iswitch_params.keys()
        ]
        all_kernel_args = param_value_vars + free_params_sanitized
        args = ", ".join(all_kernel_args)

        lines.append(
            f"    # Use CUDA-Q broadcasting - single call evaluates all combinations"
        )
        lines.append(f"    results = cudaq.observe({kernel_name}, hamiltonian, {args})")
        lines.append("")
        lines.append(f"    # Extract expectation values and reshape to tensor")
        lines.append(
            f"    result = np.array([r.expectation() for r in results], dtype=np.float64)"
        )
        lines.append(f"    return result.reshape(shape)")

    else:
        # No ISwitches - single call
        lines.append(f"    result = np.zeros(shape, dtype=np.float64)")
        all_kernel_args = free_params_sanitized
        if all_kernel_args:
            args = ", ".join(all_kernel_args)
            lines.append(
                f"    result[()] = cudaq.observe({kernel_name}, hamiltonian, {args}).expectation()"
            )
        else:
            lines.append(
                f"    result[()] = cudaq.observe({kernel_name}, hamiltonian).expectation()"
            )
        lines.append(f"    return result")

    # Generate sample_tensor function for sampling from index space
    lines.append("")
    lines.append("")

    # Function signature: required params first, then optional params
    # This avoids "non-default argument follows default argument" syntax error
    if free_params_sanitized:
        func_args = ", ".join(f"{name}: float" for name in free_params_sanitized)
        lines.append(f"def sample_tensor({func_args}, num_samples: int | None = None, indices: list[tuple[int, ...]] | None = None) -> list[tuple[tuple[int, ...], float]]:")
    else:
        lines.append(f"def sample_tensor(num_samples: int | None = None, indices: list[tuple[int, ...]] | None = None) -> list[tuple[tuple[int, ...], float]]:")

    lines.append(f'    """Sample from the index space and compute expectation values.')
    lines.append(f'    ')
    lines.append(f'    Instead of computing the full tensor, this samples indices from the index space')
    lines.append(f'    and only evaluates those using CUDA-Q broadcasting.')
    lines.append(f'    ')
    lines.append(f'    Either provide num_samples (for random sampling) or indices (for explicit indices).')
    lines.append(f'    ')
    lines.append(f'    Args:')
    lines.append(f'        num_samples: Number of random indices to sample. Mutually exclusive with indices.')
    lines.append(f'        indices: Explicit list of index tuples to evaluate. Mutually exclusive with num_samples.')
    if free_params_sanitized:
        for name in free_params_sanitized:
            lines.append(f'        {name}: Value for parameter {name}.')
    lines.append(f'    ')
    lines.append(f'    Returns:')
    lines.append(f'        List of (index_tuple, expectation_value) pairs.')
    lines.append(f'    """')
    lines.append(f"    shape = {shape}")
    lines.append("")
    lines.append(f"    # Validate arguments")
    lines.append(f"    if num_samples is None and indices is None:")
    lines.append(f"        raise ValueError('Must provide either num_samples or indices')")
    lines.append(f"    if num_samples is not None and indices is not None:")
    lines.append(f"        raise ValueError('Cannot provide both num_samples and indices')")
    lines.append("")

    if len(iswitch_params) == 0:
        # No ISwitches - just return single value
        lines.append(f"    # No ISwitch parameters - single value")
        if free_params_sanitized:
            args = ", ".join(free_params_sanitized)
            lines.append(f"    val = cudaq.observe({kernel_name}, hamiltonian, {args}).expectation()")
        else:
            lines.append(f"    val = cudaq.observe({kernel_name}, hamiltonian).expectation()")
        lines.append(f"    n = num_samples if num_samples is not None else len(indices)")
        lines.append(f"    return [((), val)] * n")
    elif len(iswitch_params) == 1:
        # Single ISwitch - sample from one dimension
        param_name = iswitch_param_names_sanitized[0]
        size = list(iswitch_params.values())[0]

        lines.append(f"    # Get indices to evaluate")
        lines.append(f"    if indices is not None:")
        lines.append(f"        sampled_indices = [idx[0] if isinstance(idx, tuple) else idx for idx in indices]")
        lines.append(f"    else:")
        lines.append(f"        sampled_indices = [np.random.randint(0, {size}) for _ in range(num_samples)]")
        lines.append("")
        lines.append(f"    # Use CUDA-Q broadcasting for all indices")
        
        all_kernel_args = ["sampled_indices"] + free_params_sanitized
        args = ", ".join(all_kernel_args)
        lines.append(f"    results = cudaq.observe({kernel_name}, hamiltonian, {args})")
        lines.append("")
        lines.append(f"    # Build result list of (index_tuple, value) pairs")
        lines.append(f"    return [((idx,), r.expectation()) for idx, r in zip(sampled_indices, results)]")
    else:
        # Multiple ISwitches - sample from multi-dimensional space
        lines.append(f"    # Get indices to evaluate")
        lines.append(f"    if indices is not None:")
        lines.append(f"        sampled_indices = [tuple(idx) for idx in indices]")
        lines.append(f"    else:")
        lines.append(f"        sampled_indices = []")
        lines.append(f"        for _ in range(num_samples):")
        
        # Build tuple of random indices
        rand_parts = []
        for orig_name, size in iswitch_params.items():
            rand_parts.append(f"np.random.randint(0, {size})")
        lines.append(f"            sampled_indices.append(({', '.join(rand_parts)},))")
        lines.append("")
        
        # Create separate lists for each parameter dimension
        for i, orig_name in enumerate(iswitch_params.keys()):
            sanitized = _sanitize_param_name(orig_name)
            lines.append(f"    {sanitized}_values = [idx[{i}] for idx in sampled_indices]")
        lines.append("")
        
        # Build args for observe call
        param_value_vars = [
            f"{_sanitize_param_name(name)}_values" for name in iswitch_params.keys()
        ]
        all_kernel_args = param_value_vars + free_params_sanitized
        args = ", ".join(all_kernel_args)
        
        lines.append(f"    # Use CUDA-Q broadcasting for all indices")
        lines.append(f"    results = cudaq.observe({kernel_name}, hamiltonian, {args})")
        lines.append("")
        lines.append(f"    # Build result list of (index_tuple, value) pairs")
        lines.append(f"    return [(idx, r.expectation()) for idx, r in zip(sampled_indices, results)]")

    # Generate warmup_jit function - calls kernel once with minimal input to trigger JIT
    lines.append("")
    lines.append("")
    
    # Build warmup function signature (same as compute_tensor)
    if free_params_sanitized:
        func_args = ", ".join(f"{name}: float = 0.0" for name in free_params_sanitized)
        lines.append(f"def warmup_jit({func_args}) -> None:")
    else:
        lines.append(f"def warmup_jit() -> None:")
    
    lines.append(f'    """Trigger CUDA-Q JIT compilation with a single kernel call.')
    lines.append(f'    ')
    lines.append(f'    This runs the kernel once with minimal input (index 0) to force')
    lines.append(f'    JIT compilation. Subsequent calls to compute_tensor will be fast.')
    lines.append(f'    """')
    
    # Generate single kernel call with index 0 for each ISwitch parameter
    if iswitch_param_names_sanitized:
        # Call with first index for each ISwitch param
        warmup_args = ["0"] * len(iswitch_param_names_sanitized) + free_params_sanitized
        warmup_args_str = ", ".join(warmup_args)
        lines.append(f"    _ = cudaq.observe({kernel_name}, hamiltonian, {warmup_args_str}).expectation()")
    else:
        # No ISwitches - just call once
        if free_params_sanitized:
            warmup_args_str = ", ".join(free_params_sanitized)
            lines.append(f"    _ = cudaq.observe({kernel_name}, hamiltonian, {warmup_args_str}).expectation()")
        else:
            lines.append(f"    _ = cudaq.observe({kernel_name}, hamiltonian).expectation()")

    # Main block with argparse for free params
    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    import argparse")
    lines.append("    parser = argparse.ArgumentParser()")
    lines.append("    parser.add_argument('-o', '--output', type=str, required=True,")
    lines.append(
        "                        help='Output path for the numpy array (.npy)')"
    )

    # Add arguments for free parameters (use sanitized names for CLI)
    for orig_name in free_params_as_args:
        sanitized = orig_to_sanitized[orig_name]
        lines.append(
            f"    parser.add_argument('--{sanitized}', type=float, required=True,"
        )
        lines.append(f"                        help='Value for parameter {orig_name}')")

    lines.append("    args = parser.parse_args()")
    lines.append("")
    lines.append("    print('Computing quantum tensor...')")

    # Call compute_tensor with free param args (sanitized)
    if free_params_sanitized:
        call_args = ", ".join(f"args.{name}" for name in free_params_sanitized)
        lines.append(f"    tensor = compute_tensor({call_args})")
    else:
        lines.append("    tensor = compute_tensor()")

    lines.append(f"    print(f'Result shape: {{tensor.shape}}')")
    lines.append("    np.save(args.output, tensor)")
    lines.append("    print(f'Saved to: {args.output}')")

    code = "\n".join(lines)
    return code, len(lines)


# Alias for backward compatibility
quantum_tensor_to_cudaq_full = quantum_tensor_to_cudaq


def compile_quantum_tensor(
    qtensor: "QuantumTensor",
    output_path: str,
    kernel_name: str = "qtpu_kernel",
    param_values: dict[str, float] | None = None,
) -> str:
    """Compile a QuantumTensor to a CUDA-Q Python file.

    Args:
        qtensor: The QuantumTensor to compile.
        output_path: Path where to write the generated Python file.
        kernel_name: Name for the generated kernel function.
        param_values: Values for free parameters. If provided, these will
            be baked into the generated code. Otherwise, they become
            command-line arguments.

    Returns:
        The path to the generated file.
    """
    code = quantum_tensor_to_cudaq(
        qtensor, kernel_name=kernel_name, param_values=param_values
    )

    with open(output_path, "w") as f:
        f.write(code)

    return output_path


def run_quantum_tensor_cudaq(
    qtensor: "QuantumTensor",
    param_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Compile and execute a circuit with ISwitches using CUDA-Q.

    Requires CUDA-Q to be installed.

    Args:
        qtensor: The QuantumTensor to compile and execute.
        param_values: Values for free parameters (non-ISwitch params).

    Returns:
        numpy array with the given shape containing Z expectations.
    """
    param_values = param_values or {}
    _, compute_func, free_param_names = compile_cudaq_kernel(
        qtensor.circuit, qtensor.shape
    )
    return _execute_compiled_kernel(compute_func, free_param_names, param_values)


def _execute_compiled_kernel(
    compute_func: callable,
    free_param_names: list[str],
    param_values: dict[str, float],
) -> np.ndarray:
    """Execute a pre-compiled compute_tensor function with parameters."""
    kwargs = {}
    for name in free_param_names:
        if name in param_values:
            kwargs[name] = param_values[name]
        else:
            # Try original name (with brackets etc)
            for orig_name, val in param_values.items():
                if _sanitize_param_name(orig_name) == name:
                    kwargs[name] = val
                    break
            else:
                raise ValueError(f"Missing parameter value for '{name}'")

    if kwargs:
        return compute_func(**kwargs)
    else:
        return compute_func()


# Module-level cache for compiled kernels
# Maps qtensor id -> (module, compute_func, free_param_names, temp_file_path)
_compiled_kernel_cache: dict[int, tuple] = {}
_temp_files: list[str] = []  # Track temp files for cleanup


def compile_cudaq_kernel(
    circuit: QuantumCircuit,
    shape: tuple[int, ...],
) -> tuple[object, callable, list[str], int]:
    """Compile a CUDA-Q kernel for a circuit with ISwitches and cache it.

    Uses importlib to load from a temp file (required by cudaq.kernel decorator
    which needs source code access). The module is cached for fast reuse.

    Args:
        circuit: The QuantumCircuit containing ISwitch instructions.
        shape: The output tensor shape.

    Returns:
        Tuple of (module, compute_func, free_param_names, num_code_lines) where:
        - module: The loaded module containing the CUDA-Q kernel
            (also has sample_tensor and warmup_jit functions)
        - compute_func: The compute_tensor function
        - free_param_names: List of free parameter names (sanitized) that need values
        - num_code_lines: Number of lines in the generated code
    """
    import tempfile
    import importlib.util

    # Use id of the circuit as cache key
    cache_key = id(circuit)

    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key][:4]

    # Generate the code with a unique kernel name to avoid CUDA-Q global registry conflicts
    kernel_name = f"qtpu_kernel_{cache_key}"
    code, num_code_lines = quantum_tensor_to_cudaq(circuit, shape, kernel_name=kernel_name, param_values=None)

    # Remove the main block (everything after if __name__)
    lines = code.split("\n")
    main_idx = next(i for i, l in enumerate(lines) if "if __name__" in l)
    code = "\n".join(lines[:main_idx])

    # Extract free parameter names from the compute_tensor signature
    free_param_names = []
    for line in lines:
        if line.startswith("def compute_tensor("):
            sig = line.split("(")[1].split(")")[0]
            if sig.strip():
                for param in sig.split(","):
                    param_name = param.split(":")[0].strip()
                    if param_name:
                        free_param_names.append(param_name)
            break

    # Write to temp file (CUDA-Q needs source code access)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    _temp_files.append(temp_path)

    # Load the module
    spec = importlib.util.spec_from_file_location(
        f"cudaq_kernel_{cache_key}", temp_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    compute_func = module.compute_tensor

    # Cache including temp path for potential cleanup
    _compiled_kernel_cache[cache_key] = (
        module,
        compute_func,
        free_param_names,
        num_code_lines,
        temp_path,
    )

    return module, compute_func, free_param_names, num_code_lines


def get_compiled_kernel(
    circuit: QuantumCircuit,
    shape: tuple[int, ...],
) -> tuple[callable, list[str]]:
    """Get the compiled compute_tensor function for a circuit.

    This compiles the kernel if not already cached.

    Args:
        circuit: The QuantumCircuit containing ISwitch instructions.
        shape: The output tensor shape.

    Returns:
        Tuple of (compute_func, free_param_names)
    """
    _, compute_func, free_param_names = compile_cudaq_kernel(circuit, shape)
    return compute_func, free_param_names


def clear_kernel_cache():
    """Clear the compiled kernel cache and remove temp files."""
    import os

    _compiled_kernel_cache.clear()

    for path in _temp_files:
        try:
            os.unlink(path)
        except Exception:
            pass
    _temp_files.clear()


def run_compiled_kernel(
    circuit: QuantumCircuit,
    shape: tuple[int, ...],
    param_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Run a compiled CUDA-Q kernel with the given parameters.

    First call compiles and JIT-compiles the kernel (slow).
    Subsequent calls reuse the cached kernel (fast).

    Args:
        circuit: The QuantumCircuit containing ISwitch instructions.
        shape: The output tensor shape.
        param_values: Values for free parameters.

    Returns:
        numpy array with the given shape.
    """
    _, compute_func, free_param_names = compile_cudaq_kernel(circuit, shape)
    return _execute_compiled_kernel(compute_func, free_param_names, param_values or {})


def execute_compiled_kernel(
    compute_func: callable,
    free_param_names: list[str],
    param_values: dict[str, float],
) -> np.ndarray:
    """Execute a pre-compiled compute_tensor function with parameters.

    This is the public version of _execute_compiled_kernel.

    Args:
        compute_func: The compiled compute_tensor function.
        free_param_names: List of parameter names the function expects.
        param_values: Dict mapping parameter names to values.

    Returns:
        numpy array result.
    """
    return _execute_compiled_kernel(compute_func, free_param_names, param_values)


# =============================================================================
# CompiledQuantumTensor
# =============================================================================


class CompiledQuantumTensor:
    """A compiled quantum tensor for fast repeated evaluation.

    This class wraps a QuantumTensor with a compiled CUDA-Q backend
    for efficient repeated execution. The kernel is JIT-compiled on first
    use and cached for subsequent calls.

    The compiled tensor is callable and returns a numpy array matching
    the original tensor's shape.

    Attributes:
        qtensor: The original QuantumTensor.
        shape: Shape of the output tensor.
        inds: Index names of the output tensor.
        backend: The compilation backend used.

    Example:
        >>> from qtpu.core import QuantumTensor
        >>> compiled = qtensor.compile("cudaq")
        >>> result = compiled()  # Returns np.ndarray with shape qtensor.shape
        >>> result = compiled(theta=0.5)  # Pass rotation parameters
    """

    def __init__(self, qtensor: "QuantumTensor", warmup: bool = True):
        """Initialize a compiled quantum tensor.

        Args:
            qtensor: The QuantumTensor to compile.
            warmup: If True, run a warmup execution to trigger JIT compilation
                during initialization. This moves the JIT overhead to compile
                time rather than first execution time.
        """
        from qtpu.core.qtensor import QuantumTensor
        
        self._qtensor = qtensor
        self._compiled_fn: callable | None = None
        self._sample_fn: callable | None = None
        self._warmup_fn: callable | None = None
        self._free_param_names: list[str] = []
        self._jit_warmup_done: bool = False
        self._num_code_lines: int = 0
        
        # Compile immediately
        self._ensure_compiled()
        
        # Optionally run warmup to trigger CUDA-Q JIT compilation
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Run a warmup execution to trigger CUDA-Q JIT compilation.
        
        This calls the kernel once with index 0 (not full broadcast).
        The result is discarded - this just warms up the JIT.
        """
        if self._jit_warmup_done:
            return
        
        if self._warmup_fn is None:
            return
        
        try:
            if self._free_param_names:
                # Provide dummy values for free parameters
                kwargs = {name: 0.0 for name in self._free_param_names}
                self._warmup_fn(**kwargs)
            else:
                self._warmup_fn()
            self._jit_warmup_done = True
        except Exception:
            # If warmup fails, we'll just pay JIT cost on first real call
            pass

    @property
    def qtensor(self) -> QuantumTensor:
        """The original QuantumTensor."""
        return self._qtensor

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the output tensor."""
        return self._qtensor.shape

    @property
    def inds(self) -> tuple[str, ...]:
        """Index names of the output tensor."""
        return self._qtensor.inds

    @property
    def is_compiled(self) -> bool:
        """Whether the kernel has been compiled."""
        return self._compiled_fn is not None
    
    @property
    def is_jit_warmed(self) -> bool:
        """Whether JIT warmup has been done."""
        return self._jit_warmup_done

    @property
    def num_code_lines(self) -> int:
        """Number of lines in the generated CUDA-Q code."""
        return self._num_code_lines

    def _ensure_compiled(self) -> None:
        """Compile the kernel if not already compiled."""
        if self._compiled_fn is not None:
            return

        module, self._compiled_fn, self._free_param_names, self._num_code_lines = compile_cudaq_kernel(
            self._qtensor.circuit, self._qtensor.shape
        )
        
        # Get sample_tensor and warmup_jit functions if available
        self._sample_fn = getattr(module, 'sample_tensor', None)
        self._warmup_fn = getattr(module, 'warmup_jit', None)

    def __call__(self, **params: float) -> np.ndarray:
        """Evaluate the compiled quantum tensor.

        Args:
            **params: Values for free parameters (rotation angles, etc.).
                ISwitch parameters are handled internally.

        Returns:
            np.ndarray: Result tensor with shape matching self.shape.

        Example:
            >>> result = compiled()  # No free parameters
            >>> result = compiled(theta=0.5, phi=1.2)  # With parameters
        """
        self._ensure_compiled()

        # Build kwargs for the compiled function
        kwargs = {}
        for name in self._free_param_names:
            if name in params:
                kwargs[name] = params[name]
            else:
                # Try to find parameter with original name (e.g., 'theta[0]' vs 'theta_0')
                for orig_name, val in params.items():
                    if _sanitize_param_name(orig_name) == name:
                        kwargs[name] = val
                        break
                else:
                    if name not in kwargs:
                        raise ValueError(
                            f"Missing parameter: '{name}'. "
                            f"Required: {self._free_param_names}"
                        )

        if kwargs:
            return self._compiled_fn(**kwargs)
        else:
            return self._compiled_fn()

    def execute(self, **params: float) -> np.ndarray:
        """Execute the compiled quantum tensor (alias for __call__).

        Args:
            **params: Values for free parameters (rotation angles, etc.).

        Returns:
            np.ndarray: Result tensor with shape matching self.shape.
        """
        return self(**params)

    def sample(
        self,
        num_samples: int | None = None,
        indices: list[tuple[int, ...]] | None = None,
        **params: float,
    ) -> list[tuple[tuple[int, ...], float]]:
        """Sample from the index space and compute expectation values.

        Instead of computing the full tensor (which requires evaluating all
        index combinations), this samples indices from the index space
        and only evaluates those using CUDA-Q broadcasting. This is useful
        for large tensors where computing all elements is expensive.

        Either provide num_samples (for random sampling) or indices (for explicit
        index tuples to evaluate).

        Args:
            num_samples: Number of random index assignments to sample.
                Mutually exclusive with indices.
            indices: Explicit list of index tuples to evaluate.
                Mutually exclusive with num_samples.
                E.g., [(0, 1), (2, 3), (1, 0)] for a 2D tensor.
            **params: Values for free parameters (rotation angles, etc.).

        Returns:
            List of (index_tuple, expectation_value) pairs, where:
            - index_tuple: The index assignment (e.g., (0, 2, 1) for
              a 3-dimensional tensor)
            - expectation_value: The computed Z⊗Z⊗...⊗Z expectation for
              that index assignment

        Example:
            >>> compiled = qtensor.compile("cudaq")
            >>> # Sample 100 random index assignments
            >>> samples = compiled.sample(num_samples=100)
            >>> for idx, val in samples[:5]:
            ...     print(f"Index {idx}: {val:.4f}")
            Index (0,): 0.7071
            Index (2,): -0.5000
            Index (1,): 0.0000
            ...
            >>> # Evaluate specific indices
            >>> samples = compiled.sample(indices=[(0,), (1,), (2,), (3,)])
            >>> # With free parameters
            >>> samples = compiled.sample(num_samples=50, theta=0.5, phi=1.2)
        """
        self._ensure_compiled()

        if self._sample_fn is None:
            raise RuntimeError(
                "Sample function not available. This may happen if the kernel "
                "was compiled with an older version of the code generator."
            )

        # Build kwargs for the sample function
        kwargs = {}
        for name in self._free_param_names:
            if name in params:
                kwargs[name] = params[name]
            else:
                # Try to find parameter with original name (e.g., 'theta[0]' vs 'theta_0')
                for orig_name, val in params.items():
                    if _sanitize_param_name(orig_name) == name:
                        kwargs[name] = val
                        break
                else:
                    if name not in kwargs:
                        raise ValueError(
                            f"Missing parameter: '{name}'. "
                            f"Required: {self._free_param_names}"
                        )

        return self._sample_fn(num_samples=num_samples, indices=indices, **kwargs)

    def clear_cache(self) -> None:
        """Clear the compiled kernel cache, forcing recompilation on next call."""
        self._compiled_fn = None
        self._sample_fn = None
        self._warmup_fn = None
        self._free_param_names = []
        self._jit_warmup_done = False

    def __repr__(self) -> str:
        status = "compiled" if self.is_compiled else "not compiled"
        return f"CompiledQuantumTensor(shape={self.shape}, {status})"
