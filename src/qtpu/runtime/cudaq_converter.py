"""Convert QTPU QuantumTensor to CUDA-Q kernel code.

This generates CUDA-Q Python kernel code from a QuantumTensor,
where ISwitches become conditional statements (if/elif chains)
over the parameter indices.

The generated kernel can be:
1. Executed directly via cudaq.sample() or cudaq.observe()
2. Compiled by CUDA-Q's MLIR-based compiler for GPU acceleration
3. Run on NVIDIA quantum hardware simulators

This bridges QTPU's declarative tensor notation with CUDA-Q's
imperative kernel programming model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from qtpu.tensor import QuantumTensor, ISwitch


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
            "rx": "rx",
            "ry": "ry",
            "rz": "rz",
            "cx": "x.ctrl",
            "cz": "z.ctrl",
            "swap": "swap",
        }

        if name in name_map:
            cudaq_name = name_map[name]

            if name in ("cx", "cz"):
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
    qtensor: "QuantumTensor",
    kernel_name: str = "qtpu_kernel",
    param_values: dict[str, float] | None = None,
) -> str:
    """Generate complete CUDA-Q program that computes the full tensor.

    This generates a program that:
    1. Defines the kernel with if/elif for ISwitches (handles arbitrary sub-circuits)
    2. Computes Z⊗Z⊗...⊗Z expectation for all parameter combinations
    3. Returns a numpy array matching qtensor.shape

    The generated code uses cudaq.observe() with spin operators to compute
    expectation values, matching QTPU's behavior.

    Args:
        qtensor: The QuantumTensor to convert.
        kernel_name: Name for the generated kernel function.
        param_values: Values for free parameters (non-ISwitch). If None,
            free parameters become kernel arguments that can be passed
            via command line. Keys can use original param names (e.g., 'theta[0]')
            or sanitized names (e.g., 'theta_0').

    Returns:
        String containing complete executable CUDA-Q Python code.
    """
    from qiskit.circuit import Parameter
    from qtpu.tensor import ISwitch

    circuit = qtensor.circuit
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
    iswitch_params = {}  # param_name -> size (ordered)
    iswitch_instances = []  # Keep track of ISwitches for sub-circuit scanning
    iswitch_qubit_mapping = {}  # iswitch_id -> list of qubit indices in main circuit
    for instr in circuit:
        if isinstance(instr.operation, ISwitch):
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
        elif isinstance(op, ISwitch):
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
    lines.append("")

    # Collect all qubits that should be in the observable
    observable_qubits = list(measured_qubits)

    # Add ancilla qubits (they hold deferred measurement results)
    if total_ancillas > 0:
        ancilla_qubits = list(range(n_qubits, n_total_qubits))
        observable_qubits.extend(ancilla_qubits)
        lines.append(
            f"# Z⊗Z⊗...⊗Z observable on measured qubits ({measured_qubits}) and ancillas ({ancilla_qubits})"
        )
    else:
        lines.append("# Z⊗Z⊗...⊗Z observable on measured qubits")

    if len(observable_qubits) == 1:
        lines.append(f"hamiltonian = spin.z({observable_qubits[0]})")
    else:
        obs_parts = [f"spin.z({q})" for q in observable_qubits]
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

    lines.append(
        f'    """Compute expectation values for all parameter combinations using async parallelism."""'
    )
    lines.append(f"    shape = {qtensor.shape}")
    lines.append(f"    result = np.zeros(shape, dtype=np.float64)")
    lines.append("")

    # Generate async parallel execution
    iswitch_param_names_sanitized = [
        _sanitize_param_name(name) for name in iswitch_params.keys()
    ]
    
    if len(iswitch_params) == 1:
        # Single ISwitch - use simple async pattern
        param_name = iswitch_param_names_sanitized[0]
        size = list(iswitch_params.values())[0]
        
        all_kernel_args_template = iswitch_param_names_sanitized + free_params_sanitized
        args_template = ", ".join(all_kernel_args_template)
        
        lines.append(f"    # Launch all observe calls asynchronously")
        lines.append(f"    futures = []")
        lines.append(f"    for {param_name} in range({size}):")
        lines.append(f"        futures.append(cudaq.observe_async({kernel_name}, hamiltonian, {args_template}))")
        lines.append("")
        lines.append(f"    # Collect results")
        lines.append(f"    for {param_name}, future in enumerate(futures):")
        lines.append(f"        result[{param_name}] = future.get().expectation()")
    
    elif len(iswitch_params) > 1:
        # Multiple ISwitches - use nested async with flat list
        lines.append(f"    # Launch all observe calls asynchronously")
        lines.append(f"    futures = []")
        lines.append(f"    indices = []")
        
        indent = "    "
        for orig_name, size in iswitch_params.items():
            sanitized = _sanitize_param_name(orig_name)
            lines.append(f"{indent}for {sanitized} in range({size}):")
            indent += "    "
        
        all_kernel_args = iswitch_param_names_sanitized + free_params_sanitized
        args = ", ".join(all_kernel_args)
        idx_tuple = ", ".join(iswitch_param_names_sanitized)
        lines.append(f"{indent}futures.append(cudaq.observe_async({kernel_name}, hamiltonian, {args}))")
        lines.append(f"{indent}indices.append(({idx_tuple}))")
        
        lines.append("")
        lines.append(f"    # Collect results")
        lines.append(f"    for idx, future in zip(indices, futures):")
        lines.append(f"        result[idx] = future.get().expectation()")
    
    else:
        # No ISwitches - single call
        all_kernel_args = free_params_sanitized
        if all_kernel_args:
            args = ", ".join(all_kernel_args)
            lines.append(f"    result[()] = cudaq.observe({kernel_name}, hamiltonian, {args}).expectation()")
        else:
            lines.append(f"    result[()] = cudaq.observe({kernel_name}, hamiltonian).expectation()")

    lines.append("")
    lines.append("    return result")

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

    return "\n".join(lines)


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
    """Compile and execute a QuantumTensor using CUDA-Q.

    Requires CUDA-Q to be installed.

    Args:
        qtensor: The QuantumTensor to evaluate.
        param_values: Values for free parameters (non-ISwitch params).

    Returns:
        numpy array with shape matching qtensor.shape containing Z expectations.
    """
    param_values = param_values or {}
    _, compute_func, free_param_names = compile_cudaq_kernel(qtensor)
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
    qtensor: "QuantumTensor",
) -> tuple[object, callable, list[str]]:
    """Compile a CUDA-Q kernel for a QuantumTensor and cache it.
    
    Uses importlib to load from a temp file (required by cudaq.kernel decorator
    which needs source code access). The module is cached for fast reuse.
    
    Args:
        qtensor: The QuantumTensor to compile.
        
    Returns:
        Tuple of (module, compute_func, free_param_names) where:
        - module: The loaded module containing the CUDA-Q kernel
        - compute_func: The compute_tensor function
        - free_param_names: List of free parameter names (sanitized) that need values
    """
    import tempfile
    import importlib.util
    
    # Use id of the qtensor as cache key
    cache_key = id(qtensor)
    
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key][:3]
    
    # Generate the code without baked-in param values
    code = quantum_tensor_to_cudaq(qtensor, param_values=None)
    
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
    spec = importlib.util.spec_from_file_location(f"cudaq_kernel_{cache_key}", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    compute_func = module.compute_tensor
    
    # Cache including temp path for potential cleanup
    _compiled_kernel_cache[cache_key] = (module, compute_func, free_param_names, temp_path)
    
    return module, compute_func, free_param_names


def get_compiled_kernel(
    qtensor: "QuantumTensor",
) -> tuple[callable, list[str]]:
    """Get the compiled compute_tensor function for a QuantumTensor.
    
    This compiles the kernel if not already cached.
    
    Args:
        qtensor: The QuantumTensor to compile.
        
    Returns:
        Tuple of (compute_func, free_param_names)
    """
    _, compute_func, free_param_names = compile_cudaq_kernel(qtensor)
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
    qtensor: "QuantumTensor",
    param_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Run a compiled CUDA-Q kernel with the given parameters.
    
    First call compiles and JIT-compiles the kernel (slow).
    Subsequent calls reuse the cached kernel (fast).
    
    Args:
        qtensor: The QuantumTensor to evaluate.
        param_values: Values for free parameters.
        
    Returns:
        numpy array with shape matching qtensor.shape.
    """
    _, compute_func, free_param_names = compile_cudaq_kernel(qtensor)
    return _execute_compiled_kernel(compute_func, free_param_names, param_values or {})


def _run_cudaq_importlib(
    qtensor: "QuantumTensor",
    param_values: dict[str, float],
) -> np.ndarray:
    """Execute CUDA-Q kernel in-memory.

    Uses caching for fast repeated evaluations.
    First call includes JIT compilation overhead.
    """
    return run_compiled_kernel(qtensor, param_values)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/home/nate/qtpu/src")

    from qiskit.circuit import QuantumCircuit, Parameter
    from qtpu.tensor import QuantumTensor, ISwitch

    print("=" * 70)
    print("QuantumTensor to CUDA-Q Converter")
    print("=" * 70)

    # Create a simple quantum tensor with ISwitches
    batch_size = 4
    n_qubits = 2

    X = np.array([0.1, 0.5, 0.9, 1.3]) * np.pi
    batch_param = Parameter("batch")

    qc = QuantumCircuit(n_qubits, n_qubits)

    # ISwitch for data encoding
    def make_selector(X):
        def selector(b):
            c = QuantumCircuit(1)
            c.ry(X[b], 0)
            return c

        return selector

    iswitch = ISwitch(batch_param, 1, batch_size, make_selector(X))
    qc.append(iswitch, [0])

    # Fixed entangling gate
    qc.cx(0, 1)

    # Another ISwitch
    X2 = np.array([0.2, 0.4, 0.6, 0.8]) * np.pi
    iswitch2 = ISwitch(batch_param, 1, batch_size, make_selector(X2))
    qc.append(iswitch2, [1])

    qc.measure(range(n_qubits), range(n_qubits))

    qtensor = QuantumTensor(qc)
    print(f"\nQuantumTensor: shape={qtensor.shape}, inds={qtensor.inds}")

    # Show generated code
    print("\n" + "=" * 70)
    print("Generated CUDA-Q code:")
    print("=" * 70)
    code = quantum_tensor_to_cudaq(qtensor)
    print(code)

    # Compile to file
    print("\n" + "=" * 70)
    print("Compile to file:")
    print("=" * 70)
    output_file = "/tmp/qtpu_example_kernel.py"
    compile_quantum_tensor(qtensor, output_file)
    print(f"Compiled to: {output_file}")
    print(f"\nTo run with CUDA-Q:")
    print(f"  python {output_file} -o /tmp/result.npy")
    print(f"\nThen load with:")
    print(f"  tensor = np.load('/tmp/result.npy')")

    # =========================================================================
    # Test with QPD Measures (deferred measurement pattern)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test with QPD Measures (deferred measurement):")
    print("=" * 70)

    from qiskit_addon_cutting.qpd import QPDMeasure

    batch_size = 2
    batch_param2 = Parameter("batch2")

    # Create ISwitch with QPD measures in sub-circuits
    def make_qpd_selector(batch_size):
        def selector(b):
            c = QuantumCircuit(2)
            c.h(0)
            c.cx(0, 1)
            # QPD measure on qubit 0 - will become deferred measurement
            c.append(QPDMeasure(), [0])
            c.rz(b * 0.5, 1)
            return c

        return selector

    qc2 = QuantumCircuit(2, 2)
    qc2.h(0)
    iswitch_qpd = ISwitch(batch_param2, 2, batch_size, make_qpd_selector(batch_size))
    qc2.append(iswitch_qpd, [0, 1])
    qc2.measure(range(2), range(2))

    qtensor2 = QuantumTensor(qc2)
    print(f"\nQuantumTensor with QPD: shape={qtensor2.shape}, inds={qtensor2.inds}")

    print("\n" + "=" * 70)
    print("Generated CUDA-Q code with deferred QPD measures:")
    print("=" * 70)
    code2 = quantum_tensor_to_cudaq(qtensor2)
    print(code2)
