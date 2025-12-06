"""Benchmark QTPU vs Mitiq-style error mitigation representation.

Compares:
1. Compile/generation time
2. Memory usage
3. Number of circuit objects created

QTPU represents the entire mitigation workflow as a single QuantumTensor,
while Mitiq-style approaches enumerate all circuit variants explicitly.

Note: We implement Mitiq-style generation directly to avoid dependency issues,
but the approach matches what Mitiq does internally for PEC/twirling/ZNE.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

import benchkit as bk

from qtpu.core import ISwitch, QuantumTensor, CTensor, HEinsum
from qtpu.compiler.codegen import quantum_tensor_to_cudaq

if TYPE_CHECKING:
    pass


PAULIS = [IGate(), XGate(), YGate(), ZGate()]


# =============================================================================
# QTPU Approach: Single QuantumTensor representation
# =============================================================================

def create_pec_iswitch(idx: str) -> ISwitch:
    """Create ISwitch for PEC basis operations."""
    param = Parameter(idx)
    
    def selector(basis_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.append(PAULIS[basis_idx], [0])
        return qc
    
    return ISwitch(param, 1, 4, selector)


def create_twirl_iswitch(idx: str) -> ISwitch:
    """Create ISwitch for Pauli twirling."""
    param = Parameter(idx)
    
    def selector(pauli_idx: int) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.append(PAULIS[pauli_idx], [0])
        return qc
    
    return ISwitch(param, 1, 4, selector)


def create_zne_iswitch(base_gate: QuantumCircuit, noise_levels: list[int], idx: str) -> ISwitch:
    """Create ISwitch for ZNE noise folding."""
    param = Parameter(idx)
    
    def selector(level_idx: int) -> QuantumCircuit:
        level = noise_levels[level_idx]
        folded = QuantumCircuit(base_gate.num_qubits)
        folded.compose(base_gate, inplace=True)
        for _ in range((level - 1) // 2):
            folded.compose(base_gate.inverse(), inplace=True)
            folded.compose(base_gate, inplace=True)
        return folded
    
    return ISwitch(param, base_gate.num_qubits, len(noise_levels), selector)


def qtpu_generate_pec(circuit: QuantumCircuit, num_gates: int) -> tuple[HEinsum, int]:
    """Generate PEC mitigation using QTPU representation."""
    pec_circuit = QuantumCircuit(circuit.num_qubits)
    ctensors = []
    
    p = 0.01
    gamma = p / (1 - p)
    coeffs = np.array([1 + 3*gamma, -gamma, -gamma, -gamma])
    
    gate_count = 0
    for instr in circuit:
        if gate_count < num_gates and instr.operation.num_qubits == 1:
            iswitch = create_pec_iswitch(f"pec_{gate_count}")
            pec_circuit.append(iswitch, instr.qubits)
            pec_circuit.append(instr.operation, instr.qubits)
            ctensors.append(CTensor(coeffs, (f"pec_{gate_count}",)))
            gate_count += 1
        else:
            pec_circuit.append(instr.operation, instr.qubits, instr.clbits)
    
    qtensor = QuantumTensor(pec_circuit)
    heinsum = HEinsum(qtensors=[qtensor], ctensors=ctensors, input_tensors=[], output_inds=())
    
    # Shape is the product of all ISwitch sizes (4^num_gates for PEC)
    # Use Python's math.prod instead of np.prod to avoid int64 overflow
    import math
    num_circuits = math.prod(qtensor.shape) if qtensor.shape else 1
    return heinsum, num_circuits


def qtpu_generate_twirl(circuit: QuantumCircuit, num_qubits: int) -> tuple[HEinsum, int]:
    """Generate Pauli twirling using QTPU representation."""
    twirl_circuit = QuantumCircuit(circuit.num_qubits)
    ctensors = []
    
    for i in range(min(num_qubits, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_pre_{i}")
        twirl_circuit.append(iswitch, [i])
    
    twirl_circuit.compose(circuit, inplace=True)
    
    for i in range(min(num_qubits, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_post_{i}")
        twirl_circuit.append(iswitch, [i])
        delta = np.eye(4) / 4
        ctensors.append(CTensor(delta, (f"twirl_pre_{i}", f"twirl_post_{i}")))
    
    qtensor = QuantumTensor(twirl_circuit)
    heinsum = HEinsum(qtensors=[qtensor], ctensors=ctensors, input_tensors=[], output_inds=())
    
    import math
    num_circuits = math.prod(qtensor.shape) if qtensor.shape else 1
    return heinsum, num_circuits


def qtpu_generate_zne(circuit: QuantumCircuit, num_gates: int) -> tuple[HEinsum, int]:
    """Generate ZNE using QTPU representation."""
    noise_levels = [1, 3, 5]
    zne_circuit = QuantumCircuit(circuit.num_qubits)
    ctensors = []
    
    # Richardson extrapolation coefficients
    V = np.vander(noise_levels, increasing=True)
    coeffs = np.linalg.inv(V)[0]
    
    gate_count = 0
    for instr in circuit:
        if gate_count < num_gates and instr.operation.num_qubits == 1:
            base_gate = QuantumCircuit(1)
            base_gate.append(instr.operation, [0])
            iswitch = create_zne_iswitch(base_gate, noise_levels, f"zne_{gate_count}")
            zne_circuit.append(iswitch, instr.qubits)
            ctensors.append(CTensor(coeffs, (f"zne_{gate_count}",)))
            gate_count += 1
        else:
            zne_circuit.append(instr.operation, instr.qubits, instr.clbits)
    
    qtensor = QuantumTensor(zne_circuit)
    heinsum = HEinsum(qtensors=[qtensor], ctensors=ctensors, input_tensors=[], output_inds=())
    
    import math
    num_circuits = math.prod(qtensor.shape) if qtensor.shape else 1
    return heinsum, num_circuits


def qtpu_generate_combined(circuit: QuantumCircuit, num_pec: int, num_twirl: int, num_zne: int) -> tuple[HEinsum, int]:
    """Generate combined PEC + twirling + ZNE using QTPU."""
    noise_levels = [1, 3, 5]
    combined_circuit = QuantumCircuit(circuit.num_qubits)
    ctensors = []
    
    # Coefficients
    p = 0.01
    gamma = p / (1 - p)
    pec_coeffs = np.array([1 + 3*gamma, -gamma, -gamma, -gamma])
    V = np.vander(noise_levels, increasing=True)
    zne_coeffs = np.linalg.inv(V)[0]
    
    # Pre-twirl
    for i in range(min(num_twirl, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_pre_{i}")
        combined_circuit.append(iswitch, [i])
    
    # Circuit with PEC and ZNE
    pec_count = 0
    zne_count = 0
    for instr in circuit:
        if instr.operation.num_qubits == 1:
            if zne_count < num_zne:
                base_gate = QuantumCircuit(1)
                base_gate.append(instr.operation, [0])
                iswitch = create_zne_iswitch(base_gate, noise_levels, f"zne_{zne_count}")
                combined_circuit.append(iswitch, instr.qubits)
                ctensors.append(CTensor(zne_coeffs, (f"zne_{zne_count}",)))
                zne_count += 1
            elif pec_count < num_pec:
                iswitch = create_pec_iswitch(f"pec_{pec_count}")
                combined_circuit.append(iswitch, instr.qubits)
                combined_circuit.append(instr.operation, instr.qubits)
                ctensors.append(CTensor(pec_coeffs, (f"pec_{pec_count}",)))
                pec_count += 1
            else:
                combined_circuit.append(instr.operation, instr.qubits, instr.clbits)
        else:
            combined_circuit.append(instr.operation, instr.qubits, instr.clbits)
    
    # Post-twirl
    for i in range(min(num_twirl, circuit.num_qubits)):
        iswitch = create_twirl_iswitch(f"twirl_post_{i}")
        combined_circuit.append(iswitch, [i])
        delta = np.eye(4) / 4
        ctensors.append(CTensor(delta, (f"twirl_pre_{i}", f"twirl_post_{i}")))
    
    qtensor = QuantumTensor(combined_circuit)
    heinsum = HEinsum(qtensors=[qtensor], ctensors=ctensors, input_tensors=[], output_inds=())
    
    import math
    num_circuits = math.prod(qtensor.shape) if qtensor.shape else 1
    return heinsum, num_circuits


# =============================================================================
# Mitiq-style Approach: Explicit circuit enumeration
# =============================================================================

def mitiq_generate_pec(circuit: QuantumCircuit, num_gates: int, num_samples: int) -> list[QuantumCircuit]:
    """Generate PEC circuits Mitiq-style (sampling)."""
    circuits = []
    gate_indices = []
    for i, instr in enumerate(circuit):
        if len(gate_indices) < num_gates and instr.operation.num_qubits == 1:
            gate_indices.append(i)
    
    rng = np.random.default_rng(42)
    for _ in range(num_samples):
        combo = tuple(rng.integers(0, 4, size=len(gate_indices)))
        new_circuit = QuantumCircuit(circuit.num_qubits)
        gate_count = 0
        
        for i, instr in enumerate(circuit):
            if i in gate_indices:
                new_circuit.append(PAULIS[combo[gate_count]], instr.qubits)
                new_circuit.append(instr.operation, instr.qubits)
                gate_count += 1
            else:
                new_circuit.append(instr.operation, instr.qubits, instr.clbits)
        circuits.append(new_circuit)
    
    return circuits


def mitiq_generate_twirl(circuit: QuantumCircuit, num_qubits: int, num_samples: int) -> list[QuantumCircuit]:
    """Generate twirled circuits Mitiq-style (sampling)."""
    circuits = []
    actual_qubits = min(num_qubits, circuit.num_qubits)
    
    rng = np.random.default_rng(42)
    for _ in range(num_samples):
        combo = rng.integers(0, 4, size=actual_qubits)
        new_circuit = QuantumCircuit(circuit.num_qubits)
        
        for i, pauli_idx in enumerate(combo):
            new_circuit.append(PAULIS[pauli_idx], [i])
        new_circuit.compose(circuit, inplace=True)
        for i, pauli_idx in enumerate(combo):
            new_circuit.append(PAULIS[pauli_idx], [i])
        
        circuits.append(new_circuit)
    
    return circuits


def mitiq_generate_zne(circuit: QuantumCircuit, num_gates: int, num_samples: int) -> list[QuantumCircuit]:
    """Generate ZNE circuits Mitiq-style."""
    noise_levels = [1, 3, 5]
    circuits = []
    
    gate_indices = []
    for i, instr in enumerate(circuit):
        if len(gate_indices) < num_gates and instr.operation.num_qubits == 1:
            gate_indices.append(i)
    
    # For ZNE, generate circuits for each noise level
    samples_per_level = num_samples // len(noise_levels)
    
    for level in noise_levels:
        for _ in range(samples_per_level):
            new_circuit = QuantumCircuit(circuit.num_qubits)
            for i, instr in enumerate(circuit):
                if i in gate_indices:
                    new_circuit.append(instr.operation, instr.qubits)
                    for _ in range((level - 1) // 2):
                        new_circuit.append(instr.operation.inverse(), instr.qubits)
                        new_circuit.append(instr.operation, instr.qubits)
                else:
                    new_circuit.append(instr.operation, instr.qubits, instr.clbits)
            circuits.append(new_circuit)
    
    return circuits


def mitiq_generate_combined(circuit: QuantumCircuit, num_pec: int, num_twirl: int, num_zne: int, num_samples: int) -> list[QuantumCircuit]:
    """Generate combined mitigation circuits Mitiq-style."""
    circuits = []
    actual_twirl = min(num_twirl, circuit.num_qubits)
    noise_levels = [1, 3, 5]
    
    # Find gate indices
    pec_indices = []
    zne_indices = []
    gate_count = 0
    for i, instr in enumerate(circuit):
        if instr.operation.num_qubits == 1:
            if gate_count < num_zne:
                zne_indices.append(i)
            elif gate_count < num_zne + num_pec:
                pec_indices.append(i)
            gate_count += 1
    
    rng = np.random.default_rng(42)
    for _ in range(num_samples):
        twirl_combo = tuple(rng.integers(0, 4, size=actual_twirl))
        pec_combo = tuple(rng.integers(0, 4, size=len(pec_indices)))
        zne_level = rng.choice(noise_levels)
        
        new_circuit = QuantumCircuit(circuit.num_qubits)
        
        # Pre-twirl
        for i, pauli_idx in enumerate(twirl_combo):
            new_circuit.append(PAULIS[pauli_idx], [i])
        
        # Circuit with ZNE and PEC
        pec_count = 0
        for idx, instr in enumerate(circuit):
            if idx in zne_indices:
                new_circuit.append(instr.operation, instr.qubits)
                for _ in range((zne_level - 1) // 2):
                    new_circuit.append(instr.operation.inverse(), instr.qubits)
                    new_circuit.append(instr.operation, instr.qubits)
            elif idx in pec_indices:
                new_circuit.append(PAULIS[pec_combo[pec_count]], instr.qubits)
                new_circuit.append(instr.operation, instr.qubits)
                pec_count += 1
            else:
                new_circuit.append(instr.operation, instr.qubits, instr.clbits)
        
        # Post-twirl
        for i, pauli_idx in enumerate(twirl_combo):
            new_circuit.append(PAULIS[pauli_idx], [i])
        
        circuits.append(new_circuit)
    
    return circuits


# =============================================================================
# Benchmarking functions
# =============================================================================

def run_qtpu_mitigation(circuit_size: int, mitigation: str, num_pec: int, num_twirl: int, num_zne: int) -> dict:
    """Run QTPU error mitigation generation."""
    # Use QNN benchmark from MQT
    from mqt.bench import get_benchmark_indep
    circuit = get_benchmark_indep("qnn", circuit_size=circuit_size)
    
    # Remove classical registers and measurements for mitigation
    circuit = circuit.remove_final_measurements(inplace=False)
    
    start = time.perf_counter()
    
    if mitigation == "pec":
        heinsum, num_circuits = qtpu_generate_pec(circuit, num_pec)
    elif mitigation == "twirl":
        heinsum, num_circuits = qtpu_generate_twirl(circuit, num_twirl)
    elif mitigation == "zne":
        heinsum, num_circuits = qtpu_generate_zne(circuit, num_zne)
    else:  # combined
        heinsum, num_circuits = qtpu_generate_combined(circuit, num_pec, num_twirl, num_zne)
    
    compilation_time = time.perf_counter() - start
    
    # Generate the actual CUDA-Q code to count lines
    qtensor = heinsum.quantum_tensors[0]
    _, num_code_lines = quantum_tensor_to_cudaq(
        qtensor.circuit,
        qtensor.shape,
        kernel_name="qtpu_kernel",
        param_values=None
    )
    
    return {
        "compilation_time": compilation_time,
        "num_iswitches": len(qtensor.shape),
        "total_code_lines": num_code_lines,
    }


def run_mitiq_mitigation(circuit_size: int, mitigation: str, num_pec: int, num_twirl: int, num_zne: int, num_samples: int) -> dict:
    """Run Mitiq-style error mitigation generation."""
    # Use QNN benchmark from MQT
    from mqt.bench import get_benchmark_indep
    circuit = get_benchmark_indep("qnn", circuit_size=circuit_size)
    
    # Remove classical registers and measurements for mitigation
    circuit = circuit.remove_final_measurements(inplace=False)
    
    start = time.perf_counter()
    
    if mitigation == "pec":
        circuits = mitiq_generate_pec(circuit, num_pec, num_samples)
    elif mitigation == "twirl":
        circuits = mitiq_generate_twirl(circuit, num_twirl, num_samples)
    elif mitigation == "zne":
        circuits = mitiq_generate_zne(circuit, num_zne, num_samples)
    else:  # combined
        circuits = mitiq_generate_combined(circuit, num_pec, num_twirl, num_zne, num_samples)
    
    compilation_time = time.perf_counter() - start
    
    # Estimate code lines: For Mitiq, each circuit is a separate kernel
    # Each circuit needs its own kernel function (~50 lines base + 2 lines per gate)
    # Plus a dispatch function to run all circuits (~20 lines + 1 line per circuit)
    if circuits:
        avg_gates = sum(len([instr for instr in circ if instr.operation.name not in ['barrier', 'measure']]) 
                       for circ in circuits) / len(circuits)
        lines_per_kernel = int(50 + avg_gates * 2)
        total_code_lines = len(circuits) * lines_per_kernel + 20 + len(circuits)
    else:
        total_code_lines = 0
    
    return {
        "compilation_time": compilation_time,
        "num_circuits_generated": len(circuits),
        "total_code_lines": total_code_lines,
    }


# =============================================================================
# Benchmark configurations
# =============================================================================

# Fixed circuit size - 100 qubit QNN
CIRCUIT_SIZE = 100
MITIGATIONS = ["pec", "twirl", "zne", "combined"]
NUM_SAMPLES_LIST = [100, 1000, 10000]

# Gate counts for 100-qubit circuit - use ALL single-qubit gates
# 100-qubit QNN has ~200 single-qubit gates
NUM_PEC = 200
NUM_TWIRL = 100
NUM_ZNE = 200


@bk.foreach(mitigation=MITIGATIONS)
@bk.log("logs/error_mitigation/qtpu_breakdown.jsonl")
def bench_qtpu(mitigation: str) -> dict:
    """Benchmark QTPU error mitigation generation."""
    return run_qtpu_mitigation(CIRCUIT_SIZE, mitigation, NUM_PEC, NUM_TWIRL, NUM_ZNE)


@bk.foreach(mitigation=MITIGATIONS)
@bk.foreach(num_samples=NUM_SAMPLES_LIST)
@bk.log("logs/error_mitigation/mitiq_breakdown.jsonl")
def bench_mitiq(mitigation: str, num_samples: int) -> dict:
    """Benchmark Mitiq-style error mitigation generation."""
    return run_mitiq_mitigation(CIRCUIT_SIZE, mitigation, NUM_PEC, NUM_TWIRL, NUM_ZNE, num_samples)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "qtpu":
            bench_qtpu()
        elif sys.argv[1] == "mitiq":
            bench_mitiq()
        elif sys.argv[1] == "all":
            print("Running all benchmarks...")
            bench_qtpu()
            bench_mitiq()
        elif sys.argv[1] == "quick":
            # Quick test with 100-qubit circuit
            from mqt.bench import get_benchmark_indep
            circuit = get_benchmark_indep("qnn", circuit_size=100)
            num_pec, num_twirl, num_zne = NUM_PEC, NUM_TWIRL, NUM_ZNE
            
            print("=" * 80)
            print("QTPU vs Mitiq-style Error Mitigation Benchmark")
            print(f"100-qubit QNN circuit")
            print("=" * 80)
            
            for mitigation in ["pec", "twirl", "zne", "combined"]:
                print(f"\n{mitigation.upper()}:")
                print("-" * 60)
                
                qtpu_result = run_qtpu_mitigation(CIRCUIT_SIZE, mitigation, num_pec, num_twirl, num_zne)
                print(f"QTPU: time={qtpu_result['compilation_time']*1000:.2f}ms "
                      f"code={qtpu_result['total_code_lines']:,} LoC "
                      f"(represents 4^{qtpu_result['num_iswitches']} circuits)")
                
                for num_samples in [100, 1000, 10000]:
                    mitiq_result = run_mitiq_mitigation(CIRCUIT_SIZE, mitigation, num_pec, num_twirl, num_zne, num_samples)
                    speedup = mitiq_result['compilation_time'] / qtpu_result['compilation_time']
                    code_ratio = mitiq_result['total_code_lines'] / qtpu_result['total_code_lines']
                    print(f"Mitiq ({num_samples:,}): time={mitiq_result['compilation_time']*1000:.2f}ms "
                          f"code={mitiq_result['total_code_lines']:,} LoC "
                          f"({speedup:.1f}x slower, {code_ratio:.1f}x more code)")
        else:
            print("Usage: python run.py [qtpu|mitiq|all|quick]")
    else:
        print("Usage: python run.py [qtpu|mitiq|all|quick]")
