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
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

import benchkit as bk

from qtpu.tensor import ISwitch, QuantumTensor, CTensor
from qtpu.heinsum import HEinsum

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
    
    num_circuits = int(np.prod(qtensor.shape)) if qtensor.shape else 1
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
    
    num_circuits = int(np.prod(qtensor.shape)) if qtensor.shape else 1
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
    
    num_circuits = int(np.prod(qtensor.shape)) if qtensor.shape else 1
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
    
    num_circuits = int(np.prod(qtensor.shape)) if qtensor.shape else 1
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
    circuit = efficient_su2(circuit_size, reps=2).decompose()
    
    tracemalloc.start()
    start = time.perf_counter()
    
    if mitigation == "pec":
        heinsum, num_circuits = qtpu_generate_pec(circuit, num_pec)
    elif mitigation == "twirl":
        heinsum, num_circuits = qtpu_generate_twirl(circuit, num_twirl)
    elif mitigation == "zne":
        heinsum, num_circuits = qtpu_generate_zne(circuit, num_zne)
    else:  # combined
        heinsum, num_circuits = qtpu_generate_combined(circuit, num_pec, num_twirl, num_zne)
    
    generation_time = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "generation_time": generation_time,
        "generation_memory": peak,
        "num_circuits_represented": num_circuits,
        "tensor_shape": heinsum.quantum_tensors[0].shape,
    }


def run_mitiq_mitigation(circuit_size: int, mitigation: str, num_pec: int, num_twirl: int, num_zne: int, num_samples: int) -> dict:
    """Run Mitiq-style error mitigation generation."""
    circuit = efficient_su2(circuit_size, reps=2).decompose()
    
    tracemalloc.start()
    start = time.perf_counter()
    
    if mitigation == "pec":
        circuits = mitiq_generate_pec(circuit, num_pec, num_samples)
    elif mitigation == "twirl":
        circuits = mitiq_generate_twirl(circuit, num_twirl, num_samples)
    elif mitigation == "zne":
        circuits = mitiq_generate_zne(circuit, num_zne, num_samples)
    else:  # combined
        circuits = mitiq_generate_combined(circuit, num_pec, num_twirl, num_zne, num_samples)
    
    generation_time = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "generation_time": generation_time,
        "generation_memory": peak,
        "num_circuits_generated": len(circuits),
    }


# =============================================================================
# Benchmark configurations
# =============================================================================

CIRCUIT_SIZES = [4, 6, 8, 10]
MITIGATIONS = ["pec", "twirl", "zne", "combined"]
NUM_SAMPLES_LIST = [100, 1000, 10000]

# Gate counts scale with circuit size
def get_gate_counts(circuit_size: int) -> tuple[int, int, int]:
    """Get (num_pec, num_twirl, num_zne) based on circuit size."""
    return circuit_size, circuit_size, circuit_size // 2


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(mitigation=MITIGATIONS)
@bk.log("logs/error_mitigation/qtpu.jsonl")
def bench_qtpu(circuit_size: int, mitigation: str) -> dict:
    """Benchmark QTPU error mitigation generation."""
    num_pec, num_twirl, num_zne = get_gate_counts(circuit_size)
    return run_qtpu_mitigation(circuit_size, mitigation, num_pec, num_twirl, num_zne)


@bk.foreach(circuit_size=CIRCUIT_SIZES)
@bk.foreach(mitigation=MITIGATIONS)
@bk.foreach(num_samples=NUM_SAMPLES_LIST)
@bk.log("logs/error_mitigation/mitiq.jsonl")
def bench_mitiq(circuit_size: int, mitigation: str, num_samples: int) -> dict:
    """Benchmark Mitiq-style error mitigation generation."""
    num_pec, num_twirl, num_zne = get_gate_counts(circuit_size)
    return run_mitiq_mitigation(circuit_size, mitigation, num_pec, num_twirl, num_zne, num_samples)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "qtpu":
            bench_qtpu()
        elif sys.argv[1] == "mitiq":
            bench_mitiq()
        elif sys.argv[1] == "quick":
            # Quick test
            circuit = efficient_su2(6, reps=2).decompose()
            num_pec, num_twirl, num_zne = 6, 6, 3
            
            print("=" * 60)
            print("QTPU vs Mitiq-style Error Mitigation Benchmark")
            print("=" * 60)
            
            for mitigation in ["pec", "twirl", "zne", "combined"]:
                print(f"\n{mitigation.upper()}:")
                print("-" * 40)
                
                qtpu_result = run_qtpu_mitigation(6, mitigation, num_pec, num_twirl, num_zne)
                print(f"QTPU: time={qtpu_result['generation_time']*1000:.2f}ms "
                      f"mem={qtpu_result['generation_memory']/1024:.1f}KB "
                      f"represents={qtpu_result['num_circuits_represented']:,} circuits")
                
                for num_samples in [100, 1000, 10000]:
                    mitiq_result = run_mitiq_mitigation(6, mitigation, num_pec, num_twirl, num_zne, num_samples)
                    speedup = mitiq_result['generation_time'] / qtpu_result['generation_time']
                    mem_ratio = mitiq_result['generation_memory'] / qtpu_result['generation_memory']
                    print(f"Mitiq ({num_samples:,}): time={mitiq_result['generation_time']*1000:.2f}ms "
                          f"mem={mitiq_result['generation_memory']/1024:.1f}KB "
                          f"({speedup:.1f}x slower, {mem_ratio:.1f}x more mem)")
        else:
            print("Usage: python run.py [qtpu|mitiq|quick]")
    else:
        print("Usage: python run.py [qtpu|mitiq|quick]")
