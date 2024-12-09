import pytest
from collections import Counter
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer.primitives import SamplerV2
import qtpu
from qtpu.evaluators import SamplerEvaluator

from tests._circuit import simple_circuit


def sample_circuit_qtpu(circuit: QuantumCircuit, num_shots: int) -> dict[str, int]:
    """
    Samples the given circuit using QTPU.
    Returns the counts of the circuit.
    """
    cut_circ = qtpu.cut(circuit, num_qubits=circuit.num_qubits // 2)
    hybrid_tn = qtpu.circuit_to_hybrid_tn(cut_circ)
    evaluator = SamplerEvaluator()
    tn = qtpu.evaluate(hybrid_tn, evaluator)
    sample_results = qtpu.sample(tn, num_samples=num_shots)
    return dict(Counter(sample_results))


def run_comparison(circuit: QuantumCircuit, num_shots: int) -> dict[str, int]:
    """
    Runs the circuit using Qiskit's SamplerV2 and returns the counts.
    """
    counts = (
        SamplerV2().run([circuit], shots=num_shots).result()[0].data.meas.get_counts()
    )
    return counts


@pytest.mark.parametrize("execution_number", range(3))
def test_sampler_integration(execution_number):
    """
    Integration test for the entire workflow.
    """
    num_qubits = 4
    num_shots = 10000

    circuit = simple_circuit(num_qubits)

    # Run the workflow using QTPU
    qtpu_counts = sample_circuit_qtpu(circuit, num_shots)

    # Run the workflow using Qiskit's SamplerV2
    qiskit_counts = run_comparison(circuit, num_shots)

    # Calculate Hellinger fidelity between the distributions
    fidelity = hellinger_fidelity(qtpu_counts, qiskit_counts)

    # Validate the fidelity is above a threshold
    assert fidelity > 0.95, f"Fidelity is too low: {fidelity}"

    # Validate that the distributions are not empty
    assert qtpu_counts, "QTPU counts are empty."
    assert qiskit_counts, "Qiskit counts are empty."
