"""Circuit evaluators for PyTorch integration."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class TorchCircuitEvaluator(abc.ABC):
    """Abstract base class for evaluating quantum circuits with PyTorch tensors.
    
    This evaluator takes a circuit and parameter values, and returns a torch tensor
    representing the circuit's output (e.g., expectation values, probabilities).
    """

    @abc.abstractmethod
    def evaluate(
        self,
        circuit: QuantumCircuit,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate a quantum circuit with given parameter values.

        Args:
            circuit: The quantum circuit template with parameters.
            params: Dictionary mapping parameter names to torch tensor values.

        Returns:
            torch.Tensor: The evaluation result (scalar or tensor).
        """
        ...

    def evaluate_batch(
        self,
        circuits: list[QuantumCircuit],
        params: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Evaluate a batch of circuits with the same parameters.

        Args:
            circuits: List of quantum circuits to evaluate.
            params: Dictionary mapping parameter names to torch tensor values.

        Returns:
            list[torch.Tensor]: List of evaluation results.
        """
        return [self.evaluate(circuit, params) for circuit in circuits]


class ExpvalTorchEvaluator(TorchCircuitEvaluator):
    """Evaluator that computes <Z...Z> expectation values.
    
    Uses parameter-shift rule for gradient computation.
    """

    def __init__(
        self,
        shots: int | None = None,
        backend: str = "aer_simulator",
    ) -> None:
        """Initialize the expectation value evaluator.

        Args:
            shots: Number of shots for sampling. None for statevector simulation.
            backend: Backend to use for simulation.
        """
        self._shots = shots
        self._backend = backend

    def evaluate(
        self,
        circuit: QuantumCircuit,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate expectation value of Z observable on measured qubits.

        Args:
            circuit: The quantum circuit with parameters.
            params: Dictionary mapping parameter names to torch tensor values.

        Returns:
            torch.Tensor: The expectation value as a scalar tensor.
        """
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp
        
        # Bind parameters
        param_values = {
            p: float(params[p.name].detach().cpu().numpy())
            for p in circuit.parameters
            if p.name in params
        }
        bound_circuit = circuit.assign_parameters(param_values)
        
        # Get Z observable for measured qubits
        observable = self._get_z_observable(bound_circuit)
        
        # Remove measurements for expectation value computation
        circuit_no_meas = bound_circuit.remove_final_measurements(inplace=False)
        
        estimator = StatevectorEstimator()
        job = estimator.run([(circuit_no_meas, observable)])
        result = job.result()[0]
        
        return torch.tensor(result.data.evs, dtype=torch.float64)

    def _get_z_observable(self, circuit: QuantumCircuit) -> str:
        """Get Z observable string for measured qubits."""
        measured_qubits = set()
        for instr in circuit:
            if instr.operation.name == "measure":
                measured_qubits.add(circuit.qubits.index(instr.qubits[0]))
        
        if not measured_qubits:
            # If no measurements, use all qubits
            return "Z" * circuit.num_qubits
        
        obs = ["I"] * circuit.num_qubits
        for qubit in measured_qubits:
            obs[qubit] = "Z"
        return "".join(reversed(obs))


class StatevectorTorchEvaluator(TorchCircuitEvaluator):
    """Evaluator that returns the full statevector as probabilities.
    
    Useful for getting probability distributions from circuits.
    """

    def evaluate(
        self,
        circuit: QuantumCircuit,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate circuit and return probability distribution.

        Args:
            circuit: The quantum circuit with parameters.
            params: Dictionary mapping parameter names to torch tensor values.

        Returns:
            torch.Tensor: Probability distribution over computational basis.
        """
        from qiskit.quantum_info import Statevector
        
        # Bind parameters
        param_values = {
            p: float(params[p.name].detach().cpu().numpy())
            for p in circuit.parameters
            if p.name in params
        }
        bound_circuit = circuit.assign_parameters(param_values)
        
        # Remove measurements
        circuit_no_meas = bound_circuit.remove_final_measurements(inplace=False)
        
        sv = Statevector(circuit_no_meas)
        probs = np.abs(sv.data) ** 2
        
        return torch.tensor(probs, dtype=torch.float64)
