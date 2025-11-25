"""Clean HTNLayer with einsum expression API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from qtpu.tensor import QuantumTensor, ISwitch
from qtpu.torch.differentiable import differentiable_evaluate

if TYPE_CHECKING:
    from collections.abc import Sequence
    
    from qiskit.circuit import QuantumCircuit
    
    from qtpu.torch.evaluator import TorchCircuitEvaluator


class ParameterizedCircuitTensor:
    """A circuit tensor with ISwitch indices and learnable parameters.
    
    Attributes:
        circuit: The quantum circuit template.
        shape: Shape from ISwitch dimensions (tensor indices).
        inds: Index names from ISwitches.
        learnable_params: Names of learnable circuit parameters.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        learnable_params: Sequence[str] | None = None,
    ) -> None:
        """Initialize a parameterized circuit tensor.

        Args:
            circuit: Quantum circuit with ISwitches and/or learnable parameters.
            learnable_params: Names of parameters to treat as learnable.
                If None, all non-ISwitch parameters are learnable.
        """
        self._circuit = circuit
        
        # Find ISwitch parameters (tensor indices, not learnable)
        iswitch_params = set()
        param_to_op = {}
        for instr in circuit:
            if isinstance(instr.operation, ISwitch):
                iswitch_params.add(instr.operation.param.name)
                param_to_op[instr.operation.param] = instr.operation
        
        # Build shape and inds from ISwitches (deterministic order)
        param_to_op_list = sorted(param_to_op.items(), key=lambda x: x[0].name)
        self._shape = tuple(len(op) for _, op in param_to_op_list)
        self._inds = tuple(p.name for p, _ in param_to_op_list)
        
        # Identify learnable parameters
        all_params = {p.name for p in circuit.parameters}
        non_iswitch_params = all_params - iswitch_params
        
        if learnable_params is None:
            self._learnable_params = sorted(non_iswitch_params)
        else:
            self._learnable_params = list(learnable_params)
            invalid = set(self._learnable_params) - non_iswitch_params
            if invalid:
                raise ValueError(f"Parameters {invalid} not in circuit or are ISwitch params")

    @property
    def circuit(self) -> QuantumCircuit:
        """The underlying quantum circuit template."""
        return self._circuit

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor (from ISwitch dimensions)."""
        return self._shape

    @property
    def inds(self) -> tuple[str, ...]:
        """Index names (from ISwitches)."""
        return self._inds

    @property
    def learnable_params(self) -> list[str]:
        """Names of learnable parameters."""
        return self._learnable_params

    def bind_iswitch(self, index: int | tuple[int, ...]) -> QuantumCircuit:
        """Bind ISwitch parameters to specific indices.

        Args:
            index: ISwitch index/indices to bind.

        Returns:
            Circuit with ISwitches bound (learnable params still symbolic).
        """
        if isinstance(index, int):
            index = (index,)
        
        if len(index) != len(self.shape):
            raise ValueError(f"Index {index} doesn't match shape {self.shape}")
        
        param_map = dict(zip(self.inds, index, strict=False))
        return self._circuit.assign_parameters(param_map)

    def flat_circuits(self) -> list[QuantumCircuit]:
        """Get all circuits with ISwitches bound (one per configuration).

        Returns:
            List of circuits with learnable parameters still symbolic.
        """
        if len(self.shape) == 0:
            return [self._circuit.copy()]
        
        indices = np.ndindex(self.shape)
        return [self.bind_iswitch(tuple(idx)) for idx in indices]


class HTNLayer(nn.Module):
    """Hybrid Tensor Network Layer with einsum-style contraction.
    
    This layer represents a hybrid tensor network where:
    - First m tensors come from quantum circuit evaluation
    - Next k tensors are learnable classical tensors (nn.Parameter)
    - Last n tensors are provided at forward time
    
    Circuit parameters can be:
    - Learnable (registered as nn.Parameter, trained via parameter-shift)
    - Fixed (passed at forward time)
    
    Example:
        >>> # Einsum: "ij,jk,kl,lm->im"
        >>> # - 2 circuit tensors (ij, jk) 
        >>> # - 1 learnable classical tensor (kl)
        >>> # - 1 input classical tensor (lm)
        >>> 
        >>> layer = HTNLayer(
        ...     einsum_expr="ij,jk,kl,lm->im",
        ...     circuit_tensors=[ct1, ct2],
        ...     classical_tensors=[torch.randn(4, 5)],  # learnable
        ...     evaluator=evaluator,
        ... )
        >>> 
        >>> # Forward: only input tensors (circuit params are learned)
        >>> output = layer(input_tensor)
        >>> 
        >>> # Or with fixed circuit params:
        >>> output = layer(input_tensor, circuit_params=[{"theta": 0.5}, {"phi": 1.2}])
    """

    def __init__(
        self,
        einsum_expr: str,
        circuit_tensors: list[QuantumCircuit | ParameterizedCircuitTensor],
        evaluator: TorchCircuitEvaluator,
        classical_tensors: list[torch.Tensor] | None = None,
        learnable_circuit_params: bool = True,
        init_circuit_params: dict[int, dict[str, float]] | None = None,
    ) -> None:
        """Initialize the HTN layer.

        Args:
            einsum_expr: Einsum expression defining the contraction.
                E.g., "ij,jk,kl->il" for three tensors contracted to output.
            circuit_tensors: List of quantum circuits (first m tensors in einsum).
                Can be QuantumCircuit or ParameterizedCircuitTensor.
            evaluator: Evaluator for running quantum circuits.
            classical_tensors: Optional learnable classical tensors (next k tensors).
                These become nn.Parameters.
            learnable_circuit_params: If True, circuit parameters become nn.Parameters.
                If False, they must be passed at forward time.
            init_circuit_params: Initial values for circuit parameters.
                Dict mapping circuit index to {param_name: value}.
        """
        super().__init__()
        
        self._einsum_expr = einsum_expr
        self._evaluator = evaluator
        self._learnable_circuit_params = learnable_circuit_params
        
        # Parse einsum to understand tensor structure
        self._input_specs, self._output_spec = self._parse_einsum(einsum_expr)
        
        # Wrap circuit tensors
        self._num_circuit_tensors = len(circuit_tensors)
        self._pcts: list[ParameterizedCircuitTensor] = []
        for i, ct in enumerate(circuit_tensors):
            if isinstance(ct, ParameterizedCircuitTensor):
                pct = ct
            else:
                pct = ParameterizedCircuitTensor(ct)
            self._pcts.append(pct)
        
        # Validate circuit tensor shapes match einsum
        for i, (pct, spec) in enumerate(zip(self._pcts, self._input_specs)):
            if len(pct.shape) != len(spec):
                raise ValueError(
                    f"Circuit tensor {i} has shape {pct.shape} ({len(pct.shape)} dims) "
                    f"but einsum expects {len(spec)} dims ('{spec}')"
                )
        
        # Register learnable circuit parameters
        self._circuit_param_keys: list[list[str]] = []  # For each circuit, list of param names
        if learnable_circuit_params:
            for i, pct in enumerate(self._pcts):
                param_keys = []
                for param_name in pct.learnable_params:
                    key = f"qparam_{i}_{param_name}"
                    # Initialize with provided value or random
                    init_val = 0.0
                    if init_circuit_params and i in init_circuit_params:
                        init_val = init_circuit_params[i].get(param_name, 0.0)
                    param = nn.Parameter(torch.tensor(init_val, dtype=torch.float64))
                    self.register_parameter(key, param)
                    param_keys.append(param_name)
                self._circuit_param_keys.append(param_keys)
        
        # Register learnable classical tensors
        self._num_learnable_classical = 0
        if classical_tensors is not None:
            self._num_learnable_classical = len(classical_tensors)
            for i, tensor in enumerate(classical_tensors):
                param = nn.Parameter(tensor.clone().detach().to(torch.float64))
                self.register_parameter(f"classical_{i}", param)
        
        # Number of input tensors expected at forward time
        total_tensors = len(self._input_specs)
        self._num_input_tensors = total_tensors - self._num_circuit_tensors - self._num_learnable_classical
        
        if self._num_input_tensors < 0:
            raise ValueError(
                f"Einsum has {total_tensors} input tensors, but got "
                f"{self._num_circuit_tensors} circuits + {self._num_learnable_classical} learnable classical"
            )

    def _parse_einsum(self, expr: str) -> tuple[list[str], str]:
        """Parse einsum expression into input specs and output spec."""
        if "->" not in expr:
            raise ValueError(f"Einsum expression must contain '->': {expr}")
        
        inputs_str, output = expr.replace(" ", "").split("->")
        input_specs = inputs_str.split(",")
        return input_specs, output

    def _get_circuit_params(self, circuit_idx: int) -> dict[str, torch.Tensor]:
        """Get the current parameter values for a circuit tensor."""
        params = {}
        for param_name in self._circuit_param_keys[circuit_idx]:
            key = f"qparam_{circuit_idx}_{param_name}"
            params[param_name] = getattr(self, key)
        return params

    def _evaluate_circuit_tensor_differentiable(
        self,
        circuit_idx: int,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate circuit tensor with differentiable parameters.

        Uses parameter-shift rule for gradient computation.
        """
        pct = self._pcts[circuit_idx]
        
        if len(pct.shape) == 0:
            # No ISwitches, single circuit - use differentiable evaluation
            return differentiable_evaluate(pct.circuit, self._evaluator, params)
        
        # Evaluate all ISwitch configurations with differentiable params
        circuits = pct.flat_circuits()
        results = []
        for circuit in circuits:
            result = differentiable_evaluate(circuit, self._evaluator, params)
            results.append(result)
        
        # Stack and reshape
        result_array = torch.stack(results)
        return result_array.reshape(pct.shape)

    def _evaluate_circuit_tensor(
        self,
        circuit_idx: int,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate circuit tensor (non-differentiable w.r.t. circuit params)."""
        pct = self._pcts[circuit_idx]
        
        if len(pct.shape) == 0:
            return self._evaluator.evaluate(pct.circuit, params)
        
        circuits = pct.flat_circuits()
        results = self._evaluator.evaluate_batch(circuits, params)
        result_array = torch.stack(results)
        return result_array.reshape(pct.shape)

    def forward(
        self,
        *input_tensors: torch.Tensor,
        circuit_params: list[dict[str, float | torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Forward pass: evaluate circuits and contract.

        Args:
            *input_tensors: Classical tensors to contract (last n in einsum).
            circuit_params: Optional override for circuit parameters.
                If None and learnable_circuit_params=True, uses learned params.
                If provided, uses these values instead.

        Returns:
            torch.Tensor: The contracted result.
        """
        # Validate input tensors
        if len(input_tensors) != self._num_input_tensors:
            raise ValueError(
                f"Expected {self._num_input_tensors} input tensors, got {len(input_tensors)}"
            )
        
        for i, tensor in enumerate(input_tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Input {i} should be torch.Tensor, got {type(tensor)}")
        
        # Evaluate circuit tensors
        operands = []
        for i in range(self._num_circuit_tensors):
            if circuit_params is not None:
                # Use provided params (convert to tensors)
                params = {}
                for name, value in circuit_params[i].items():
                    if isinstance(value, torch.Tensor):
                        params[name] = value
                    else:
                        params[name] = torch.tensor(value, dtype=torch.float64)
                qtensor = self._evaluate_circuit_tensor(i, params)
            elif self._learnable_circuit_params:
                # Use learned params with gradient support
                params = self._get_circuit_params(i)
                qtensor = self._evaluate_circuit_tensor_differentiable(i, params)
            else:
                raise ValueError(
                    "circuit_params must be provided when learnable_circuit_params=False"
                )
            operands.append(qtensor)
        
        # Add learnable classical tensors
        for i in range(self._num_learnable_classical):
            param = getattr(self, f"classical_{i}")
            operands.append(param)
        
        # Add input tensors
        operands.extend(input_tensors)
        
        # Contract using torch.einsum
        return torch.einsum(self._einsum_expr, *operands)

    def initialize_circuit_params(
        self,
        method: str = "random",
        seed: int | None = None,
    ) -> None:
        """Initialize circuit parameters.

        Args:
            method: Initialization method - "random", "zeros", or "uniform".
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        for i, pct in enumerate(self._pcts):
            for param_name in pct.learnable_params:
                key = f"qparam_{i}_{param_name}"
                param = getattr(self, key)
                if method == "random":
                    param.data = torch.randn(1, dtype=torch.float64).squeeze() * np.pi
                elif method == "zeros":
                    param.data = torch.tensor(0.0, dtype=torch.float64)
                elif method == "uniform":
                    param.data = torch.rand(1, dtype=torch.float64).squeeze() * 2 * np.pi

    @property
    def num_circuit_tensors(self) -> int:
        """Number of circuit tensors."""
        return self._num_circuit_tensors

    @property
    def num_learnable_classical(self) -> int:
        """Number of learnable classical tensors."""
        return self._num_learnable_classical

    @property
    def num_input_tensors(self) -> int:
        """Number of classical tensors expected at forward time."""
        return self._num_input_tensors

    @property
    def circuit_tensors(self) -> list[ParameterizedCircuitTensor]:
        """The parameterized circuit tensors."""
        return self._pcts

    @property
    def einsum_expr(self) -> str:
        """The einsum contraction expression."""
        return self._einsum_expr

    @property
    def num_circuit_params(self) -> int:
        """Total number of learnable circuit parameters."""
        return sum(len(keys) for keys in self._circuit_param_keys)

    def get_circuit_param(self, circuit_idx: int, param_name: str) -> nn.Parameter:
        """Get a specific circuit parameter."""
        key = f"qparam_{circuit_idx}_{param_name}"
        return getattr(self, key)

    def get_learnable_classical(self, idx: int) -> nn.Parameter:
        """Get a learnable classical tensor by index."""
        return getattr(self, f"classical_{idx}")

    @classmethod
    def from_hybrid_tn(
        cls,
        hybrid_tn,  # HybridTensorNetwork
        evaluator: TorchCircuitEvaluator,
        output_inds: Sequence[str] = (),
        learnable_classical: bool = True,
        learnable_circuit_params: bool = True,
    ) -> HTNLayer:
        """Create an HTNLayer from a HybridTensorNetwork.

        Args:
            hybrid_tn: The hybrid tensor network.
            evaluator: Circuit evaluator.
            output_inds: Output indices for the contraction.
            learnable_classical: If True, classical tensors become learnable.
            learnable_circuit_params: If True, circuit params become learnable.

        Returns:
            HTNLayer instance.
        """
        # Build einsum expression
        input_specs = []
        
        # Circuit tensor indices
        for qt in hybrid_tn.qtensors:
            ct = QuantumTensor(qt.circuit) if not isinstance(qt, QuantumTensor) else qt
            input_specs.append("".join(ct.inds))
        
        # Classical tensor indices
        for ct in hybrid_tn.ctensors:
            input_specs.append("".join(ct.inds))
        
        einsum_expr = ",".join(input_specs) + "->" + "".join(output_inds)
        
        # Extract circuits and classical tensors
        circuits = [qt.circuit for qt in hybrid_tn.qtensors]
        
        classical_tensors = None
        if learnable_classical:
            classical_tensors = [
                torch.tensor(ct.data, dtype=torch.float64) for ct in hybrid_tn.ctensors
            ]
        
        return cls(
            einsum_expr=einsum_expr,
            circuit_tensors=circuits,
            evaluator=evaluator,
            classical_tensors=classical_tensors if learnable_classical else None,
            learnable_circuit_params=learnable_circuit_params,
        )
