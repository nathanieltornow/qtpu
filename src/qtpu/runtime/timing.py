"""Timing breakdown utilities for runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimingBreakdown:
    """Detailed timing breakdown for a single execution.
    
    Attributes:
        # Preprocessing (one-time costs, amortized over many executions)
        circuit_generation_time: Time to generate/instantiate quantum circuits.
        circuit_compilation_time: Time to compile circuits (e.g., CUDA-Q JIT).
        optimization_time: Time for tensor network optimization (path finding).
        
        # Per-execution costs
        quantum_eval_time: Wall-clock time spent evaluating quantum circuits.
        quantum_estimated_qpu_time: Estimated time on real QPU hardware.
        classical_contraction_time: Time spent on tensor contraction.
        data_transfer_time: Time spent moving data (e.g., CPU <-> GPU).
        
        # Totals
        num_circuits: Total number of circuits evaluated.
        total_time: Total wall-clock time for the execution.
        
        # Metadata
        device: Device used for classical computation.
        backend: Quantum backend used.
    """
    
    # Preprocessing timing (one-time, can be amortized)
    circuit_generation_time: float = 0.0
    circuit_compilation_time: float = 0.0
    optimization_time: float = 0.0
    
    # Quantum timing (per-execution)
    quantum_eval_time: float = 0.0
    quantum_estimated_qpu_time: float = 0.0
    num_circuits: int = 0
    
    # Classical timing (per-execution)
    classical_contraction_time: float = 0.0
    data_transfer_time: float = 0.0
    
    # Total
    total_time: float = 0.0
    
    # Metadata
    device: str = "cpu"
    backend: str = "simulator"
    
    def __post_init__(self):
        if self.total_time == 0.0:
            self.total_time = (
                self.preprocessing_time
                + self.quantum_eval_time 
                + self.classical_contraction_time 
                + self.data_transfer_time
            )
    
    @property
    def preprocessing_time(self) -> float:
        """Total preprocessing time (circuit gen + compile + optimize)."""
        return (
            self.circuit_generation_time 
            + self.circuit_compilation_time 
            + self.optimization_time
        )
    
    @property
    def quantum_time(self) -> float:
        """Total quantum-related time (evaluation wall clock)."""
        return self.quantum_eval_time
    
    @property
    def classical_time(self) -> float:
        """Total classical time (contraction + transfer)."""
        return self.classical_contraction_time + self.data_transfer_time
    
    @property
    def execution_time(self) -> float:
        """Per-execution time (excludes preprocessing)."""
        return self.quantum_eval_time + self.classical_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            # Preprocessing
            "circuit_generation_time": self.circuit_generation_time,
            "circuit_compilation_time": self.circuit_compilation_time,
            "optimization_time": self.optimization_time,
            "preprocessing_time": self.preprocessing_time,
            # Quantum
            "quantum_eval_time": self.quantum_eval_time,
            "quantum_estimated_qpu_time": self.quantum_estimated_qpu_time,
            "num_circuits": self.num_circuits,
            # Classical
            "classical_contraction_time": self.classical_contraction_time,
            "data_transfer_time": self.data_transfer_time,
            # Totals
            "execution_time": self.execution_time,
            "total_time": self.total_time,
            # Metadata
            "device": self.device,
            "backend": self.backend,
        }
    
    def __repr__(self) -> str:
        parts = []
        if self.preprocessing_time > 0:
            parts.append(f"preprocess={self.preprocessing_time:.3f}s")
        parts.append(f"quantum={self.quantum_eval_time:.3f}s")
        parts.append(f"classical={self.classical_time:.3f}s")
        parts.append(f"total={self.total_time:.3f}s")
        return f"TimingBreakdown({', '.join(parts)})"
    
    def summary(self) -> str:
        """Return a detailed multi-line summary."""
        lines = [
            "Timing Breakdown:",
            f"  Preprocessing: {self.preprocessing_time*1000:.1f}ms",
            f"    - Circuit generation: {self.circuit_generation_time*1000:.1f}ms",
            f"    - Circuit compilation: {self.circuit_compilation_time*1000:.1f}ms",
            f"    - Optimization: {self.optimization_time*1000:.1f}ms",
            f"  Quantum: {self.quantum_eval_time*1000:.1f}ms ({self.num_circuits} circuits)",
            f"    - Estimated QPU time: {self.quantum_estimated_qpu_time*1000:.1f}ms",
            f"  Classical: {self.classical_time*1000:.1f}ms",
            f"    - Contraction: {self.classical_contraction_time*1000:.1f}ms",
            f"    - Data transfer: {self.data_transfer_time*1000:.1f}ms",
            f"  Total: {self.total_time*1000:.1f}ms",
        ]
        return "\n".join(lines)


@dataclass 
class AggregateTimingStats:
    """Aggregate statistics over multiple executions.
    
    Example:
        >>> stats = AggregateTimingStats()
        >>> for _ in range(10):
        ...     result, timing = runtime.execute(inputs)
        ...     stats.add(timing)
        >>> print(f"Mean: {stats.mean_timing}")
    """
    
    timings: list[TimingBreakdown] = field(default_factory=list)
    
    # One-time preprocessing costs (recorded separately)
    preprocessing: TimingBreakdown | None = None
    
    def add(self, timing: TimingBreakdown) -> None:
        """Add a timing breakdown to the aggregate."""
        self.timings.append(timing)
    
    def set_preprocessing(self, timing: TimingBreakdown) -> None:
        """Set the one-time preprocessing timing."""
        self.preprocessing = timing
    
    def clear(self) -> None:
        """Clear all collected timings."""
        self.timings.clear()
    
    def __len__(self) -> int:
        return len(self.timings)
    
    @property
    def total_preprocessing_time(self) -> float:
        """Total preprocessing time (one-time cost)."""
        if self.preprocessing:
            return self.preprocessing.preprocessing_time
        return sum(t.preprocessing_time for t in self.timings)
    
    @property
    def total_quantum_time(self) -> float:
        """Sum of quantum evaluation time across all executions."""
        return sum(t.quantum_eval_time for t in self.timings)
    
    @property
    def total_classical_time(self) -> float:
        """Sum of classical time across all executions."""
        return sum(t.classical_time for t in self.timings)
    
    @property
    def total_time(self) -> float:
        """Sum of total time across all executions."""
        return sum(t.total_time for t in self.timings)
    
    @property
    def total_estimated_qpu_time(self) -> float:
        """Sum of estimated QPU time across all executions."""
        return sum(t.quantum_estimated_qpu_time for t in self.timings)
    
    @property
    def mean_timing(self) -> TimingBreakdown:
        """Get mean timing across all executions."""
        n = len(self.timings)
        if n == 0:
            return TimingBreakdown()
        return TimingBreakdown(
            circuit_generation_time=sum(t.circuit_generation_time for t in self.timings) / n,
            circuit_compilation_time=sum(t.circuit_compilation_time for t in self.timings) / n,
            optimization_time=sum(t.optimization_time for t in self.timings) / n,
            quantum_eval_time=self.total_quantum_time / n,
            quantum_estimated_qpu_time=self.total_estimated_qpu_time / n,
            num_circuits=sum(t.num_circuits for t in self.timings) // n,
            classical_contraction_time=sum(t.classical_contraction_time for t in self.timings) / n,
            data_transfer_time=sum(t.data_transfer_time for t in self.timings) / n,
            total_time=self.total_time / n,
            device=self.timings[0].device if self.timings else "cpu",
            backend=self.timings[0].backend if self.timings else "simulator",
        )
    
    @property
    def std_timing(self) -> TimingBreakdown:
        """Get standard deviation of timing across all executions."""
        import math
        
        n = len(self.timings)
        if n < 2:
            return TimingBreakdown()
        
        mean = self.mean_timing
        
        def std(values: list[float], mean_val: float) -> float:
            variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
            return math.sqrt(variance)
        
        return TimingBreakdown(
            circuit_generation_time=std([t.circuit_generation_time for t in self.timings], mean.circuit_generation_time),
            circuit_compilation_time=std([t.circuit_compilation_time for t in self.timings], mean.circuit_compilation_time),
            optimization_time=std([t.optimization_time for t in self.timings], mean.optimization_time),
            quantum_eval_time=std([t.quantum_eval_time for t in self.timings], mean.quantum_eval_time),
            quantum_estimated_qpu_time=std([t.quantum_estimated_qpu_time for t in self.timings], mean.quantum_estimated_qpu_time),
            classical_contraction_time=std([t.classical_contraction_time for t in self.timings], mean.classical_contraction_time),
            data_transfer_time=std([t.data_transfer_time for t in self.timings], mean.data_transfer_time),
            total_time=std([t.total_time for t in self.timings], mean.total_time),
            device=mean.device,
            backend=mean.backend,
        )
    
    def summary(self) -> str:
        """Return a detailed multi-line summary."""
        n = len(self.timings)
        if n == 0:
            return "No timings recorded"
        
        mean = self.mean_timing
        lines = [
            f"Aggregate Timing Stats ({n} executions):",
            f"  Preprocessing (one-time): {self.total_preprocessing_time*1000:.1f}ms",
            f"  Mean per execution:",
            f"    - Quantum: {mean.quantum_eval_time*1000:.1f}ms",
            f"    - Classical: {mean.classical_time*1000:.1f}ms",
            f"    - Total: {mean.execution_time*1000:.1f}ms",
            f"  Total across all executions:",
            f"    - Quantum: {self.total_quantum_time*1000:.1f}ms",
            f"    - Classical: {self.total_classical_time*1000:.1f}ms", 
            f"    - Total: {self.total_time*1000:.1f}ms",
        ]
        return "\n".join(lines)
