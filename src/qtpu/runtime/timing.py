"""Timing breakdown utilities for runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimingBreakdown:
    """Detailed timing breakdown for a single execution.
    
    Attributes:
        quantum_eval_time: Wall-clock time spent evaluating quantum circuits.
        quantum_estimated_qpu_time: Estimated time on real QPU hardware (from scheduling).
        num_circuits: Total number of circuits evaluated.
        classical_contraction_time: Time spent on tensor contraction.
        data_transfer_time: Time spent moving data (e.g., CPU <-> GPU).
        total_time: Total wall-clock time for the execution.
        device: Device used for classical computation.
        backend: Quantum backend used.
    """
    
    # Quantum timing
    quantum_eval_time: float = 0.0
    quantum_estimated_qpu_time: float = 0.0
    num_circuits: int = 0
    
    # Classical timing
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
                self.quantum_eval_time 
                + self.classical_contraction_time 
                + self.data_transfer_time
            )
    
    @property
    def quantum_time(self) -> float:
        """Total quantum-related time (evaluation wall clock)."""
        return self.quantum_eval_time
    
    @property
    def classical_time(self) -> float:
        """Total classical time (contraction + transfer)."""
        return self.classical_contraction_time + self.data_transfer_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "quantum_eval_time": self.quantum_eval_time,
            "quantum_estimated_qpu_time": self.quantum_estimated_qpu_time,
            "num_circuits": self.num_circuits,
            "classical_contraction_time": self.classical_contraction_time,
            "data_transfer_time": self.data_transfer_time,
            "total_time": self.total_time,
            "device": self.device,
            "backend": self.backend,
        }
    
    def __repr__(self) -> str:
        return (
            f"TimingBreakdown(quantum={self.quantum_eval_time:.3f}s, "
            f"classical={self.classical_time:.3f}s, "
            f"total={self.total_time:.3f}s)"
        )


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
    
    def add(self, timing: TimingBreakdown) -> None:
        """Add a timing breakdown to the aggregate."""
        self.timings.append(timing)
    
    def clear(self) -> None:
        """Clear all collected timings."""
        self.timings.clear()
    
    def __len__(self) -> int:
        return len(self.timings)
    
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
            quantum_eval_time=std([t.quantum_eval_time for t in self.timings], mean.quantum_eval_time),
            quantum_estimated_qpu_time=std([t.quantum_estimated_qpu_time for t in self.timings], mean.quantum_estimated_qpu_time),
            classical_contraction_time=std([t.classical_contraction_time for t in self.timings], mean.classical_contraction_time),
            data_transfer_time=std([t.data_transfer_time for t in self.timings], mean.data_transfer_time),
            total_time=std([t.total_time for t in self.timings], mean.total_time),
            device=mean.device,
            backend=mean.backend,
        )
