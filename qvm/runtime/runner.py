import abc
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit import QuantumRegister as Fragment

from qvm.virtual_circuit import VirtualCircuit

from ._types import Counts
from .virtualizer import generate_instance_parameters


@dataclass
class RuntimeInfo:
    qpu_time: float
    knit_time: float


class Runner(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        circuit: QuantumCircuit,
        arg_batch: list[dict[Parameter, float]],
        shots: int = 10000,
    ) -> list[Counts]:
        ...


def sample_fragments(
    virtual_circuit: VirtualCircuit, runner: Runner, shots: int = 10000
) -> dict[Fragment, NDArray[np.float32]]:
    results = {}
    for frag, param_batch in generate_instance_parameters(virtual_circuit).items():
        circuit = virtual_circuit.fragment_circuits[frag]
        results[frag] = np.array(
            [
                expval_from_counts(counts)
                for counts in runner.run(circuit, param_batch, shots=shots)
            ],
            dtype=np.float32,
        )
    return results


def expval_from_counts(counts: dict[str, int]) -> float:
    expval = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        parity = 1 - 2 * int(bitstring.count("1") % 2)
        expval += parity * (count / shots)
    return expval
