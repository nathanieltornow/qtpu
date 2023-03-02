import abc

from qiskit.circuit import QuantumCircuit

from qvm.quasi_distr import QuasiDistr


def _circuit_hash(circuit: QuantumCircuit) -> int:
    return hash(
        tuple(
            (instr.name, *instr.qubits, *instr.clbits, *instr.params)
            for instr in circuit.data
        )
    )


class Sampler(abc.ABC):
    @abc.abstractmethod
    def _sample(self, circuits: list[QuantumCircuit], shots: int) -> list[QuasiDistr]:
        pass

    def sample(self, circuits: list[QuantumCircuit], shots: int) -> list[QuasiDistr]:
        # figure out the distinct circuits
        distinct_circuits: dict[int, tuple[list[int], QuantumCircuit]] = {}

        for i, circuit in enumerate(circuits):
            circ_hash = _circuit_hash(circuit)
            if circ_hash in distinct_circuits:
                distinct_circuits[circ_hash][0].append(i)
            else:
                distinct_circuits[circ_hash] = ([i], circuit)

        circs_to_sample = [circuit for _, circuit in distinct_circuits.values()]
        distrs = self._sample(circs_to_sample, shots)

        results: list[QuasiDistr] = [QuasiDistr({})] * len(circuits)
        for i, (indices, _) in enumerate(distinct_circuits.values()):
            for j in indices:
                results[j] = distrs[i]
        return results


class SimulationSampler(Sampler):
    def __init__(self, num_qubits: int) -> None:
        super().__init__()
        self._num_qubits = num_qubits


class IBMQBackendSampler(Sampler):
    pass
