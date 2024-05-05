import abc

from qiskit_aer import AerSimulator

import numpy as np
import quimb.tensor as qtn

from qvm.tensor import QuantumTensor


class QPUManager(abc.ABC):
    @abc.abstractmethod
    def run_quantum_tensor(self, tensor: QuantumTensor) -> qtn.Tensor: ...


class _DummyQPUManager(QPUManager):
    def run_quantum_tensor(self, tensor: QuantumTensor) -> qtn.Tensor:
        return qtn.Tensor(np.random.randn(tensor.shape), inds=tensor.indices)


class SimulatorQPUManager(QPUManager):
    def run_quantum_tensor(self, quantum_tensor: QuantumTensor) -> qtn.Tensor:
        simulator = AerSimulator()
        circuits = list(quantum_tensor.instances())

        expvals = []

        for circuit in circuits:
            try:
                counts = simulator.run(circuit, shots=20000).result().get_counts()
                expvals.append(expval_from_counts(counts))
            except Exception as e:
                expvals.append(1.0)

        expvals = np.array(expvals, dtype=np.float32)
        expvals = expvals.reshape(quantum_tensor.shape)

        return qtn.Tensor(expvals, inds=quantum_tensor.indices)


def expval_from_counts(counts: dict[str, int]) -> float:
    expval = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        parity = 1 - 2 * int(bitstring.count("1") % 2)
        expval += parity * (count / shots)
    return expval
