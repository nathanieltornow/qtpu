import abc

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

import numpy as np
import quimb.tensor as qtn

from qvm.tensor import QuantumTensor


class QPUManager(abc.ABC):
    @abc.abstractmethod
    def run_quantum_tensor(
        self, tensor: QuantumTensor, shots: int = 100000, **kwargs
    ) -> qtn.Tensor: ...


class _DummyQPUManager(QPUManager):
    def run_quantum_tensor(self, tensor: QuantumTensor, **kwargs) -> qtn.Tensor:
        return qtn.Tensor(np.random.randn(*tensor.shape), inds=tensor.inds)


class SingleBackendQPUManager(QPUManager):
    def __init__(self, backend: BackendV2) -> None:
        self._backend = backend
        super().__init__()

    def run_quantum_tensor(
        self, quantum_tensor: QuantumTensor, shots: int = 100000, **kwargs
    ) -> qtn.Tensor:
        instances = list(quantum_tensor.instances())
        circuits = [transpile(instance[0], self._backend) for instance in instances]
        cid_withour_meas = [
            i
            for i, circ in enumerate(circuits)
            if circ.count_ops().get("measure", 0) == 0
        ]

        nums_shots = [int(instance[1] * shots) for instance in instances]

        for i in reversed(cid_withour_meas):
            circuits.pop(i)
            nums_shots.pop(i)

        expvals = [
            expval_from_counts(
                self._backend.run(circ, shots=nshots).result().get_counts()
            )
            for circ, nshots in zip(circuits, nums_shots)
        ]

        for i in cid_withour_meas:
            expvals.insert(i, 1.0)

        expvals = np.array(expvals, dtype=np.float32).reshape(quantum_tensor.shape)
        return qtn.Tensor(expvals, inds=quantum_tensor.inds)


class SimulatorQPUManager(SingleBackendQPUManager):
    def __init__(self, gpu: bool = False) -> None:
        sim = AerSimulator(method="statevector", device="GPU" if gpu else "CPU")
        # sim.set_option("cusvaer_enable", False)
        super().__init__(sim)


def expval_from_counts(counts: dict[str, int]) -> float:
    expval = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        parity = 1 - 2 * int(bitstring.count("1") % 2)
        expval += parity * (count / shots)
    return expval
