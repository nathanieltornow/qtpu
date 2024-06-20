import abc
import time

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

from qtpu.quasi_distr import QuasiDistr


class QuantumInterface(abc.ABC):
    @abc.abstractmethod
    def run(
        self, circuits: list[QuantumCircuit], shots: list[int]
    ) -> list[QuasiDistr]: ...


class BackendInterface(QuantumInterface):
    def __init__(self, backend: BackendV2 | None = None, optimization_level: int = 3):
        self.backend = backend or AerSimulator()
        self.optimization_level = optimization_level

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots_per_circuit: list[int],
    ) -> list[QuasiDistr]:
        assert len(circuits) == len(shots_per_circuit)

        cid_withour_meas = [
            (i, s)
            for i, (circ, s) in enumerate(zip(circuits, shots_per_circuit))
            if circ.count_ops().get("measure", 0) == 0
        ]

        for i, _ in reversed(cid_withour_meas):
            circuits.pop(i)
            shots_per_circuit.pop(i)

        circuits = [
            transpile(
                circ, backend=self.backend, optimization_level=self.optimization_level
            )
            for circ in circuits
        ]
        jobs = [
            self.backend.run(circuit, shots=shot)
            for circuit, shot in zip(circuits, shots_per_circuit)
        ]
        counts = [job.result().get_counts() for job in jobs]

        for i, s in cid_withour_meas:
            counts.insert(i, {"0": s})

        return [QuasiDistr.from_counts(count) for count in counts]


class DummyQuantumInterface(QuantumInterface):
    def run(self, circuits: list[QuantumCircuit], shots: list[int]) -> list[QuasiDistr]:
        depths = [circuit.depth() * s for circuit, s in zip(circuits, shots)]

        # each layer of depth takes 10 ns
        sleeptime = sum(depths) * 10e-9
        print(f"Sleeping for {sleeptime} seconds")
        time.sleep(sleeptime)

        return [QuasiDistr({0: s // 2, 1: s // 2}) for s in shots]
