import itertools
from abc import ABC, abstractmethod
from typing import List, Optional

from qiskit.circuit import Barrier, QuantumCircuit

from vqc.prob_distr import Counts, ProbDistr


class VirtualGate(Barrier, ABC):

    _ids = itertools.count(0)

    def __init__(self, params: Optional[List] = None):
        super().__init__(2)
        if params is None:
            params = []
        self.id = next(self._ids)
        self._params = params

    def __eq__(self, other):
        return super().__eq__(other) and self.id == other.id

    def __hash__(self):
        return self.id

    @abstractmethod
    def configure(self) -> List[QuantumCircuit]:
        pass

    @abstractmethod
    def knit(self, results: List[ProbDistr]) -> ProbDistr:
        pass

    def configuration(self, config_id: int) -> QuantumCircuit:
        return self.configure()[config_id]


class Executor(ABC):
    @abstractmethod
    def execute(
        self, sampled_circuits: dict[str, list[QuantumCircuit]]
    ) -> dict[str, list[Counts]]:
        pass
