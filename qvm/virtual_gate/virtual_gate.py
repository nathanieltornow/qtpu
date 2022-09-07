from abc import ABC, abstractmethod
from cProfile import label
import itertools
from re import A
from typing import List, Optional, Tuple, Type
from qiskit.circuit import QuantumCircuit, Instruction, Barrier

from qvm.prob import ProbDistribution


class VirtualBinaryGate(Barrier, ABC):

    _ids = itertools.count(0)
    params: List

    def __init__(self, params: Optional[List] = None):
        if params is None:
            params = []
        self.id = next(self._ids)
        self.params = params
        super().__init__(2)

    def __eq__(self, other):
        return super().__eq__(other) and self.id == other.id

    def __hash__(self):
        return self.id

    @abstractmethod
    def original_gate_type(self) -> Type[Instruction]:
        """
        Returns the type of the original gate.
        """
        pass

    @abstractmethod
    def configure(self) -> List[QuantumCircuit]:
        pass

    @abstractmethod
    def knit(self, results: List[ProbDistribution]) -> ProbDistribution:
        pass

    def configuration(self, config_id: int) -> QuantumCircuit:
        return self.configure()[config_id]

    def partial_config(self, config_id: int) -> Tuple[QuantumCircuit, QuantumCircuit]:
        conf = self.configuration(config_id)
        qubit0, qubit1 = tuple(conf.qubits)
        conf0 = QuantumCircuit(1, 1, name="conf")
        conf1 = QuantumCircuit(1, 1, name="conf")
        for circ_inst in conf:
            if circ_inst.qubits == (qubit0,):
                conf0.append(
                    circ_inst.operation,
                    (conf0.qubits[0],),
                    (circ_inst.clbits[0],) if len(circ_inst.clbits) > 0 else (),
                )
            elif circ_inst.qubits == (qubit1,):
                conf1.append(
                    circ_inst.operation,
                    (conf1.qubits[0],),
                    (circ_inst.clbits[0],) if len(circ_inst.clbits) > 0 else (),
                )
        conf0.i(0)
        conf0.i(0)
        conf1.i(0)
        conf1.i(0)
        return conf0, conf1

    def _define(self):
        qc = QuantumCircuit(2)
        qc.append(self.original_gate_type(*self.params), [0, 1], [])
        self._definition = qc
