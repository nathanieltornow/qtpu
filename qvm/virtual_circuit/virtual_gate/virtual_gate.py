from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from qvm.circuit.operation import BinaryGate, UnaryGate, Measurement
from qvm.result import Result

Configuration = List[Union[UnaryGate, Measurement]]


class VirtualGate(BinaryGate, ABC):
    clbit: int

    def __init__(
        self,
        binary_gate: BinaryGate,
        clbit: int,
    ) -> None:
        self.clbit = clbit
        super().__init__(
            binary_gate.name, binary_gate.qubit1, binary_gate.qubit2, binary_gate.params
        )

    @abstractmethod
    def configurations(self) -> List[Configuration]:
        pass

    @abstractmethod
    def knit(self, results: List[Result]) -> Result:
        pass
