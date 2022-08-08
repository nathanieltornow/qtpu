from abc import ABC
from typing import Optional, Tuple


class Operation(ABC):
    pass


class Gate(Operation, ABC):
    name: str
    params: Optional[Tuple[float, ...]]

    def __init__(self, name: str, params: Optional[Tuple[float, ...]] = None) -> None:
        self.name = name
        self.params = params


class UnaryGate(Gate):
    qubit: int

    def __init__(
        self, name: str, qubit: int, params: Optional[Tuple[float, ...]] = None
    ) -> None:
        self.qubit = qubit
        super().__init__(name=name, params=params)


class BinaryGate(Gate):
    qubit1: int
    qubit2: int

    def __init__(
        self,
        name: str,
        qubit1: int,
        qubit2: int,
        params: Optional[Tuple[float, ...]] = None,
    ) -> None:
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        super().__init__(name, params)


class Measurement(Operation):
    qubit: int
    clbit: int

    def __init__(self, qubit: int, clbit: int) -> None:
        self.qubit = qubit
        self.clbit = clbit
