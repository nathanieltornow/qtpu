import abc
from multiprocessing.pool import Pool

from qiskit.circuit import QuantumCircuit

from qvm.quasi_distr import QuasiDistr
from qvm.types import Argument, Fragment
from qvm.util import fragment_circuit


class Virtualizer(abc.ABC):
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit

    def fragments(self) -> dict[Fragment, QuantumCircuit]:
        sub_circs: dict[Fragment, QuantumCircuit] = {
            frag: QuantumCircuit(frag, *self._circuit.cregs)
            for frag in self._circuit.qregs
        }
        for cinstr in self._circuit.data:
            op, qubits, clbits = cinstr.operation, cinstr.qubits, cinstr.clbits
            appended = False
            for frag in self._circuit.qregs:
                if set(qubits) <= set(frag):
                    sub_circs[frag].append(op, qubits, clbits)
                    appended = True
                    break
            assert appended or op.name == "barrier"
        return sub_circs

    @abc.abstractmethod
    def instantiate(self) -> dict[Fragment, list[Argument]]:
        ...

    @abc.abstractmethod
    def knit(self, results: dict[Fragment, list[QuasiDistr]], pool: Pool) -> QuasiDistr:
        ...

