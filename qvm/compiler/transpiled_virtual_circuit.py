from qiskit.providers import Backend

from qvm.circuit import VirtualCircuit


class TranspiledVirtualCircuit(VirtualCircuit):
    backend: Backend
