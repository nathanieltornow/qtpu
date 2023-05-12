from qiskit.circuit import Barrier, Clbit, QuantumCircuit, QuantumRegister


class PlaceholderGate(Barrier):
    def __init__(self, key: str, clbit: Clbit | None = None):
        super().__init__(num_qubits=1, label=key)
        self.key = key
        self.clbit = clbit


Argument = dict[str, QuantumCircuit]

Fragment = QuantumRegister
