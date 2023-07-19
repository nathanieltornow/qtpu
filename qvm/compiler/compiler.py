from qiskit.circuit import QuantumCircuit

from qvm.virtual_circuit import VirtualCircuit
from .types import VirtualCircuitCompiler, CutCompiler


class QVMCompiler:
    def __init__(
        self,
        cut_compiler: CutCompiler | None = None,
        vc_compiler: VirtualCircuitCompiler | None = None,
    ):
        self._cut_compiler = cut_compiler
        self._vc_compiler = vc_compiler

    def run(self, circuit: QuantumCircuit) -> VirtualCircuit:
        cut_circuit = circuit.copy()
        if self._cut_compiler is not None:
            cut_circuit = self._cut_compiler.run(cut_circuit)
        vc = VirtualCircuit(cut_circuit)

        if self._vc_compiler is not None:
            self._vc_compiler.run(vc)
        return vc
