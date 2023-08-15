from qiskit.circuit import QuantumCircuit

from qvm.virtual_circuit import VirtualCircuit
from qvm.compiler.virtualization import (
    OptimalDecompositionPass,
    GreedyDependencyBreaker,
)
from qvm.compiler.distr_transpiler import QubitReuser
from qvm.compiler.types import DistributedTranspilerPass, VirtualizationPass


class QVMCompiler:
    def __init__(
        self,
        virt_passes: list[VirtualizationPass] | None = None,
        dt_passes: list[DistributedTranspilerPass] | None = None,
    ):
        self._virt_passes = virt_passes or []
        self._distributed_transpilers = dt_passes or []

    def run(self, circuit: QuantumCircuit, budget: int) -> VirtualCircuit:
        circuit = circuit.copy()
        for vpass in self._virt_passes:
            circuit = vpass.run(circuit, budget)

        virt_circuit = VirtualCircuit(circuit)
        for dtpass in self._distributed_transpilers:
            dtpass.run(virt_circuit)
        return virt_circuit


class StandardQVMCompiler(QVMCompiler):
    def __init__(self, size_to_reach: int) -> None:
        super().__init__(
            virt_passes=[
                OptimalDecompositionPass(size_to_reach),
                GreedyDependencyBreaker(),
            ],
            dt_passes=[QubitReuser(size_to_reach)],
        )
