from typing import Dict, List, Optional
from qiskit.circuit import QuantumCircuit
from qvm.execution.exec import execute_fragmented_circuit
from qvm.transpiler.fragmented_circuit import FragmentedCircuit

from qvm.transpiler.transpiler import DistributedPass, VirtualizationPass


def transpile(
    circuit: QuantumCircuit,
    virt_passes: Optional[List[VirtualizationPass]] = None,
    distr_passes: Optional[List[DistributedPass]] = None,
) -> FragmentedCircuit:
    if virt_passes is None:
        virt_passes = []
    if distr_passes is None:
        distr_passes = []

    t_circ = circuit.copy()
    for vpass in virt_passes:
        vpass.run(t_circ)
    frag_circ = FragmentedCircuit(t_circ)
    for dpass in distr_passes:
        dpass.run(frag_circ)
    return frag_circ


def execute(frag_circ: FragmentedCircuit, shots: int = 10000) -> Dict[str, int]:
    return execute_fragmented_circuit(frag_circ, shots)
