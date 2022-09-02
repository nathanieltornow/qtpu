from typing import Dict, List, Optional
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qvm.execution.exec import execute_fragmented_circuit
from qvm.transpiler.fragmented_circuit import FragmentedCircuit

from qvm.transpiler.transpiler import DistributedPass, VirtualizationPass
from qvm.virtual_gate.virtual_gate import VirtualBinaryGate


def transpile(
    circuit: QuantumCircuit,
    virt_passes: Optional[List[VirtualizationPass]] = None,
    distr_passes: Optional[List[DistributedPass]] = None,
) -> FragmentedCircuit:
    if virt_passes is None:
        virt_passes = []
    if distr_passes is None:
        distr_passes = []

    t_dag = circuit_to_dag(circuit.copy())
    for vpass in virt_passes:
        vpass.run(t_dag)
    frag_circ = FragmentedCircuit(dag_to_circuit(t_dag).decompose([VirtualBinaryGate]))
    print(frag_circ)
    for dpass in distr_passes:
        dpass.run(frag_circ)
    return frag_circ


def execute(frag_circ: FragmentedCircuit, shots: int = 10000) -> Dict[str, int]:
    return execute_fragmented_circuit(frag_circ, shots)
