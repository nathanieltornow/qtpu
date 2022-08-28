from typing import Dict, Optional, Sequence, Set, Tuple, Union
from qiskit.circuit.quantumcircuit import (
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    Register,
    Bit,
    ParameterValueType,
)
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator

import networkx as nx

from qvm.converters import circuit_to_connectivity_graph


class Fragment(QuantumCircuit):
    backend: Backend

    def __init__(
        self,
        *regs: Union[Register, int, Sequence[Bit]],
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[Dict] = None,
        backend: Optional[Backend] = None,
    ):
        super().__init__(*regs, name=name, global_phase=global_phase, metadata=metadata)
        if backend is None:
            backend = AerSimulator()
        self.backend = backend

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Fragment) and self.name == other.name


class FragmentedCircuit:
    _fragments: Set[Fragment]

    def __init__(self, circuit: QuantumCircuit):
        circuit = circuit.decompose()
        con_graph = circuit_to_connectivity_graph(circuit)

        node_sets = list(nx.connected_components(con_graph))
        new_frags = [
            Fragment(QuantumRegister(len(nodes)), *circuit.cregs, name=f"frag_{i}")
            for i, nodes in enumerate(node_sets)
        ]
        qubit_map: Dict[Qubit, Tuple[Fragment, Qubit]] = {}  # old -> new Qubit
        for nodes, circ in zip(node_sets, new_frags):
            node_l = list(nodes)
            for i in range(len(node_l)):
                qubit_map[node_l[i]] = (circ, circ.qubits[i])

        for circ_instr in circuit.data:
            circ = qubit_map[circ_instr.qubits[0]][0]
            circ.append(
                circ_instr.operation,
                [qubit_map[q][1] for q in circ_instr.qubits],
                circ_instr.clbits,
            )
        self._fragments = set(new_frags)

    def __str__(self) -> str:
        frag_l = list(self._fragments)
        circstr = f"{frag_l[0].name}\n\n" + str(frag_l[0])
        for circ in frag_l[1:]:
            circstr += f"\n\n\n {circ.name}\n\n" + str(circ)
        return circstr

    @property
    def fragments(self) -> Set[QuantumCircuit]:
        # shallow copy of fragments since they should not be modified
        return self._fragments
