from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
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

    @staticmethod
    def from_circuit(circuit: QuantumCircuit) -> "Fragment":
        frag = Fragment(
            *circuit.qregs,
            *circuit.cregs,
            name=circuit.name,
            global_phase=circuit.global_phase,
            metadata=circuit.metadata,
        )
        for circ_instr in circuit:
            frag.append(circ_instr)
        return frag

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Fragment) and self.name == other.name


class FragmentedCircuit:
    _fragments: Set[Fragment]

    def __init__(self, circuit: QuantumCircuit):
        con_graph = circuit_to_connectivity_graph(circuit)

        node_sets = list(nx.connected_components(con_graph))
        new_frags = [
            Fragment(QuantumRegister(len(nodes)), *circuit.cregs)
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
        return self._fragments.copy()

    def to_circuit(self) -> QuantumCircuit:
        pass

    def replace_fragment(self, fragment: Fragment, new_circuit: QuantumCircuit) -> None:
        if fragment.cregs != new_circuit.cregs:
            raise ValueError("Cregs must be the same")
        self._fragments.remove(fragment)
        self._fragments.add(Fragment.from_circuit(new_circuit))

    def merge_fragments(self, frag1: Fragment, frag2: Fragment) -> None:
        pass
