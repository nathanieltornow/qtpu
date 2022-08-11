from typing import Set

import networkx as nx

from qvm.circuit.circuit import Circuit
from qvm.virtual_circuit.fragment import Fragment


class VirtualCircuit(Circuit):
    fragments: Set[Fragment]

    def __init__(self, circuit: Circuit) -> None:
        super().__init__(
            *circuit.operations,
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits
        )

