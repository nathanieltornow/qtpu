import itertools

import networkx as nx
from qiskit.circuit import QuantumCircuit, Barrier, QuantumRegister, Qubit

from qvm.dag import DAG


def initial_layout_from_transpiled_circuit(circuit: QuantumCircuit) -> list[int]:
    layout = circuit._layout
    if layout is None:
        raise ValueError("Circuit has no layout")
    return layout.get_virtual_bits()


def dag_to_qcg(dag: DAG, use_qubit_idx: bool = False) -> nx.Graph:
    graph = nx.Graph()
    bb = nx.edge_betweenness_centrality(graph, normalized=False)
    nx.set_edge_attributes(graph, bb, "weight")
    if use_qubit_idx:
        graph.add_nodes_from(range(len(dag.qubits)))
    else:
        graph.add_nodes_from(dag.qubits)

    for node in dag.nodes:
        cinstr = dag.get_node_instr(node)
        op, qubits = cinstr.operation, cinstr.qubits
        if isinstance(op, Barrier):
            continue
        if len(qubits) >= 2:
            for qubit1, qubit2 in itertools.combinations(qubits, 2):
                if use_qubit_idx:
                    qubit1, qubit2 = dag.qubits.index(qubit1), dag.qubits.index(qubit2)

                if not graph.has_edge(qubit1, qubit2):
                    graph.add_edge(qubit1, qubit2, weight=0)
                graph[qubit1][qubit2]["weight"] += 1
    return graph


def fragment_dag(dag: DAG) -> None:
    con_qubits = list(
        nx.connected_components(
            dag_to_qcg(
                dag,
            )
        )
    )
    new_frags = [
        QuantumRegister(len(qubits), name=f"frag{i}")
        for i, qubits in enumerate(con_qubits)
    ]
    qubit_map: dict[Qubit, Qubit] = {}  # old -> new Qubit
    for nodes, circ in zip(con_qubits, new_frags):
        node_l = list(nodes)
        for i in range(len(node_l)):
            qubit_map[node_l[i]] = circ[i]

    for node in dag.nodes:
        instr = dag.get_node_instr(node)
        instr.qubits = [qubit_map[qubit] for qubit in instr.qubits]
