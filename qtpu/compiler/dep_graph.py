from itertools import combinations

import networkx as nx
from qiskit.circuit import QuantumCircuit
from circuit_knitting.cutting.qpd import QPDBasis
from circuit_knitting.cutting.instructions import Move

from qtpu.circuit import insert_cuts


def circuit_to_dep_graph(circuit: QuantumCircuit) -> nx.DiGraph:
    graph = nx.DiGraph()
    qubit_to_node = {}

    node_id = 0
    for op_idx, instr in enumerate(circuit):
        for qubit in instr.qubits:
            graph.add_node(
                node_id,
                op_idx=op_idx,
                abs_qubit=circuit.qubits.index(qubit),
                rel_qubit=instr.qubits.index(qubit),
            )
            if qubit in qubit_to_node:
                graph.add_edge(
                    qubit_to_node[qubit],
                    node_id,
                    weight=1,  # weight=QPDBasis.from_instruction(Move()).overhead,
                )
            qubit_to_node[qubit] = node_id
            node_id += 1

        num_qubits = len(instr.qubits)
        if num_qubits == 1:
            continue

        # weight = (
        #     QPDBasis.from_instruction(instr.operation).overhead
        #     if num_qubits == 2
        #     else 1e15  # large number so we don't cut gates with > 2 qubits
        # )
        weight = 1

        for i in range(num_qubits - 1):
            u, v = qubit_to_node[instr.qubits[i]], qubit_to_node[instr.qubits[i + 1]]
            graph.add_edge(u, v, weight=weight)
            graph.add_edge(v, u, weight=weight)

    return graph


def is_wire_edge(graph: nx.DiGraph, u: int, v: int) -> tuple[int, int] | None:
    return (
        (graph.nodes[u]["op_idx"], graph.nodes[u]["rel_qubit"])
        if graph.has_edge(u, v)
        and graph.nodes[u]["abs_qubit"] == graph.nodes[v]["abs_qubit"]
        else None
    )


def is_gate_edge(graph: nx.DiGraph, u: int, v: int) -> int | None:
    return (
        graph.nodes[u]["op_idx"]
        if graph.has_edge(u, v) and graph.nodes[u]["op_idx"] == graph.nodes[v]["op_idx"]
        else None
    )


def uncycle(graph: nx.DiGraph) -> tuple[nx.DiGraph, dict[int, set[int]]]:
    edges = list(graph.edges)
    next_node_id = len(graph.nodes)

    op_idx_to_nodes = {}

    for u, v in edges:
        if not is_gate_edge(graph, u, v) or not graph.has_edge(u, v):
            continue

        graph.remove_edge(u, v)
        graph.remove_edge(v, u)

        graph.add_node(next_node_id, **graph.nodes[u])
        graph.add_node(next_node_id + 1, **graph.nodes[v])

        for succ in graph.successors(u):
            graph.add_edge(next_node_id, succ)
        for succ in graph.successors(v):
            graph.add_edge(next_node_id + 1, succ)

        graph.add_edge(v, next_node_id)
        graph.add_edge(u, next_node_id + 1)

        op_idx_to_nodes[graph.nodes[u]["op_idx"]] = {next_node_id, next_node_id + 1}
        next_node_id += 2

    assert nx.is_directed_acyclic_graph(graph)

    return graph, op_idx_to_nodes


def remove_critical_gates(
    circuit: QuantumCircuit, max_overhead: float, num_edges: int = 1
) -> QuantumCircuit:
    graph = circuit_to_dep_graph(circuit)
    uncycled_graph, op_idx_to_nodes = uncycle(graph)

    gate_cuts = set()
    wire_cuts = set()

    overhead = 0
    while overhead <= max_overhead:
        impact_to_edges = {}
        for u, v in uncycled_graph.edges:
            impact = (len(nx.ancestors(uncycled_graph, u)) + 1) * (
                len(nx.descendants(uncycled_graph, v)) + 1
            )

            if impact not in impact_to_edges:
                impact_to_edges[impact] = []
            impact_to_edges[impact].append((u, v))

        max_impact = max(impact_to_edges.keys())
        u, v = impact_to_edges[max_impact].pop()
        if op_idx := is_gate_edge(uncycled_graph, u, v):
            gate_cuts.add(op_idx)
            uncycled_graph.remove_nodes_from(op_idx_to_nodes[op_idx])
            overhead += 1
        elif wire := is_wire_edge(uncycled_graph, u, v):
            wire_cuts.add(wire)
            uncycled_graph.remove_edge(u, v)
            overhead += 1
        else:
            raise ValueError(
                f"Invalid edge type {uncycled_graph.nodes[u]} | {uncycled_graph.nodes[v]}"
            )

    return insert_cuts(circuit, gate_cuts, wire_cuts)


def reduce_deps_greedy(
    circuit: QuantumCircuit, max_overhead: float, num_edges: int = 1
) -> QuantumCircuit:
    graph = circuit_to_dep_graph(circuit)

    gate_cuts = set()
    wire_cuts = set()

    overhead = 0
    while overhead <= max_overhead:
        print(overhead, max_overhead)
        betweenness_sorted = sorted(
            nx.edge_betweenness_centrality(graph).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        betweenness_sorted = [
            ((u, v), w) for (u, v), w in betweenness_sorted if is_gate_edge(graph, u, v)
        ]

        relevant_edges = [
            (u, v, graph[u][v]["weight"])
            for (u, v), _ in betweenness_sorted[:num_edges]
        ]

        u, v, w = min(relevant_edges, key=lambda x: x[2])

        print(graph.nodes[u], graph.nodes[v])

        if is_wire_edge(graph, u, v):
            wire_cuts.add((graph.nodes[u]["op_idx"], graph.nodes[u]["rel_qubit"]))
        elif is_gate_edge(graph, u, v):
            gate_cuts.add(graph.nodes[u]["op_idx"])
        else:
            raise ValueError(f"Invalid edge type {graph.nodes[u]} | {graph.nodes[v]}")

        graph.remove_edge(u, v)
        if graph.has_edge(v, u):
            graph.remove_edge(v, u)

        overhead += w

    return insert_cuts(circuit, gate_cuts, wire_cuts)
