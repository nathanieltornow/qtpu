from qiskit.circuit import QuantumCircuit, Qubit


from qvm.dag import DAG


def dag_to_asp(dag: DAG) -> str:
    qubits = dag.qubits
    asp = ""
    for node in dag.nodes:
        asp += f"gate({node}).\n"
        qubits = dag.get_node_instr(node).qubits
        for qubit in dag.get_node_instr(node).qubits:
            asp += f"gate_on_qubit({node}, {dag.qubits.index(qubit)}).\n"

        for next_node in dag.successors(node):
            next_qubits = dag.get_node_instr(next_node).qubits
            same_qubits = set(qubits) & set(next_qubits)
            if len(same_qubits) == 0:
                raise Exception("No common qubits")
            for qubit in same_qubits:
                asp += f"wire({dag.qubits.index(qubit)}, {node}, {next_node}).\n"
    return asp
