import numpy as np
import networkx as nx

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library import TwoLocal, QFT
from qiskit.circuit.library import standard_gates


def ghz(num_qubits: int):
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    circuit.cx(list(range(num_qubits - 1)), list(range(1, num_qubits)))
    circuit.measure_all()
    return circuit


def twolocal(num_qubits: int, reps: int) -> QuantumCircuit:
    num_qubits = num_qubits
    circuit = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz", "rx"],
        entanglement="linear",
        entanglement_blocks="rzz",
        reps=reps,
    )
    circuit.measure_all()
    circuit = circuit.decompose()
    params = [(np.random.uniform(0.0, np.pi)) for _ in range(len(circuit.parameters))]
    return circuit.bind_parameters(params)


def qft(num_qubits: int, approx_degree: int = 0) -> QuantumCircuit:
    circuit = QFT(num_qubits, approximation_degree=approx_degree, do_swaps=False)
    circuit.measure_all()
    return circuit.decompose()


def qaoa(graph: nx.Graph) -> QuantumCircuit:
    """Source https://qiskit.org/textbook/ch-applications/qaoa.html"""
    nqubits = len(graph.nodes())
    circuit = QuantumCircuit(nqubits)
    beta = np.random.uniform(0, np.pi)
    gamma = np.random.uniform(0, np.pi)

    for i in range(0, nqubits):
        circuit.h(i)
    for u, v in list(graph.edges()):
        circuit.rzz(2 * gamma, u, v)
    for i in range(0, nqubits):
        circuit.rx(2 * beta, i)

    circuit.measure_all()
    return circuit


def random_circuit(num_qubits, depth, seed=None):
    if num_qubits == 0:
        return QuantumCircuit()

    gates_1q = [
        (standard_gates.XGate, 1, 0),
        (standard_gates.RZGate, 1, 1),
        (standard_gates.HGate, 1, 0),
        (standard_gates.RXGate, 1, 1),
        (standard_gates.RYGate, 1, 1),
    ]
    gates_2q = [
        (standard_gates.CXGate, 2, 0),
        (standard_gates.CPhaseGate, 2, 1),
        (standard_gates.CYGate, 2, 0),
        (standard_gates.CZGate, 2, 0),
        (standard_gates.RZZGate, 2, 1),
    ]

    gates = gates_1q.copy()
    gates.extend(gates_2q)
    gates = np.array(
        gates,
        dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)],
    )
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(num_qubits)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Apply arbitrary random operations in layers across all qubits.
    for _ in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
        gate_specs = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)
        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack:
            gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

        # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomises those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        for gate, q_start, q_end, p_start, p_end in zip(
            gate_specs["class"],
            q_indices[:-1],
            q_indices[1:],
            p_indices[:-1],
            p_indices[1:],
        ):
            operation = gate(*parameters[p_start:p_end])
            qc._append(
                CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end])
            )

    qc.measure_all()

    return qc


def dj_oracle(case: str, n: int) -> QuantumCircuit:
    # plus one output qubit
    oracle_qc = QuantumCircuit(n + 1)

    if case == "balanced":
        np.random.seed(10)
        b_str = ""
        for _ in range(n):
            b = np.random.randint(0, 2)
            b_str = b_str + str(b)

        for qubit in range(len(b_str)):
            if b_str[qubit] == "1":
                oracle_qc.x(qubit)

        for qubit in range(n):
            oracle_qc.cx(qubit, n)

        for qubit in range(len(b_str)):
            if b_str[qubit] == "1":
                oracle_qc.x(qubit)

    if case == "constant":
        output = np.random.randint(2)
        if output == 1:
            oracle_qc.x(n)

    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle"  # To show when we display the circuit
    return oracle_gate


def dj_algorithm(oracle: QuantumCircuit, n: int) -> QuantumCircuit:
    dj_circuit = QuantumCircuit(n + 1, n)

    dj_circuit.x(n)
    dj_circuit.h(n)

    for qubit in range(n):
        dj_circuit.h(qubit)

    dj_circuit.append(oracle, range(n + 1))

    for qubit in range(n):
        dj_circuit.h(qubit)

    dj_circuit.barrier()
    for i in range(n):
        dj_circuit.measure(i, i)

    return dj_circuit


def dj(n: int, balanced: bool = True) -> QuantumCircuit:
    """Returns a quantum circuit implementing the Deutsch-Josza algorithm.
    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    balanced -- True for a balanced and False for a constant oracle
    """

    oracle_mode = "balanced" if balanced else "constant"
    n = n - 1  # because of ancilla qubit
    oracle_gate = dj_oracle(oracle_mode, n)
    qc = dj_algorithm(oracle_gate, n)
    qc.name = "dj"

    return qc


if __name__ == "__main__":
    import os
    import networkx as nx
    
    os.makedirs(f"qasm/twolocal", exist_ok=True)
    for i in range(20, 101, 10):
        with open(f"qasm/twolocal/2_{i}.qasm", "w") as f:
            f.write(twolocal(i, 2).decompose().qasm())
