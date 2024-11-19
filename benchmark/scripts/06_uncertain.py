from qiskit.circuit import QuantumCircuit

from qiskit import transpile

# circuit = QuantumCircuit.from_qasm_file(
#     "benchmark/qasm/square-heisenberg/square_heisenberg_N16.qasm"
# )
# circuit = QuantumCircuit.from_qasm_file(
#     "benchmark/qasm/qaoa/qaoa_barabasi_albert_N15_3reps.qasm"
# )

file = "benchmark/qasm/qasmbench-medium/cat_state_n22/cat_state_n22.qasm"
circuit = QuantumCircuit.from_qasm_file(file)


def simplify_rzz(circuit: QuantumCircuit) -> QuantumCircuit:
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    i = 0
    while i < len(circuit.data):

        if i + 2 >= len(circuit.data):
            new_circuit.append(
                circuit.data[i].operation,
                circuit.data[i].qubits,
                circuit.data[i].clbits,
            )
            i += 1
            continue

        op1, qubits1 = circuit.data[i].operation, circuit.data[i].qubits
        op2, qubits2 = circuit.data[i + 1].operation, circuit.data[i + 1].qubits
        op3, qubits3 = circuit.data[i + 2].operation, circuit.data[i + 2].qubits

        if (
            len(qubits1) == 2
            and qubits1 == qubits3
            and qubits1[1] == qubits2[0]
            and op1.name == "cx"
            and op2.name == "rz"
            and op3.name == "cx"
        ):
            new_circuit.rzz(op2.params[0], qubits1[0], qubits1[1])
            i += 3
        else:
            new_circuit.append(op1, qubits1, circuit.data[i].clbits)
            i += 1

    return new_circuit


# circuit = simplify_rzz(circuit)

from qiskit_aer.primitives import EstimatorV2
from qiskit_aer import AerSimulator

estimator = EstimatorV2()


# print(circuit)
# print(circuit)
# circuit.remove_final_measurements(inplace=True)
N = 10
circuit = QuantumCircuit(N)
circuit.h(0)
circuit.cx(range(0, N - 1), range(1, N))

# print(circuit)
res = estimator.run([(circuit, "Z" * circuit.num_qubits)]).result()
print(res[0].data.evs)

# print(circuit)

from benchmark.exec_ckt import cut_ckt
from qtpu.compiler.compiler import compile_reach_size
from qtpu.contract import contract
from qtpu.circuit import circuit_to_hybrid_tn, cuts_to_moves
from benchmark.util import get_info
from benchmark.exec_qtpu import qtpu_execute


# print(expval_quimb(circuit))


circuit.measure_all()


cut_circ = compile_reach_size(
    circuit, 5, show_progress_bar=True, n_trials=100, compression_methods=["2q"]
)

print(qtpu_execute(cut_circ))

# cut_circ = cut_ckt(circuit, 15)
print(get_info(cut_circ))
# cut_circ = cuts_to_moves(cut_circ)
# # # print(cut_circ)

# cut_circ.draw(output="mpl", filename="circuit.png")

htn = circuit_to_hybrid_tn(cut_circ)
res = contract(htn)
print(res)

from qiskit.primitives import Sampler, Estimator


ref_circ = circuit.copy()
ref_circ.remove_final_measurements()
ref_expval = Estimator().run(ref_circ, ["Z" * N]).result().values[0]
print(ref_expval)
