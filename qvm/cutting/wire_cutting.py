from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister

from qvm.util import fold_circuit, unfold_circuit, decompose_qubits
from qvm.virtual_gates import WireCut, VirtualSWAP


def cut_wires_optimal(
    circuit: QuantumCircuit,
    num_fragments: int = 2,
    max_cuts: int = 4,
    max_fragment_size: int | None = None,
) -> QuantumCircuit:
    pass
#     from clingo.control import Control
#     import importlib.resources

#     two_qubit_circ, one_qubit_circs = fold_circuit(circuit)

#     asp = _circuit_to_asp(two_qubit_circ)

#     with importlib.resources.path("qvm", "asp") as path:
#         asp_file = path / "dag_partition.lp"
#         asp += asp_file.read_text()
        
#     asp += f"#const num_fragments = {num_fragments}.\n"

#     control = Control()
#     control.configuration.solve.models = 0  # type: ignore
#     control.add("base", [], asp)
#     control.ground([("base", [])])
#     solve_result = control.solve(yield_=True)  # type: ignore
#     opt_model = None
#     for model in solve_result:  # type: ignore
#         opt_model = model
#         print(model)

#     if opt_model is None:
#         raise ValueError("No solution found.")
#     print(opt_model)

    # qubits_sets: list[set[Qubit]] = [set() for _ in range(num_fragments)]
    # for symbol in opt_model.symbols(shown=True):
    #     if symbol != "partition" and len(symbol.arguments) != 2:
    #         continue
    #     qubit_idx, partition = (
    #         symbol.arguments[0].number,
    #         symbol.arguments[1].number,
    #     )
    #     qubits_sets[partition].add(circuit.qubits[qubit_idx])

    # return decompose_qubits(circuit, qubits_sets)


