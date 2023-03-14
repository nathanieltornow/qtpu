

from benchmarking import HamiltonianSimulationBenchmark
from qvm.cut_library.layout import fit_to_coupling_basic_virtualization
from qiskit.transpiler import CouplingMap
from qiskit.compiler import transpile

bench = HamiltonianSimulationBenchmark(5, 2)

circuit = bench.circuit()

coupling_map = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [2, 4], [4, 2]])

circuit = transpile(circuit, basis_gates=["rz", "rx", "rzz"])

circuit = fit_to_coupling_basic_virtualization(circuit, coupling_map, 4)
# circuit = transpile(circuit, coupling_map=coupling_map, optimization_level=3)
print(circuit)

num_swaps = sum(1 for gate in circuit if gate.operation.name == "swap")
print(num_swaps)