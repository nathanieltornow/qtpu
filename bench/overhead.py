from dataclasses import dataclass, asdict

import networkx as nx
from qiskit.circuit import QuantumCircuit
from tqdm import tqdm

from qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
from qvm.compiler.virtualization.gate_decomp import OptimalGateDecomposer
from qvm.compiler.virtualization.reduce_deps import QubitDependencyMinimizer
from qvm.compiler.qubit_reuse import QubitReuseCompiler
from qvm.virtual_circuit import VirtualCircuit
from qvm.virtual_gates import VirtualBinaryGate, VirtualMove

from util._util import append_dict_to_csv
from util.circuits import two_local, qaoa, hamsim


QPU_SIZE = 7


@dataclass
class OverheadComparison:
    num_qubits: int
    num_vgates: int
    num_wire_cuts: int
    num_vgates_qr: int

    def append_to_csv(self, filepath: str) -> None:
        append_dict_to_csv(filepath, asdict(self))


def _num_wire_cuts(circuit: QuantumCircuit, qpu_size: int) -> int:
    print("cutting wires...")
    cutter = OptimalWireCutter(qpu_size)
    cut_circ = cutter.run(circuit)
    return sum(1 for instr in cut_circ.data if isinstance(instr.operation, VirtualMove))


def _num_vgates(circuit: QuantumCircuit, qpu_size: int) -> int:
    print("cutting vgates...")
    cutter = OptimalGateDecomposer(qpu_size)
    cut_circ = cutter.run(circuit)
    return sum(
        1 for instr in cut_circ.data if isinstance(instr.operation, VirtualBinaryGate)
    )


def _num_vgates_qr(circuit: QuantumCircuit, qpu_size: int) -> int:
    print("cutting vgates+qr...")
    vgates = 0
    while True:
        print(f"vgates: {vgates}")
        cutter = QubitDependencyMinimizer(vgates)
        cut_circuit = cutter.run(circuit.copy())

        virt = VirtualCircuit(cut_circuit)
        QubitReuseCompiler(qpu_size).run(virt)
        if all(circ.num_qubits <= qpu_size for circ in virt.fragment_circuits.values()):
            break
        vgates += 1
    return vgates


def _overhead_comparison(circuit: QuantumCircuit, qpu_size: int) -> OverheadComparison:
    return OverheadComparison(
        num_qubits=circuit.num_qubits,
        num_vgates=_num_vgates(circuit, qpu_size),
        num_wire_cuts=_num_wire_cuts(circuit, qpu_size),
        num_vgates_qr=_num_vgates_qr(circuit, qpu_size),
    )


def run_overhead_comparison(
    csv_file: str, circuits: list[QuantumCircuit], qpu_size: int
) -> None:
    progress = tqdm(total=len(circuits))
    progress.set_description("Running Overhead Comparison")
    for circ in circuits:
        overhead = _overhead_comparison(circ, qpu_size)
        overhead.append_to_csv(csv_file)
        progress.update(1)


def two_local_comp(layers: int):
    circuits = [two_local(13, layers)]
    run_overhead_comparison(
        f"bench/results/cut_comp/two_local_{layers}.csv", circuits, QPU_SIZE
    )


def hamsim_comp(layers: int):
    circuits = [hamsim(13, layers)]
    run_overhead_comparison(
        f"bench/results/cut_comp/hamsim_{layers}.csv", circuits, QPU_SIZE
    )


def qaoa_comp(degree: int):
    circuits = [qaoa(nx.random_regular_graph(degree, 14))]
    run_overhead_comparison(
        f"bench/results/cut_comp/qaoa_{degree}.csv", circuits, QPU_SIZE
    )


if __name__ == "__main__":
    # hamsim_comp(1)
    # hamsim_comp(2)
    # hamsim_comp(3)
    # two_local_comp(1)
    # two_local_comp(2)
    # two_local_comp(3)
    # qaoa_comp()
    qaoa_comp(2)
    qaoa_comp(3)
    qaoa_comp(4)