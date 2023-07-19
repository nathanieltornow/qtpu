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
from circuits import get_circuits

QPU_SIZE = 5


@dataclass
class OverheadComparison:
    name: str
    num_vgates: int
    depth_vgates: int
    num_wire_cuts: int
    depth_wire_cuts: int
    num_vgates_qr: int
    depth_vagtes_qr: int

    def append_to_csv(self, filepath: str) -> None:
        append_dict_to_csv(filepath, asdict(self))


def _num_wire_cuts(circuit: QuantumCircuit, qpu_size: int) -> VirtualCircuit:
    print("cutting wires...")
    cutter = OptimalWireCutter(qpu_size)
    cut_circ = cutter.run(circuit)
    vc = VirtualCircuit(cut_circ)
    while any(circ.num_qubits > qpu_size for circ in vc.fragment_circuits.values()):
        qpu_size -= 1
        cutter = OptimalWireCutter(qpu_size)
        cut_circ = cutter.run(circuit)
        vc = VirtualCircuit(cut_circ)
    return vc


def _num_vgates(circuit: QuantumCircuit, qpu_size: int) -> VirtualCircuit:
    print("cutting vgates...")
    cutter = OptimalGateDecomposer(qpu_size)
    cut_circ = cutter.run(circuit)
    return VirtualCircuit(cut_circ)


def _num_vgates_qr(circuit: QuantumCircuit, qpu_size: int) -> VirtualCircuit:
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
    return virt


from qiskit.providers.fake_provider import FakePerth
from qiskit.compiler import transpile


def virt_depth(virt: VirtualCircuit) -> int:
    m = max(
        transpile(circ, backend=FakePerth(), optimization_level=3).depth(lambda x: True)
        for circ in virt.fragment_circuits.values()
    )

    return m


def run_bench():
    benches = [
        "hamsim_2",
        "hamsim_3",
        "twolocal_1",
        "twolocal_2",
        # "twolocal_3",
        "vqe_2",
        "vqe_3",
        "qaoa_r2",
        "qaoa_r3",
        "qaoa_r4",
        # "qaoa_ba1",
        # "qaoa_ba2",
        # "qaoa_ba3",
        # "qaoa_ba4",
        "qsvm",
        "wstate",
        "ghz",
    ]

    result_file = "bench/results/overhead8.csv"

    for bench in benches:
        circuit = get_circuits(bench, (8, 9))[0]
        v1 = _num_vgates(circuit, QPU_SIZE)
        v2 = _num_wire_cuts(circuit, QPU_SIZE)
        # v3 = _num_vgates_qr(circuit, 7)

        append_dict_to_csv(
            result_file,
            {
                "name": bench,
                "num_vgates": len(v1._vgate_instrs),
                "vgate_depth": virt_depth(v1),
                "num_wires": len(v2._vgate_instrs),
                "wire_depth": virt_depth(v2),
                # "num_vgates_qr": len(v3._vgate_instrs),
                # "vgate_qr_depth": virt_depth(v3),
            },
        )


# def run_overhead_comparison(
#     csv_file: str, circuits: list[QuantumCircuit], qpu_size: int
# ) -> None:
#     progress = tqdm(total=len(circuits))
#     progress.set_description("Running Overhead Comparison")
#     for circ in circuits:
#         overhead = _overhead_comparison(circ, qpu_size)
#         overhead.append_to_csv(csv_file)
#         progress.update(1)


# def two_local_comp(layers: int):
#     circuits = [two_local(13, layers)]
#     run_overhead_comparison(
#         f"bench/results/cut_comp/two_local_{layers}.csv", circuits, QPU_SIZE
#     )


# def hamsim_comp(layers: int):
#     circuits = [hamsim(13, layers)]
#     run_overhead_comparison(
#         f"bench/results/cut_comp/hamsim_{layers}.csv", circuits, QPU_SIZE
#     )


# def qaoa_comp(degree: int):
#     circuits = [qaoa(nx.random_regular_graph(degree, 14))]
#     run_overhead_comparison(
#         f"bench/results/cut_comp/qaoa_{degree}.csv", circuits, QPU_SIZE
#     )


if __name__ == "__main__":
    # hamsim_comp(1)
    # hamsim_comp(2)
    # hamsim_comp(3)
    # two_local_comp(1)
    # two_local_comp(2)
    # two_local_comp(3)
    run_bench()
