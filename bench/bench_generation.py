from itertools import product
from dataclasses import dataclass, asdict

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2

from qvm.compiler import VirtualizationCompiler
from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler
from qvm.compiler.virtualization.general_bisection import GeneralBisectionCompiler

from _circuits import get_circuits


@dataclass
class Benchmark:
    circuits: list[QuantumCircuit]
    backend: BackendV2
    virt_compiler: VirtualizationCompiler
    result_file: str
    base_backend: BackendV2 | None = None


def generate_vqr_benchmarks(
    circ_name: str,
    params: list[int | float],
    backend: BackendV2,
    max_distance: int = 2,
    num_vgates: int = 3,
    reverse_order: bool = False,
    times: int = 3,
) -> list[Benchmark]:
    nums_qubits = sorted(list(range(4, backend.num_qubits - 2, 2)) * times)
    benches = []
    for param in params:
        comp = ReduceSWAPCompiler(
            backend,
            max_virtual_gates=num_vgates,
            max_distance=max_distance,
            reverse_order=reverse_order,
        )
        circuits = get_circuits(circ_name, param, nums_qubits=nums_qubits)
        benches.append(
            Benchmark(
                circuits=circuits,
                backend=backend,
                virt_compiler=comp,
                result_file=f"results/vqr/{backend.name}/{num_vgates}_vgates/{max_distance}_distance/{circ_name}_{param}.csv",
            )
        )
    return benches


def generate_gen_bisection_benchmarks(
    circ_name: str,
    params: list[int | float],
    backend: BackendV2,
    num_vgates: int = 3,
    reverse_order: bool = True,
    times: int = 3,
):
    nums_qubits = sorted(list(range(4, backend.num_qubits - 2, 2)) * times)
    benches = []
    for param in params:
        comp = GeneralBisectionCompiler(num_vgates, reverse_order)
        circuits = get_circuits(circ_name, param, nums_qubits=nums_qubits)
        benches.append(
            Benchmark(
                circuits=circuits,
                backend=backend,
                virt_compiler=comp,
                result_file=f"results/gen_bisect/{backend.name}/{num_vgates}_vgates/{circ_name}_{param}.csv",
            )
        )
    return benches
