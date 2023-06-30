from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.providers.fake_provider import FakeOslo

from qvm.compiler import VirtualizationCompiler
from qvm.compiler.virtualization.reduce_swap import ReduceSWAPCompiler
from qvm.compiler.virtualization.general_bisection import GeneralBisectionCompiler
from qvm.compiler.virtualization.gate_decomp import OptimalDecompositionCompiler

from _circuits import get_circuits


@dataclass
class Benchmark:
    circuits: list[QuantumCircuit]
    backend: BackendV2
    result_file: str
    virt_compiler: VirtualizationCompiler | None = None
    base_backend: BackendV2 | None = None


def generate_vqr_benchmarks(
    circ_name: str,
    params: list[int | float],
    backend: BackendV2,
    num_vgates: int = 3,
    reverse_order: bool = False,
    times: int = 3,
) -> list[Benchmark]:
    nums_qubits = sorted(list(range(6, backend.num_qubits, 2)) * times)
    benches = []
    for param in params:
        comp = ReduceSWAPCompiler(
            backend,
            max_virtual_gates=num_vgates,
            reverse_order=reverse_order,
        )
        circuits = get_circuits(circ_name, param, nums_qubits=nums_qubits)
        benches.append(
            Benchmark(
                circuits=circuits,
                backend=backend,
                virt_compiler=comp,
                result_file=f"results/vqr/{backend.name}/{num_vgates}_vgates/{circ_name}_{param}.csv",
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
) -> list[Benchmark]:
    nums_qubits = sorted(list(range(6, backend.num_qubits - 2, 2)) * times)
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


def generate_cut_comparison_benchmarks(
    circ_name: str,
    params: list[int | float],
    times: int = 3,
) -> list[Benchmark]:
    nums_qubits = sorted(list(range(10, 31, 5)) * times)
    benches = []
    for param in params:
        circuits = get_circuits(circ_name, param, nums_qubits=nums_qubits)
        benches.append(
            Benchmark(
                circuits=circuits,
                backend=FakeOslo(),
                virt_compiler=None,
                result_file=f"results/cut_comparison/{circ_name}_{param}.csv",
            )
        )
    return benches


def generate_scale_noisy_benchmarks(
    circ_name: str,
    params: list[int | float],
    backend: BackendV2,
    base_backend: BackendV2,
    times: int = 3,
) -> list[Benchmark]:
    nums_qubits = sorted(list(range(4, base_backend.num_qubits - 2, 2)) * times)
    benches = []
    for param in params:
        circuits = get_circuits(circ_name, param, nums_qubits=nums_qubits)
        comp = OptimalDecompositionCompiler(backend.num_qubits)
        benches.append(
            Benchmark(
                circuits=circuits,
                backend=backend,
                virt_compiler=comp,
                base_backend=base_backend,
                result_file=f"results/scale_noisy/{backend.name}/{circ_name}_{param}.csv",
            )
        )
    return benches
