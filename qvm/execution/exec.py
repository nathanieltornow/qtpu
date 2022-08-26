import imp
import itertools
from pyclbr import Function
from typing import Any, Dict, Iterator, List, Optional, Tuple

from qiskit import transpile, QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
import lithops as lh


from qvm.circuit import VirtualCircuitInterface, VirtualCircuit, Fragment
from qvm.circuit.virtual_gate.virtual_gate import VirtualBinaryGate
from qvm.execution.knit import knit
from qvm.execution.merge import merge
from qvm.result import Result
from qvm.transpiler.default_flags import DEFAULT_TRANSPILER_FLAGS, DEFAULT_EXEC_FLAGS
from qvm.transpiler.transpiled_fragment import DeviceInfo, TranspiledVirtualCircuit
from .configurator import (
    FragmentConfigurator,
    VirtualCircuitConfigurator,
)


def execute_virtual_circuit(
    virtual_circuits: List[VirtualCircuit],
    backend: Optional[Backend] = None,
    transpile_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
    exec_flags: Dict[str, Any] = DEFAULT_EXEC_FLAGS,
) -> List[Result]:
    if backend is None:
        backend = AerSimulator()

    dimensions: List[int] = []
    all_circuits: List[QuantumCircuit] = []
    all_configurators: List[VirtualCircuitConfigurator] = []

    for vc in virtual_circuits:
        configurator = VirtualCircuitConfigurator(vc)
        if len(configurator.virtual_gates()) == 0:
            all_circuits.append(vc.to_circuit())
            dimensions.append(1)
            all_configurators.append(configurator)
            continue

        conf_circs = list(configurator)
        all_circuits += conf_circs
        dimensions.append(len(conf_circs))
        all_configurators.append(configurator)

    t_circs = transpile(all_circuits, backend, **transpile_flags)
    all_results: List[Result] = []

    if len(t_circs) == 1:
        return [Result.from_counts(backend.run(t_circs[0]).result().get_counts())]

    else:
        all_results = [
            Result.from_counts(cnt)
            for cnt in backend.run(t_circs, **exec_flags).result().get_counts()
        ]
    results: List[Result] = []
    for i, dim in enumerate(dimensions):
        cur_res = all_results[:dim]
        results.append(knit(cur_res, all_configurators[i].virtual_gates()))
        all_results = all_results[dim:]
    return results


def execute_fragment(
    virtual_circuit: VirtualCircuit, fragment: Fragment, device_info: DeviceInfo
) -> Dict[Tuple[int, ...], Result]:
    configurator = FragmentConfigurator(virtual_circuit, fragment).configured_circuits()
    conf_ids, conf_circs = zip(*configurator)
    results = execute_virtual_circuit(
        list(conf_circs),
        device_info.backend,
        device_info.transpile_flags,
        device_info.exec_flags,
    )
    return dict(zip(conf_ids, results))


def config_ids(virtual_gates: List[VirtualBinaryGate]) -> List[Tuple[int, ...]]:
    conf_list = [tuple(range(len(vg.configure()))) for vg in virtual_gates]
    return list(itertools.product(*conf_list))


def execute_fragmented_circuit(virtual_circuit: TranspiledVirtualCircuit) -> Result:
    fragments = virtual_circuit.fragments
    all_results = []
    for frag in fragments:
        all_results.append(
            execute_fragment(virtual_circuit, frag, virtual_circuit.device_info(frag))
        )
    merged = merge(config_ids(virtual_circuit.virtual_gates(True)), all_results)
    return knit(merged, virtual_circuit.virtual_gates(True))
