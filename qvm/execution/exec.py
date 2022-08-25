import imp
from pyclbr import Function
from typing import Any, Dict, List, Optional

from qiskit import transpile, QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
import lithops as lh


from qvm.circuit import VirtualCircuitInterface, VirtualCircuit, Fragment
from qvm.execution.knit import knit
from qvm.execution.merge import merge
from qvm.result import Result
from qvm.transpiler.default_flags import DEFAULT_TRANSPILER_FLAGS, DEFAULT_EXEC_FLAGS
from .configurations import BinaryFragmentsConfigurator, VirtualCircuitConfigurator


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


def execute_fragmented_circuit(
    virtual_circuits: List[VirtualCircuit],
    backend: Optional[Backend] = None,
    transpile_flags: Dict[str, Any] = DEFAULT_TRANSPILER_FLAGS,
    exec_flags: Dict[str, Any] = DEFAULT_EXEC_FLAGS,
) -> List[Result]:
    if backend is None:
        backend = AerSimulator()

    results = []

    fexec = lh.FunctionExecutor()

    for vc in virtual_circuits:

        fragments = vc.fragments

        print("HALLO", len(fragments))

        if len(fragments) == 1:
            results.append(
                execute_virtual_circuit([vc], backend, transpile_flags, exec_flags)[0]
            )
        elif len(fragments) == 2:
            frag_configs = BinaryFragmentsConfigurator(vc)
            frags1, frags2 = zip(*frag_configs.configured_fragments())

            fexec.call_async(
                execute_virtual_circuit,
                (list(frags1), backend, transpile_flags, exec_flags),
            )
            fexec.call_async(
                execute_virtual_circuit,
                (list(frags2), backend, transpile_flags, exec_flags),
            )
            res = fexec.get_result()

            merged = merge(res[0], res[1])
            results.append(knit(merged, frag_configs.virtual_gates()))
        elif len(fragments) >= 3:

            frag_list = list(fragments)
            frag1_qubits = set().union(
                *[frag.qubits for frag in frag_list[: len(frag_list) // 2]]
            )
            frag2_qubits = set().union(
                *[frag.qubits for frag in frag_list[len(frag_list) // 2 :]]
            )
            frag_configs = BinaryFragmentsConfigurator(vc, frag1_qubits, frag2_qubits)
            frags1, frags2 = zip(*frag_configs.configured_fragments())

            fexec.call_async(
                execute_fragmented_circuit,
                (list(frags1), backend, transpile_flags, exec_flags),
            )
            fexec.call_async(
                execute_fragmented_circuit,
                (list(frags2), backend, transpile_flags, exec_flags),
            )
            res = fexec.get_result()
            merged = merge(res[0], res[1])
            results.append(knit(merged, frag_configs.virtual_gates()))
        else:
            raise NotImplementedError("Should not be here")

    return results
