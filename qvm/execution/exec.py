from typing import Dict, List, Tuple

from qiskit.providers.aer import AerSimulator
import lithops as lh


from qvm.execution.configurator import (
    Configurator,
    VirtualGateInfo,
    VirtualizationInfo,
    compute_virtual_gate_info,
)
from qvm.execution.knit import knit

from qvm.result import Result
from .merge import merge
from qvm.transpiler.fragmented_circuit import FragmentedCircuit, Fragment


def _execute_fragment(
    frag: Fragment, vgate_infos: List[VirtualGateInfo], shots: int = 20000
) -> Dict[Tuple[int, ...], Result]:
    conf = Configurator(frag, vgate_infos)
    conf_ids, circs = zip(*conf.configured_circuits())
    if frag.backend is None:
        frag.backend = AerSimulator()
    circ_l = [c.decompose(["conf"]) for c in circs]
    counts = frag.backend.run(circ_l, shots=shots).result().get_counts()
    if isinstance(counts, dict):
        assert len(conf_ids) == 1
        counts = [counts]
    return dict(zip(conf_ids, [Result.from_counts(cnt) for cnt in counts]))


def execute_fragmented_circuit(
    frag_circ: FragmentedCircuit, shots: int = 20000
) -> Dict[str, int]:
    vgate_infos = compute_virtual_gate_info(frag_circ)
    fexec = lh.FunctionExecutor()
    for frag in frag_circ.fragments:
        fexec.call_async(_execute_fragment, (frag, vgate_infos, shots))
    frag_results = fexec.get_result()
    if len(frag_circ.fragments) == 1:
        frag_results = [frag_results]

    virt_info = VirtualizationInfo(frag_circ)
    merged = merge(list(virt_info.config_ids), frag_results)
    return knit(merged, virt_info.virtual_gates).counts(shots)
