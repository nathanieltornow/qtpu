import itertools
from typing import Iterable, Iterator, List, Tuple
from qvm.executor.frag_executor import FragmentExecutor
from qvm.prob import ProbDistribution
from qvm.virtual_gate.virtual_gate import VirtualBinaryGate


def chunk(l: Iterable, n: int) -> Iterator[List]:
    it = iter(l)
    while True:
        ch = list(itertools.islice(it, n))
        if not ch:
            return
        yield ch


def merge(
    frag_execs: List[FragmentExecutor], config_id: Tuple[int, ...]
) -> ProbDistribution:
    prob_dists = [fexec.get_result(config_id) for fexec in frag_execs]
    assert len(prob_dists) > 0
    res = prob_dists[0]
    for prob_dist in prob_dists[1:]:
        res = res.merge(prob_dist)
    return res


def knit(
    frag_execs: List[FragmentExecutor], vgates: List[VirtualBinaryGate]
) -> ProbDistribution:
    conf_l = [list(range(len(vgate.configure()))) for vgate in vgates]
    config_ids = iter(itertools.product(*conf_l))
    results = [merge(frag_execs, config_id) for config_id in config_ids]
    while len(vgates) > 0:
        vgate = vgates.pop(-1)
        chunks = list(chunk(list(results), len(vgate.configure())))
        results = [vgate.knit(ch) for ch in chunks]
    return results[0]
