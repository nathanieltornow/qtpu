import itertools
from typing import Iterable, Iterator, List, Tuple

import ray

from qvm.prob import ProbDistribution
from qvm.virtual_gate import VirtualBinaryGate


def chunk(l: Iterable, n: int) -> Iterator[List]:
    it = iter(l)
    while True:
        ch = list(itertools.islice(it, n))
        if not ch:
            return
        yield ch


@ray.remote
def merge_frag_results(handles: List, config_id: Tuple[int, ...]) -> ProbDistribution:
    prob_dists = ray.get([handle.get_result.remote(config_id) for handle in handles])
    assert len(prob_dists) > 0
    res = prob_dists[0]
    for prob_dist in prob_dists[1:]:
        res = res.merge(prob_dist)
    return res


@ray.remote
def merge(
    handles: List, config_ids: Iterable[Tuple[int, ...]]
) -> List[ProbDistribution]:
    futures = [
        merge_frag_results.remote(handles, config_id) for config_id in config_ids
    ]
    return ray.get(futures)


@ray.remote
def knit_virtual_gate(
    vgate: VirtualBinaryGate, results: List[ProbDistribution]
) -> ProbDistribution:
    return vgate.knit(results)


@ray.remote
def knit(
    handles: List, vgates: List[VirtualBinaryGate], merge_chunk_size: int = 1296
) -> ProbDistribution:
    conf_l = [list(range(len(vgate.configure()))) for vgate in vgates]
    config_ids = iter(itertools.product(*conf_l))

    conf_id_chunks = chunk(config_ids, merge_chunk_size)
    merge_futures = [merge.remote(handles, chunk) for chunk in conf_id_chunks]
    results = list(itertools.chain.from_iterable(ray.get(merge_futures)))

    while len(vgates) > 0:
        vgate = vgates.pop(-1)
        chunks = list(chunk(list(results), len(vgate.configure())))
        if len(chunks) >= 36:
            knit_futures = [knit_virtual_gate.remote(vgate, chunk) for chunk in chunks]
            results = ray.get(knit_futures)
        else:
            results = [vgate.knit(ch) for ch in chunks]
    return list(results)[0]
