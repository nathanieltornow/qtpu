import itertools
from typing import Iterable, Iterator

from qiskit.circuit import QuantumCircuit

from vqc.knitting.sample import SampleIdType, _sample
from vqc.prob_distr import Counts, ProbDistr
from vqc.types import VirtualGate
from vqc.virtual_circuit import VirtualCircuit


def chunk(l: Iterable, n: int) -> Iterator[list]:
    it = iter(l)
    while True:
        ch = list(itertools.islice(it, n))
        if not ch:
            return
        yield ch


def _matching_result(
    sample_id: SampleIdType, results: dict[SampleIdType, ProbDistr]
) -> ProbDistr:
    first_key = next(iter(results))
    key = []
    for i in range(len(first_key)):
        if first_key[i] == -1:
            key.append(-1)
        else:
            key.append(sample_id[i])
    return results[tuple(key)]


def _merge_one(
    sample_id: SampleIdType, frag_results: list[dict[SampleIdType, ProbDistr]]
) -> ProbDistr:
    results = [_matching_result(sample_id, results) for results in frag_results]
    merged_result = results[0]
    for res in results[1:]:
        merged_result = merged_result.merge(res)
    return merged_result


def _merge(
    vgates: list[VirtualGate], frag_results: list[dict[SampleIdType, ProbDistr]]
) -> list[ProbDistr]:
    sample_ids_list = [range(len(vgate.configure())) for vgate in vgates]
    sample_ids = itertools.product(*sample_ids_list)
    return [_merge_one(sample_id, frag_results) for sample_id in sample_ids]


def _knit(
    vgates: list[VirtualGate], frag_results: list[dict[SampleIdType, ProbDistr]]
) -> ProbDistr:
    results = _merge(vgates, frag_results)

    while len(vgates) > 0:
        vgate = vgates.pop(-1)
        chunks = list(chunk(list(results), len(vgate.configure())))
        results = list(map(vgate.knit, chunks))
    return results[0]


class Knitter:
    def __init__(self, vc: VirtualCircuit) -> None:
        self._vc = vc
        self._samples = _sample(vc)

    def samples(self) -> dict[str, list[QuantumCircuit]]:
        return {
            name: [s for _, s in samples] for name, samples in self._samples.items()
        }

    def knit(self, sample_results: dict[str, list[Counts]]):
        frag_results = []
        for name, res in sample_results.items():
            frag_res = {
                sample_id: ProbDistr.from_counts(counts)
                for (sample_id, _), counts in zip(self._samples[name], res)
            }
            frag_results.append(frag_res)

        return _knit(self._vc.virtual_gates, frag_results)
