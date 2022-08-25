import itertools
import multiprocessing
from multiprocessing.resource_sharer import stop
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import lithops

from qvm.circuit.virtual_gate import VirtualBinaryGate
from qvm.result import Result


def chunk(it: Iterable, n: int) -> Iterator[List[Any]]:
    it = iter(it)
    while True:
        ch = list(itertools.islice(it, n))
        if not ch:
            return
        yield ch


def knit(results: Iterable[Result], virtual_gates: List[VirtualBinaryGate]) -> Result:
    fexec = lithops.FunctionExecutor()
    while len(virtual_gates) > 0:
        vgate = virtual_gates.pop(-1)
        chunks = list(chunk(list(results), len(vgate.configure())))
        if len(chunks) >= 36:
            results = fexec.map(vgate.knit, chunks).get_result()
        else:
            results = [vgate.knit(ch) for ch in chunks]
    return list(results)[0]
