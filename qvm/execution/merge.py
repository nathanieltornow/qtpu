from typing import Dict, Iterable, List, Optional, Tuple

import lithops

from qvm.result import Result


def merge_res(res1: Result, res2: Result) -> Result:
    return res1.merge(res2)


def merge(results1: List[Result], results2: List[Result]) -> List[Result]:
    zipped = list(zip(results1, results2))
    fexec = lithops.FunctionExecutor()
    return fexec.map(merge_res, zipped).get_result()
