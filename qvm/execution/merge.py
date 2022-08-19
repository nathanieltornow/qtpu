from typing import Dict, Optional, Tuple

import lithops.multiprocessing as mp

from qvm.result import Result


ConfigResultsType = Dict[Tuple[int, ...], Result]


def _merge_wrapper(
    config: Tuple[int, ...], result: Result, config_results: ConfigResultsType
) -> Tuple[Tuple[int, ...], Result]:
    return config, result.merge(config_results[config])


def merge(
    result1: ConfigResultsType, result2: ConfigResultsType, pool: Optional[mp.Pool]
) -> ConfigResultsType:
    result: ConfigResultsType = {}
    if pool is None:
        for conf1, res1 in result1.items():
            result[conf1] = res1.merge(result2[conf1])
    else:
        raise NotImplementedError("parallel merge not implemented")
    return {}
