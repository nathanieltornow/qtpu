from typing import Dict, Iterable, List, Optional, Tuple

import lithops.multiprocessing as mp

from qvm.result import Result


ConfigResultsType = Dict[Tuple[int, ...], Result]


def _match_config_id(
    config_id1: Tuple[int, ...], config_id2: Tuple[int, ...]
) -> Optional[Tuple[int, ...]]:
    if config_id1 == config_id2:
        return config_id1
    if len(config_id1) != len(config_id2):
        return None
    res_conf_id: Tuple[int, ...] = ()
    for i, c in enumerate(config_id1):
        other_c = config_id2[i]
        if c == other_c:
            res_conf_id += (c,)
        elif c < 0:
            res_conf_id += (other_c,)
        elif other_c < 0:
            res_conf_id += (c,)
        else:
            return None
    return res_conf_id


def matching_result(
    config_id: Tuple[int, ...], config_results: ConfigResultsType
) -> Result:
    # TODO make more efficient, binary search
    for other_config_id, result in config_results.items():
        matched_id = _match_config_id(config_id, other_config_id)
        if matched_id is not None:
            return result
    raise ValueError(
        f"no result found for config_id {config_id}, where always should be found"
    )


def merge_two(
    result1: ConfigResultsType, result2: ConfigResultsType, pool: Optional[mp.Pool]
) -> ConfigResultsType:
    result: ConfigResultsType = {}
    if pool is None:
        for conf1, res1 in result1.items():
            result[conf1] = res1.merge(result2[conf1])
    else:
        raise NotImplementedError("parallel merge not implemented")
    return {}


def merge(
    results: List[ConfigResultsType], pool: Optional[mp.Pool] = None
) -> ConfigResultsType:
    if len(results) == 0:
        return {}
    if len(results) == 1:
        return results[0]

    result: ConfigResultsType = results[0]
    if pool is None:
        for res in results[1:]:
            result = merge_two(result, res, None)
    else:
        raise NotImplementedError("parallel merge not implemented")
    return result
