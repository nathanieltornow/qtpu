from typing import Dict, Iterable, List, Optional, Tuple

import lithops

from qvm.result import Result


def merge_res(*res: Result) -> Result:
    res_l = list(res)
    if len(res_l) == 0:
        raise ValueError("No results to merge")
    if len(res_l) == 1:
        return res_l[0]
    merged = res[0]
    for r in res[1:]:
        merged.merge(r)
    return merged


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


def find_matching_result(
    config_id: Tuple[int, ...], results: Dict[Tuple[int, ...], Result]
) -> Result:
    for conf_id, res in results.items():
        if _match_config_id(conf_id, config_id) is not None:
            return res
    raise ValueError("No matching result found")


def merge_one(
    config_id: Tuple[int, ...], all_results: List[Dict[Tuple[int, ...], Result]]
) -> Result:
    res = find_matching_result(config_id, all_results[0])
    for results in all_results[1:]:
        res = res.merge(find_matching_result(config_id, results))
    return res


def merge(
    config_ids: List[Tuple[int, ...]], results: List[Dict[Tuple[int, ...], Result]]
) -> List[Result]:
    merged_results = []
    for conf_id in config_ids:
        merged_results.append(merge_one(conf_id, results))
    return merged_results


def zip_results(
    config_ids: List[Tuple[int, ...]], results: List[Tuple[Tuple[int, ...], Result]]
) -> List[Result]:
    it = iter(results)
    zipped: List[Result] = []
    for conf_id in config_ids:
        while True:
            n = next(it, None)
            if n is None:
                return zipped
            if _match_config_id(conf_id, n[0]) is not None:
                zipped.append(next(it)[1])
            else:
                break
    return zipped
