from dataclasses import dataclass
from typing import Union
from itertools import chain

import sparse
import numpy as np


# @dataclass
# class QuasiDistribution:
#     indices: tuple[str, ...]
#     data: sparse.COO

Indices = tuple[str, ...]


class JointQuasiDistribution:
    def __init__(self, quasi_distributions: dict[Indices, sparse.COO]):
        if len(quasi_distributions) == 0:
            raise ValueError("At least one quasi distribution is required")

        first_inds = next(iter(quasi_distributions.keys()))
        assert all(
            set(first_inds) & set(inds) == set() for inds in quasi_distributions.keys()
        )
        self.quasi_distributions = quasi_distributions

    def add_tensor(self, other: "JointQuasiDistribution") -> "JointQuasiDistribution":
        assert self.quasi_distributions.keys() == other.quasi_distributions.keys()
        new_qds = {
            key: self.quasi_distributions[key] + other.quasi_distributions[key]
            for key in self.quasi_distributions.keys()
        }
        return JointQuasiDistribution(new_qds)

    # def produce_result(self) -> QuasiDistribution:
    #     qd_list = list(self.quasi_distributions.values())
    #     data = qd_list[0].data
    #     for qd in qd_list[1:]:
    #         data = sparse.tensordot(data, qd.data, axes=0)
    #     return QuasiDistribution(
    #         chain.from_iterable(self.quasi_distributions.keys()), data
    #     )

    # def add_tensor(self, other: "JointQuasiDistribution") -> "JointQuasiDistribution":
    #     assert self.quasi_distributions.keys() == other.quasi_distributions.keys()
    #     new_qds = {
    #         key: self.quasi_distributions[key].data
    #         + other.quasi_distributions[key].data
    #         for key in self.quasi_distributions.keys()
    #     }
    #     return JointQuasiDistribution(
    #         [QuasiDistribution(key, data) for key, data in new_qds.items()]
    #     )

    # def multiply_scalar(self, scalar: float | int) -> "JointQuasiDistribution":
    #     pass

    # def tensordot(self, other: "JointQuasiDistribution") -> "JointQuasiDistribution":
    #     return JointQuasiDistribution(
    #         self.quasi_distributions + other.quasi_distributions
    #     )
