from typing import Union

ACCURACY = 1e-4


class QuasiDistr(dict[int, float]):
    def __init__(self, data: dict[int, float]) -> None:
        super().__init__(data)

    @staticmethod
    def from_counts(counts: dict[str, int]) -> "QuasiDistr":
        shots = sum(counts.values())
        return QuasiDistr(
            {
                int("".join(key.split()), 2): value / shots
                for key, value in counts.items()
            }
        )

    def to_counts(self, num_clbits: int, shots: int) -> dict[str, int]:
        return {
            bin(key)[2:].zfill(num_clbits): int(abs(value * shots))
            for key, value in self.items()
        }

    def nearest_probability_distribution(self) -> dict[int, float]:
        sorted_probs = dict(sorted(self.items(), key=lambda item: item[1]))
        num_elems = len(sorted_probs)
        new_probs = {}
        beta = 0.0
        diff = 0.0
        for key, val in sorted_probs.items():
            temp = val + beta / num_elems
            if temp < 0:
                beta += val
                num_elems -= 1
                diff += val * val
            else:
                diff += (beta / num_elems) * (beta / num_elems)
                new_probs[key] = sorted_probs[key] + beta / num_elems
        return new_probs

    def merge(self, other: "QuasiDistr") -> "QuasiDistr":
        merged_data = {}
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                if abs(value1 * value2) > ACCURACY:
                    merged_data[key1 ^ key2] = value1 * value2
        return QuasiDistr(merged_data)

    def __add__(self, other: "QuasiDistr") -> "QuasiDistr":
        added_data = {key: self[key] + other.get(key, 0.0) for key in self.keys()}
        only_others = {
            key: other[key] for key in other.keys() if key not in self.keys()
        }
        added_data.update(only_others)
        return QuasiDistr(added_data)

    def __sub__(self, other: "QuasiDistr") -> "QuasiDistr":
        subbed_data = {key: self[key] - other.get(key, 0.0) for key in self.keys()}
        only_others = {
            key: -other[key] for key in other.keys() if key not in self.keys()
        }
        subbed_data.update(only_others)
        return QuasiDistr(subbed_data)

    def __mul__(self, other: Union[int, float, "QuasiDistr"]) -> "QuasiDistr":
        if isinstance(other, QuasiDistr):
            return self.merge(other)
        elif isinstance(other, float) or isinstance(other, int):
            return QuasiDistr(
                {
                    key: self[key] * other
                    for key in self.keys()
                    if abs(self[key] * other) > ACCURACY
                }
            )
        raise TypeError(f"Cannot multiply QuasiDistr by {type(other)}")

    def __rmul__(self, other: Union[int, float, "QuasiDistr"]) -> "QuasiDistr":
        return self.__mul__(other)

    def __repr__(self) -> str:
        return "QuasiDistr(" + str(len(self)) + ")"

    def add_value(self, key: int, value: float) -> None:
        self[key] = self.get(key, 0.0) + value


def prepare_quasidist(quasidist: QuasiDistr, num_bits: int) -> QuasiDistr:
    new_quasidist = QuasiDistr({})
    for key, value in quasidist.items():
        mask = (1 << num_bits) - 1
        first_bits = bin(key & ~mask)[2:]
        parity = sum(int(c) for c in first_bits) % 2
        new_value = value if parity == 0 else -value
        new_quasidist.add_value(key & mask, new_value)
    return new_quasidist
