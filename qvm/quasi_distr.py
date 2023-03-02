class QuasiDistr(dict[str, float]):
    def __init__(self, data: dict[str, float]) -> None:
        super().__init__(data)

    @staticmethod
    def from_counts(counts: dict[str, int], num_shots: int) -> "QuasiDistr":
        return QuasiDistr({k: v / num_shots for k, v in counts.items()})

    def to_counts(self, num_shots: int) -> dict[str, int]:
        return {k: max(int(v * num_shots), 0) for k, v in self.items()}

    def divide_by_first_bit(self) -> tuple["QuasiDistr", "QuasiDistr"]:
        data1, data2 = {}, {}
        for key, value in self.items():
            if key[0] == "0":
                data1[key[1:]] = value
            else:
                data2[key[1:]] = value
        return QuasiDistr(data1), QuasiDistr(data2)

    @staticmethod
    def _merged_state(state1: str, state2: str) -> str:
        if len(state1) != len(state2):
            raise ValueError("States must have the same length")
        return "".join(
            ["1" if s1 == "1" or s2 == "1" else "0" for s1, s2 in zip(state1, state2)]
        )

    def kron(self, other: "QuasiDistr") -> "QuasiDistr":
        kron_data = {}
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                kron_key = key1 + key2
                kron_data[kron_key] = value1 * value2
        return QuasiDistr(kron_data)

    def merge(self, other: "QuasiDistr") -> "QuasiDistr":
        merged_data = {}
        for key1, value1 in self.items():
            for key2, value2 in other.items():
                merged_key = self._merged_state(key1, key2)
                merged_data[merged_key] = value1 * value2
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

    def __mul__(self, other: float) -> "QuasiDistr":
        return QuasiDistr({key: self[key] * other for key in self.keys()})

    def __rmul__(self, other: float) -> "QuasiDistr":
        return self * other
