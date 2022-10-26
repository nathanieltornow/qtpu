Counts = dict[str, int]


class ProbDistr(dict):
    def __init__(self, data: dict[str, float]):
        super().__init__(data)

    @staticmethod
    def from_counts(counts: Counts) -> "ProbDistr":
        total = sum(counts.values())
        return ProbDistr(
            {state.replace(" ", ""): count / total for state, count in counts.items()}
        )

    def counts(self, shots: int) -> Counts:
        return {state: int(prob * shots) for state, prob in self.items()}

    def without_first_bit(self) -> tuple["ProbDistr", "ProbDistr"]:
        zeros, ones = ProbDistr({}), ProbDistr({})
        for state, prob in self.items():
            if state[0] == "0":
                zeros[state[1:]] = prob
            else:
                ones[state[1:]] = prob
        return zeros, ones

    def _merged_state(self, state1: str, state2: str) -> str:
        if len(state1) != len(state2):
            raise ValueError("States must have the same length")
        return "".join(
            ["1" if s1 == "1" or s2 == "1" else "0" for s1, s2 in zip(state1, state2)]
        )

    def merge(self, other: "ProbDistr") -> "ProbDistr":
        res = {}
        for state1, prob1 in self.items():
            for state2, prob2 in other.items():
                res[self._merged_state(state1, state2)] = prob1 * prob2
        return ProbDistr(res)

    def __add__(self, other: "ProbDistr") -> "ProbDistr":
        only_others = set(other) - set(self)
        res = {state: prob + other.get(state, 0.0) for state, prob in self.items()}
        res.update({state: other[state] for state in only_others})
        return ProbDistr(res)

    def __sub__(self, other: "ProbDistr") -> "ProbDistr":
        only_others = set(other) - set(self)
        res = {state: prob - other.get(state, 0.0) for state, prob in self.items()}
        res.update({state: -other[state] for state in only_others})
        return ProbDistr(res)

    def __mul__(self, scalar: float) -> "ProbDistr":
        return ProbDistr({state: prob * scalar for state, prob in self.items()})

    def __rmul__(self, scalar: float) -> "ProbDistr":
        return self * scalar
