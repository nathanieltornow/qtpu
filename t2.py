from qvm.result import Result
from sortedcontainers import SortedDict

# res1 = Result.from_counts({"00": 1, "11": 1})
# res2 = Result.from_counts({"11": 1, "01": 1})

# a, b = res1.without_first_bit()
# print(a.counts(), b.counts())


r1 = Result(SortedDict({0: 0.283203125, 1: 0.244140625}), 2)
r2 = Result(SortedDict({2: 0.232421875, 3: 0.240234375}), 2)

r3 = r1 + r2

print(r3._probs)
