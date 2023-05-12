from qvm.quasi_distr import QuasiDistr

quasidistr1 = QuasiDistr({"00": 0.5, "01": 0.5})
quasidistr2 = QuasiDistr({"00": 0.2, "01": 0.2, "10": 0.4, "11": 0.2})


def test_quasidistr_add():
    assert quasidistr1 + quasidistr2 == QuasiDistr(
        {"00": 0.7, "01": 0.7, "10": 0.4, "11": 0.2}
    )


def test_quasidistr_merge():
    quasi_merge_1 = QuasiDistr({"0000": 0.5, "0001": 0.5})
    quasi_merge_2 = QuasiDistr({"0000": 0.2, "0100": 0.4, "1000": 0.2, "1100": 0.2})
    assert QuasiDistr.merge(quasi_merge_1, quasi_merge_2) == QuasiDistr(
        {
            "0000": 0.1,
            "0100": 0.2,
            "1000": 0.1,
            "1100": 0.1,
            "0001": 0.1,
            "0101": 0.2,
            "1001": 0.1,
            "1101": 0.1,
        }
    )


def test_quasidistr_sub():
    assert quasidistr1 - quasidistr2 == QuasiDistr(
        {"00": 0.3, "01": 0.3, "10": -0.4, "11": -0.2}
    )


def test_quasidistr_mul():
    scalar = 0.5
    assert scalar * quasidistr2 == QuasiDistr(
        {"00": 0.1, "01": 0.1, "10": 0.2, "11": 0.1}
    )


def test_quasidistr_divide_by_first_bit():
    assert quasidistr1.divide_by_first_bit() == (
        QuasiDistr({"0": 0.5, "1": 0.5}),
        QuasiDistr({}),
    )
    assert quasidistr2.divide_by_first_bit() == (
        QuasiDistr({"0": 0.2, "1": 0.2}),
        QuasiDistr({"0": 0.4, "1": 0.2}),
    )


def test_quasidistr_counts():
    quasi_from_counts = QuasiDistr.from_counts(
        {"00": 2, "01": 2, "10": 2, "11": 4}, shots=10
    )
    assert quasi_from_counts == QuasiDistr({"00": 0.2, "01": 0.2, "10": 0.2, "11": 0.4})
    assert {"00": 2, "01": 2, "10": 2, "11": 4} == quasi_from_counts.to_counts(shots=10)
