import math
from multiprocessing.pool import Pool

from qiskit.circuit.library import RZZGate, CZGate

from qvm.int_quasi_distr import QuasiDistr


def knit_cz(results: list[QuasiDistr], meas_index: int) -> QuasiDistr:
    assert len(results) == 6
    r0, _ = results[0].split(meas_index)
    r1, _ = results[1].split(meas_index)
    r20, r21 = results[2].split(meas_index)
    r30, r31 = results[3].split(meas_index)
    r40, r41 = results[4].split(meas_index)
    r50, r51 = results[5].split(meas_index)
    return 0.5 * (r0 + r1 + r21 - r20 + r31 - r30 + r40 - r41 + r50 - r51)


def knit_rzz(results: list[QuasiDistr], theta: float, meas_index: int) -> QuasiDistr:
    assert len(results) == 6
    r0, _ = results[0].split(meas_index)
    r1, _ = results[1].split(meas_index)
    r23 = results[2] + results[3]
    r45 = results[4] + results[5]

    r230, r231 = r23.split(meas_index)
    r450, r451 = r45.split(meas_index)

    m_theta = -theta / 2
    return (
        (r0 * math.cos(m_theta) ** 2)
        + (r1 * math.sin(m_theta) ** 2)
        + (r230 - r231 - r450 + r451) * math.cos(m_theta) * math.sin(m_theta)
    )


def log_base_6(x: int) -> float:
    return math.log(x) / math.log(6)


def _chunk_list(lst: list, chunk_size: int) -> list[list]:
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def knit_cz_batch(
    results: list[QuasiDistr], meas_index: int, pool: Pool
) -> list[QuasiDistr]:
    assert log_base_6(len(results)).is_integer()
    chunks = _chunk_list(results, 6)
    return pool.starmap(knit_cz, [(chunk, meas_index) for chunk in chunks])


def knit_rzz_batch(
    results: list[QuasiDistr], theta: float, meas_index: int, pool: Pool
) -> list[QuasiDistr]:
    assert log_base_6(len(results)).is_integer()
    chunks = _chunk_list(results, 6)
    return pool.starmap(knit_rzz, [(chunk, theta, meas_index) for chunk in chunks])


def knit(
    virtual_gates: list[CZGate | RZZGate],
    results: list[QuasiDistr],
    num_clbits: int,
    pool: Pool,
) -> QuasiDistr:
    vgate_meas_index = num_clbits - 1
    while len(virtual_gates) > 0:
        vgate = virtual_gates.pop(-1)
        if isinstance(vgate, CZGate):
            results = knit_cz_batch(results, vgate_meas_index, pool)
        elif isinstance(vgate, RZZGate):
            results = knit_rzz_batch(results, vgate.params[0], vgate_meas_index, pool)
        else:
            raise TypeError(f"Cannot knit {type(vgate)}")
        vgate_meas_index -= 1
    return results[0]


