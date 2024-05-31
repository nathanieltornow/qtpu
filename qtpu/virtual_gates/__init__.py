from typing import Callable

from qtpu.instructions import VirtualBinaryGate

from .cx import VirtualCX
from .cz import VirtualCZ
from .move import VirtualMove
from .rzz import VirtualRZZ
from .cp import VirtualCPhase


def generate_virtual_cz(_: list) -> VirtualCZ:
    return VirtualCZ()


def generate_virtual_cx(_: list) -> VirtualCX:
    return VirtualCX()


def generate_virtual_rzz(params: list) -> VirtualRZZ:
    return VirtualRZZ(params)


def generate_virtual_cp(params: list) -> VirtualCPhase:
    return VirtualCPhase(params)


VIRTUAL_GATE_GENERATORS: dict[str, Callable[[list], VirtualBinaryGate]] = {
    "cz": generate_virtual_cz,
    "cx": generate_virtual_cx,
    "rzz": generate_virtual_rzz,
    "cp": generate_virtual_cp,
}
