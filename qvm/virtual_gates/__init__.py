from typing import Callable

from qvm.instructions import VirtualBinaryGate
from .cz import VirtualCZ
from .cx import VirtualCX
from .rzz import VirtualRZZ
from .move import VirtualMove


def generate_virtual_cz(_: list) -> VirtualCZ:
    return VirtualCZ()


def generate_virtual_cx(_: list) -> VirtualCX:
    return VirtualCX()


def generate_virtual_rzz(params: list) -> VirtualRZZ:
    return VirtualRZZ(params)


VIRTUAL_GATE_GENERATORS: dict[str, Callable[[list], VirtualBinaryGate]] = {
    "cz": generate_virtual_cz,
    "cx": generate_virtual_cx,
    "rzz": generate_virtual_rzz,
}
