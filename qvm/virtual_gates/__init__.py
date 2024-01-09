from .cz import VirtualCZ
from .cx import VirtualCX
from .rzz import VirtualRZZ
from .move import VirtualMove


VIRTUAL_GATE_TYPES = {
    "cx": VirtualCX,
    "cz": VirtualCZ,
    "rzz": VirtualRZZ,
}
