from dataclasses import dataclass
from typing import Any, Dict, Optional

from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator

from qvm.circuit import Fragment
from qvm.circuit.virtual_circuit import VirtualCircuit

DEFAULT_TRANSPILER_FLAGS = {"optimization_level": 3}
DEFAULT_EXEC_FLAGS = {"shots": 10000}


@dataclass
class DeviceInfo:
    backend: Backend
    transpile_flags: Dict[str, Any]
    exec_flags: Dict[str, Any]


class TranspiledVirtualCircuit(VirtualCircuit):
    _transp_fragments: Dict[Fragment, DeviceInfo]
    _default_device_info: DeviceInfo

    def __init__(
        self,
        vc: VirtualCircuit,
        default_device_info: Optional[DeviceInfo] = None,
    ):
        super().__init__(
            *vc.qregs,
            *vc.cregs,
            name=vc.name,
            global_phase=vc.global_phase,
            metadata=vc.metadata,
        )
        for circ_instr in vc.data:
            self.append(circ_instr.copy())
        self._transp_fragments = {}
        if default_device_info is None:
            default_device_info = DeviceInfo(
                backend=AerSimulator(),
                transpile_flags={"optimization_level": 3},
                exec_flags={"shots": 10000},
            )
        self._default_device_info = default_device_info

    def transpile_fragment(self, fragment: Fragment, device_info: DeviceInfo) -> None:
        if fragment not in self.fragments:
            raise ValueError(f"Fragment {fragment} not in virtual circuit")
        self._transp_fragments[fragment] = device_info

    def device_info(self, fragment: Fragment) -> DeviceInfo:
        if fragment not in self.fragments:
            self._transp_fragments.pop(fragment, None)
            raise ValueError(f"Fragment {fragment} not in virtual circuit")
        if fragment not in self._transp_fragments:
            return self._default_device_info
        return self._transp_fragments[fragment]
