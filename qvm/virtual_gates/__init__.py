# class VirtualCX(VirtualCZ):
#     def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
#         cx_insts = []
#         for cz_inst in super()._instantiations():
#             cx_insts.append((cz_inst[0], [HGate()] + cz_inst[1] + [HGate()]))
#         return cx_insts


# class VirtualCY(VirtualCX):
#     def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
#         cy_insts = []
#         for cx_inst in super()._instantiations():
#             cy_insts.append(
#                 (
#                     cx_inst[0],
#                     [RZGate(-np.pi / 2)] + cx_inst[1] + [RZGate(np.pi / 2)],
#                 )
#             )
#         return cy_insts


# class VirtualRZZ(VirtualBinaryGate):
#     def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
#         pauli = ZGate()
#         r_plus = SGate()
#         r_minus = SdgGate()
#         meas = Measure()
#         return [
#             ([], []),  # Identity
#             ([pauli], [pauli]),
#             ([meas], [r_plus]),
#             ([meas], [r_minus]),
#             ([r_plus], [meas]),
#             ([r_minus], [meas]),
#         ]

#     def coefficients(self) -> np.ndarray:
#         theta = -self._params[0] / 2
#         cs = np.cos(theta) * np.sin(theta)
#         return np.array(
#             [
#                 np.cos(theta) ** 2,
#                 np.sin(theta) ** 2,
#                 -cs,
#                 cs,
#                 -cs,
#                 cs,
#             ]
#         )


# class WireCut(Barrier):
#     def __init__(self):
#         super().__init__(num_qubits=1, label="wc")

#     def _define(self):
#         self._definition = QuantumCircuit(1)


# class VirtualMove(VirtualBinaryGate):
#     def _instantiations(self) -> list[tuple[list[Instruction], list[Instruction]]]:
#         return [
#             ([], []),
#             (
#                 [],
#                 [XGate()],
#             ),
#             (
#                 [HGate(), Measure()],
#                 [HGate()],
#             ),
#             (
#                 [HGate(), Measure()],
#                 [XGate(), HGate()],
#             ),
#             (
#                 [SXGate(), Measure()],
#                 [SXdgGate()],
#             ),
#             (
#                 [SXGate(), Measure()],
#                 [XGate(), SXdgGate()],
#             ),
#             (
#                 [Measure()],
#                 [],
#             ),
#             (
#                 [Measure()],
#                 [XGate()],
#             ),
#         ]

#     def coefficients(self) -> np.ndarray:
#         return 0.5 * np.array([1, 1, 1, -1, 1, -1, 1, -1])

from .cz import VirtualCZ
from .cx import VirtualCX
from .rzz import VirtualRZZ

VIRTUAL_GATE_TYPES = {
    "cx": VirtualCX,
    # "cy": VirtualCY,
    "cz": VirtualCZ,
    "rzz": VirtualRZZ,
}
