import os

import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Pauli
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator

from qvm.virtual_circuit import VirtualCircuit
from qvm.virtual_gates import VirtualMove
from qvm import build_dummy_tensornetwork


def get_perfect_z_expval(circuit: QuantumCircuit) -> float:
    circuit = circuit.remove_final_measurements(inplace=False)
    op = Pauli("Z" * circuit.num_qubits)
    sv: Statevector = StatevectorSimulator().run(circuit).result().get_statevector()
    return sv.expectation_value(op)


def get_base_knit_overhead(virtual_circuit: VirtualCircuit) -> int:
    num_fragments = len(virtual_circuit.fragments)

    vgate_overhead = np.prod(
        [
            8 if isinstance(vgate, VirtualMove) else 6
            for vgate in virtual_circuit.virtual_gates
        ]
    )
    num_adds = vgate_overhead - 1
    num_muls = num_adds * num_fragments

    return num_muls


def get_knit_overhead(virtual_circuit: VirtualCircuit) -> int:
    tn = build_dummy_tensornetwork(virtual_circuit)
    tn.draw()
    return int(tn.contraction_cost(optimize="auto"))
