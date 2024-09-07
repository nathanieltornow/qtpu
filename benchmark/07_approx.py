import os
import pandas as pd
import optuna
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator

import qtpu
from qtpu.compiler.compiler import compile_reach_size

from benchmark._plot_util import *

from benchmark.ansatz import generate_ansatz, qaoa2
from benchmark.util import get_info, append_to_csv
from qtpu.circuit import cuts_to_moves
from benchmark.exec_qtpu import qtpu_execute_cutensor
from qtpu.contract import contract
from qtpu.circuit import circuit_to_hybrid_tn
from circuit_knitting.cutting.qpd import TwoQubitQPDGate

for x in range(8, 10):
    circuit = qaoa2(4, 5, 2)

    act_res = Estimator().run(circuit, ["Z" * circuit.num_qubits]).result().values[0]

    circuit.measure_all()
    cut_circuit = compile_reach_size(circuit, 5)
    cut_circuit = cuts_to_moves(cut_circuit)

    for tolerance in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:

        res, timings = qtpu_execute_cutensor(
            cut_circuit, tolerance=tolerance, return_result=True
        )
        # htn = circuit_to_hybrid_tn(cut_circuit)
        # errs = htn.simplify(tolerance)
        # print(errs)
        # res = contract(htn)

        error = np.abs(res - act_res)

        print(res, act_res)

        print(error)

        append_to_csv(
            "07_approx.csv",
            {
                "try": x,
                "tolerance": tolerance,
                "error": error,
                **timings,
            },
        )
    # print(timings)
