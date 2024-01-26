from typing import Callable, Any, Iterable

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister as Fragment
from qiskit_aer import AerSimulator

from qvm.virtual_circuit import VirtualCircuit
from qvm.tn import build_tensornetwork


RunCircFuncType = Callable[
    [QuantumCircuit, Iterable[dict[Parameter, float]], int, Any], NDArray[np.float32]
]


def run_virtual_circuit(
    virtual_circuit: VirtualCircuit,
    run_circ_func: RunCircFuncType | None = None,
    shots: int = 100000,
    **kwargs: Any,
) -> float:
    results = sample_virtual_circuit(virtual_circuit, run_circ_func, shots, **kwargs)
    tn = build_tensornetwork(virtual_circuit, results)
    result = tn.contract(all, optimize="auto")
    return result


def sample_virtual_circuit(
    virtual_circuit: VirtualCircuit,
    run_circ_func: RunCircFuncType | None = None,
    shots: int = 100000,
    **kwargs: Any,
) -> dict[Fragment, NDArray[np.float32]]:
    if run_circ_func is None:
        run_circ_func = simulate_circuit

    results = {}
    for frag, param_batch in virtual_circuit.generate_instance_parameters().items():
        circuit = virtual_circuit.fragment_circuits[frag]
        results[frag] = run_circ_func(circuit, param_batch, shots, **kwargs)
    return results


def expval_from_counts(counts: dict[str, int]) -> float:
    expval = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        parity = 1 - 2 * int(bitstring.count("1") % 2)
        expval += parity * (count / shots)
    return expval


def simulate_circuit(
    circuit: QuantumCircuit,
    arg_batch: Iterable[dict[Parameter, float]],
    shots: int,
    **kwargs: Any,
) -> NDArray[np.float32]:
    circuits = [circuit.assign_parameters(args).decompose() for args in arg_batch]
    # check if circuit has no measurements

    def _run_circ(circuit: QuantumCircuit, shots: int) -> float:
        sim = AerSimulator()
        try:
            counts = sim.run(circuit, shots=shots).result().get_counts()
            return expval_from_counts(counts)
        except Exception:
            print("Error running circuit, returning 1.0")
            return 1.0

    return np.array(
        [_run_circ(circuit=circuit, shots=shots) for circuit in circuits],
        dtype=np.float32,
    )
