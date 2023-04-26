import logging

from time import perf_counter

from qiskit.circuit import QuantumCircuit

from qvm.cutting.gate_cutting import bisect, fragment_circuit

logger = logging.getLogger("qvm")


def cut(circuit: QuantumCircuit, technique: str, **kwargs) -> QuantumCircuit:
    """
    Cuts a circuit into fragments by inserting virtual operations
    according to the given technique.

    Args:
        circuit (QuantumCircuit): The circuit to cut.
        technique (str): The technique to use.

    Returns:
        QuantumCircuit: The cut circuit.
    """
    logger.info(f"Cutting circuit with technique {technique}.")
    now = perf_counter()
    if technique == "gate_bisection":
        bisection_kwargs = {}
        if "num_fragments" in kwargs:
            bisection_kwargs["num_fragments"] = kwargs["num_fragments"]

        cut_circ = bisect(circuit, **bisection_kwargs)

    elif technique == "gate_optimal":
        pass

    elif technique == "wire_optimal":
        raise NotImplementedError("Wire-optimal cutting is not yet implemented.")

    elif technique == "optimal":
        raise NotImplementedError("Optimal cutting is not yet implemented.")

    else:
        raise ValueError(f"Unknown cutting technique: {technique}")

    logger.info(f"Cutting circuit took {perf_counter() - now} seconds.")
    return cut_circ
