from time import perf_counter
from typing import Callable

import numpy as np

from qiskit.primitives import Sampler
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.compiler import transpile


from circuit_knitting.cutting import (
    cut_wires,
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from circuit_knitting.cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)

from qtpu.helpers import defer_mid_measurements

from benchmark.util import DummySampler


def ckt_execute(
    circuit: QuantumCircuit,
    sampler: Sampler,
    num_samples: int = np.inf,
    obs: str | None = None,
) -> tuple[float, dict]:
    if obs is None:
        obs = "Z" * circuit.num_qubits
    observable = SparsePauliOp([obs])

    start = perf_counter()
    qc_w_ancilla = cut_wires(circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)
    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )

    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )
    print(len(coefficients))

    preptime = perf_counter() - start

    # isa_subexperiments = {
    #     label: [transpile(c) for c in partition_subexpts]
    #     for label, partition_subexpts in subexperiments.items()
    # }

    print(
        f"Running {sum(len(subexpts) for subexpts in subexperiments.values())} circuits."
    )
    start = perf_counter()
    subexperiments = {
        label: [defer_mid_measurements(circ) for circ in subsystem_subexpts]
        for label, subsystem_subexpts in subexperiments.items()
    }

    results = {
        label: sampler.run(subsystem_subexpts, shots=20000).result()
        for label, subsystem_subexpts in subexperiments.items()
    }
    runtime = perf_counter() - start

    start = perf_counter()
    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    posttime = perf_counter() - start

    return reconstructed_expvals[0], {
        "ckt_pre": preptime,
        "ckt_run": runtime,
        "ckt_post": posttime,
    }


def ckt_numcoeffs(circuit: QuantumCircuit, num_samples: int = np.inf) -> int:
    obs = "Z" * circuit.num_qubits
    observable = SparsePauliOp([obs])

    qc_w_ancilla = cut_wires(circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)
    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )

    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    subexperiments, coefficients = generate_cutting_experiments_dummy(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )
    return len(coefficients)


def ckt_execute_dummy(circuit: QuantumCircuit, num_samples: int = np.inf) -> dict:
    obs = "Z" * circuit.num_qubits
    observable = SparsePauliOp([obs])

    start = perf_counter()
    qc_w_ancilla = cut_wires(circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)
    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )

    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    subexperiments, coefficients = generate_cutting_experiments_dummy(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )

    preptime = perf_counter() - start

    # isa_subexperiments = {
    #     label: [transpile(c) for c in partition_subexpts]
    #     for label, partition_subexpts in subexperiments.items()
    # }

    start = perf_counter()
    sampler = DummySampler()
    results = {
        label: sampler.run(subsystem_subexpts, shots=1).result()
        for label, subsystem_subexpts in subexperiments.items()
    }
    runtime = perf_counter() - start

    start = perf_counter()
    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    posttime = perf_counter() - start

    return {
        "ckt_pre": preptime,
        "ckt_run": runtime,
        "ckt_post": posttime,
        "ckt_num_instances": sum(len(subexpts) for subexpts in subexperiments.values()),
    }


def cut_ckt(circuit: QuantumCircuit, subcircuit_size: int) -> QuantumCircuit:
    optimization_settings = OptimizationParameters(max_gamma=6000000)

    # Specify the size of the QPUs available
    device_constraints = DeviceConstraints(qubits_per_subcircuit=subcircuit_size)

    cut_circuit, metadata = find_cuts(
        circuit, optimization_settings, device_constraints
    )
    print(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.'
    )
    return cut_circuit, {
        "overhead": metadata["sampling_overhead"],
        "num_cuts": len(metadata["cuts"]),
    }


from circuit_knitting.cutting.cutting_experiments import *
from circuit_knitting.cutting.cutting_experiments import (
    _append_measurement_register,
    _append_measurement_circuit,
    _remove_resets_in_zero_state,
    _remove_final_resets,
    _consolidate_resets,
    _get_bases,
    _get_mapping_ids_by_partition,
    _get_bases_by_partition,
)


def generate_cutting_experiments_dummy(
    circuits: QuantumCircuit | dict[Hashable, QuantumCircuit],
    observables: PauliList | dict[Hashable, PauliList],
    num_samples: int | float,
) -> tuple[
    list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]],
    list[tuple[float, WeightType]],
]:
    r"""
    Generate cutting subexperiments and their associated coefficients.

    If the input, ``circuits``, is a :class:`QuantumCircuit` instance, the
    output subexperiments will be contained within a 1D array, and ``observables`` is
    expected to be a :class:`PauliList` instance.

    If the input circuit and observables are specified by dictionaries with partition labels
    as keys, the output subexperiments will be returned as a dictionary which maps each
    partition label to a 1D array containing the subexperiments associated with that partition.

    In both cases, the subexperiment lists are ordered as follows:

        :math:`[sample_{0}observable_{0}, \ldots, sample_{0}observable_{N-1}, sample_{1}observable_{0}, \ldots, sample_{M-1}observable_{N-1}]`

    The coefficients will always be returned as a 1D array -- one coefficient for each unique sample.

    Args:
        circuits: The circuit(s) to partition and separate
        observables: The observable(s) to evaluate for each unique sample
        num_samples: The number of samples to draw from the quasi-probability distribution. If set
            to infinity, the weights will be generated rigorously rather than by sampling from
            the distribution.
    Returns:
        A tuple containing the cutting experiments and their associated coefficients.
        If the input circuits is a :class:`QuantumCircuit` instance, the output subexperiments
        will be a sequence of circuits -- one for every unique sample and observable. If the
        input circuits are represented as a dictionary keyed by partition labels, the output
        subexperiments will also be a dictionary keyed by partition labels and containing
        the subexperiments for each partition.
        The coefficients are always a sequence of length-2 tuples, where each tuple contains the
        coefficient and the :class:`WeightType`. Each coefficient corresponds to one unique sample.

    Raises:
        ValueError: ``num_samples`` must be at least one.
        ValueError: ``circuits`` and ``observables`` are incompatible types
        ValueError: :class:`SingleQubitQPDGate` instances must have their cut ID
            appended to the gate label so they may be associated with other gates belonging
            to the same cut.
        ValueError: :class:`SingleQubitQPDGate` instances are not allowed in unseparated circuits.
    """
    if isinstance(circuits, QuantumCircuit) and not isinstance(observables, PauliList):
        raise ValueError(
            "If the input circuits is a QuantumCircuit, the observables must be a PauliList."
        )
    if isinstance(circuits, dict) and not isinstance(observables, dict):
        raise ValueError(
            "If the input circuits are contained in a dictionary keyed by partition labels, the input observables must also be represented by such a dictionary."
        )
    if not num_samples >= 1:
        raise ValueError("num_samples must be at least 1.")

    # Retrieving the unique bases, QPD gates, and decomposed observables is slightly different
    # depending on the format of the execute_experiments input args, but the 2nd half of this function
    # can be shared between both cases.
    if isinstance(circuits, QuantumCircuit):
        is_separated = False
        subcircuit_dict: dict[Hashable, QuantumCircuit] = {"A": circuits}
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        subsystem_observables = {
            label: ObservableCollection(subobservables)
            for label, subobservables in subobservables_by_subsystem.items()
        }
        # Gather the unique bases from the circuit
        bases, qpd_gate_ids = _get_bases(circuits)
        subcirc_qpd_gate_ids: dict[Hashable, list[list[int]]] = {"A": qpd_gate_ids}

    else:
        is_separated = True
        subcircuit_dict = circuits
        # Gather the unique bases across the subcircuits
        subcirc_qpd_gate_ids, subcirc_map_ids = _get_mapping_ids_by_partition(
            subcircuit_dict
        )
        bases = _get_bases_by_partition(subcircuit_dict, subcirc_qpd_gate_ids)

        # Create the commuting observable groups
        subsystem_observables = {
            label: ObservableCollection(so) for label, so in observables.items()
        }

    # Sample the joint quasiprobability decomposition
    random_samples = generate_qpd_weights(bases, num_samples=num_samples)

    # Calculate terms in coefficient calculation
    kappa = np.prod([basis.kappa for basis in bases])
    num_samples = sum([value[0] for value in random_samples.values()])

    # Sort samples in descending order of frequency
    sorted_samples = sorted(random_samples.items(), key=lambda x: x[1][0], reverse=True)

    # Generate the output experiments and their respective coefficients
    subexperiments_dict: dict[Hashable, list[QuantumCircuit]] = defaultdict(list)
    coefficients: list[tuple[float, WeightType]] = []
    for z, (map_ids, (redundancy, weight_type)) in enumerate(sorted_samples):
        actual_coeff = np.prod(
            [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
        )
        sampled_coeff = (redundancy / num_samples) * (kappa * np.sign(actual_coeff))
        coefficients.append((sampled_coeff, weight_type))
        map_ids_tmp = map_ids
        for label, so in subsystem_observables.items():
            subcircuit = subcircuit_dict[label]
            if is_separated:
                map_ids_tmp = tuple(map_ids[j] for j in subcirc_map_ids[label])
            for j, cog in enumerate(so.groups):
                # new_qc = _append_measurement_register(subcircuit, cog)
                # decompose_qpd_instructions(
                #     new_qc, subcirc_qpd_gate_ids[label], map_ids_tmp, inplace=True
                # )
                # _append_measurement_circuit(new_qc, cog, inplace=True)
                new_qc = QuantumCircuit(1, 1)
                new_qc.measure(0, 0)
                subexperiments_dict[label].append(new_qc)

    # Remove initial and final resets from the subexperiments.  This will
    # enable the `Move` operation to work on backends that don't support
    # `Reset`, as long as qubits are not re-used.  See
    # https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/issues/452.
    # While we are at it, we also consolidate each run of multiple resets
    # (which can arise when re-using qubits) into a single reset.
    for subexperiments in subexperiments_dict.values():
        for circ in subexperiments:
            _remove_resets_in_zero_state(circ)
            _remove_final_resets(circ)
            _consolidate_resets(circ)

    # If the input was a single quantum circuit, return the subexperiments as a list
    subexperiments_out: list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]] = (
        dict(subexperiments_dict)
    )
    assert isinstance(subexperiments_out, dict)
    if isinstance(circuits, QuantumCircuit):
        assert len(subexperiments_out.keys()) == 1
        subexperiments_out = list(subexperiments_dict.values())[0]

    return subexperiments_out, coefficients
