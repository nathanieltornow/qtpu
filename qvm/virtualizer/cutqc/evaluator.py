import itertools, copy, random
import numpy as np
from time import time
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate
from qiskit.quantum_info import Statevector

# from qiskit_helper_functions.non_ibmq_functions import read_dict, find_process_jobs, evaluate_circ


def run_subcircuit_instances(
    subcircuits, subcircuit_instances, eval_mode, num_shots_fn
):
    """
    subcircuit_instance_probs[subcircuit_idx][subcircuit_instance_idx] = measured probability
    """
    subcircuit_instance_probs = {}
    for subcircuit_idx in subcircuit_instances:
        subcircuit_instance_probs[subcircuit_idx] = {}
        num_shots = num_shots_fn(subcircuits[subcircuit_idx])
        for init_meas in subcircuit_instances[subcircuit_idx]:
            subcircuit_instance_idx = subcircuit_instances[subcircuit_idx][init_meas]
            if subcircuit_instance_idx not in subcircuit_instance_probs[subcircuit_idx]:
                # print('Subcircuit %d instance %d'%(subcircuit_idx,subcircuit_instance_idx))
                subcircuit_instance = modify_subcircuit_instance(
                    subcircuit=subcircuits[subcircuit_idx],
                    init=init_meas[0],
                    meas=init_meas[1],
                )
                subcircuit_inst_prob = simulate_subcircuit(
                    subcircuit=subcircuit_instance,
                    eval_mode=eval_mode,
                    num_shots=num_shots,
                )
                mutated_meas = mutate_measurement_basis(meas=init_meas[1])
                for meas in mutated_meas:
                    measured_prob = measure_prob(
                        unmeasured_prob=subcircuit_inst_prob, meas=meas
                    )
                    mutated_subcircuit_instance_idx = subcircuit_instances[
                        subcircuit_idx
                    ][(init_meas[0], meas)]
                    subcircuit_instance_probs[subcircuit_idx][
                        mutated_subcircuit_instance_idx
                    ] = measured_prob
                    # print('Measured instance %d'%mutated_subcircuit_instance_idx)
    return subcircuit_instance_probs


def mutate_measurement_basis(meas):
    """
    I and Z measurement basis correspond to the same logical circuit
    """
    if all(x != "I" for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != "I":
                mutated_meas.append([x])
            else:
                mutated_meas.append(["I", "Z"])
        mutated_meas = list(itertools.product(*mutated_meas))
        return mutated_meas


def modify_subcircuit_instance(subcircuit, init, meas):
    """
    Modify the different init, meas for a given subcircuit
    Returns:
    Modified subcircuit_instance
    List of mutated measurements
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)
    return subcircuit_instance_circuit


def simulate_subcircuit(subcircuit, eval_mode, num_shots):
    """
    Simulate a subcircuit
    """
    from qiskit_aer import StatevectorSimulator

    if eval_mode == "sv":
        statevector = StatevectorSimulator().run(subcircuit).result().get_statevector()

        subcircuit_inst_prob = Statevector(statevector).probabilities()
    # elif eval_mode=='qasm':
    #     subcircuit_inst_prob = evaluate_circ(circuit=subcircuit,backend='noiseless_qasm_simulator',options={'num_shots':num_shots})
    else:
        raise NotImplementedError
    return subcircuit_inst_prob


def measure_prob(unmeasured_prob, meas):
    if meas.count("comp") == len(meas):
        return unmeasured_prob
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        # print('Measuring in',meas)
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            # TODO: Add states merging here. Change effective_state to merged_bin
            measured_prob[effective_state] += sigma * p
        return measured_prob


def measure_state(full_state, meas):
    """
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1]):
        if meas_bit == "1" and meas_basis != "I" and meas_basis != "comp":
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))
    return sigma, effective_state
