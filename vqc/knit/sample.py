import itertools

from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister, Barrier

from vqc.circuit import VirtualCircuit, Fragment
from vqc.prob_distr import ProbDistr
from vqc.virtual_gate import VirtualGate


SampleIdType = tuple[int, ...]


def _sample_ids(vc: VirtualCircuit, fragment: Fragment) -> list[tuple[int, ...]]:
    vgate_instrs = [
        instr for instr in vc.data if isinstance(instr.operation, VirtualGate)
    ]
    conf_l = [
        tuple(range(len(instr.operation.configure())))
        if set(instr.qubits) & set(fragment)
        else (-1,)
        for instr in vgate_instrs
    ]
    return iter(itertools.product(*conf_l))


def _add_config_register(circuit: QuantumCircuit, size: int) -> QuantumRegister:
    num_conf_register = sum(1 for creg in circuit.cregs if creg.name.startswith("conf"))
    reg = ClassicalRegister(size, name=f"conf_{num_conf_register}")
    circuit.add_register(reg)
    return reg


def _circuit_on_index(circuit: QuantumCircuit, index: int) -> QuantumCircuit:
    qreg = QuantumRegister(1)
    qubit = circuit.qubits[index]
    circ = QuantumCircuit(qreg, *circuit.cregs)
    for instr in circuit.data:
        if len(instr.qubits) == 1 and instr.qubits[0] == qubit:
            circ.append(instr.operation, (qreg[0],), instr.clbits)
    return circ


def _circuit_with_config(
    vc: VirtualCircuit, fragment: Fragment, config_id: tuple[int, ...]
) -> QuantumCircuit:
    conf_circ = QuantumCircuit(fragment, *vc.cregs)
    conf_reg = _add_config_register(conf_circ, len(config_id))

    ctr = 0
    for instr in vc.data:
        if not isinstance(instr.operation, VirtualGate):
            if isinstance(instr.operation, Barrier):
                qubits = list(set(instr.qubits) & set(fragment))
                conf_circ.barrier(qubits)
            elif set(instr.qubits) <= set(fragment):
                conf_circ.append(instr.operation, instr.qubits, instr.clbits)
            continue

        if config_id[ctr] == -1:
            ctr += 1
            continue
        conf_def = instr.operation.configuration(config_id[ctr])

        if set(instr.qubits) <= set(fragment):
            conf_circ.append(conf_def.to_instruction(), instr.qubits, (conf_reg[ctr],))

        elif set(instr.qubits) & set(fragment):
            index = 0 if instr.qubits[0] in fragment else 1

            conf_circ.append(
                _circuit_on_index(conf_def, index).to_instruction(),
                (instr.qubits[index],),
                (conf_reg[ctr],),
            )
        else:
            raise RuntimeError("should not happen")

        ctr += 1
    return conf_circ


def _sample(vc: VirtualCircuit) -> dict[str, list[tuple[SampleIdType, QuantumCircuit]]]:
    samples = {}
    for frag in vc.fragments:
        new_samples = [
            (sample_id, _circuit_with_config(vc, frag, sample_id))
            for sample_id in _sample_ids(vc, frag)
        ]
        samples[frag.name] = new_samples
    return samples
