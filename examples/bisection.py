from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from vqc import Knitter, cut, Bisection
from vqc.knit.knit import _knit

if __name__ == "__main__":
    # initialize a 4-qubit circuit
    # circuit = QuantumCircuit.from_qasm_file("examples/qasm/circuit1.qasm")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.h(1)
    circuit.cx(0, 1)
    circuit.measure_all()

    # build and run a transpiler using the bisection pass.

    virt_circ = cut(circuit, Bisection())
    print(virt_circ)

    knitter = Knitter(virt_circ)
    print(knitter.samples())

    results = {}
    for name, frag_circs in knitter.samples().items():
        print("asdas", frag_circs)
        results[name] = (
            AerSimulator()
            .run([circ.decompose() for circ in frag_circs])
            .result()
            .get_counts()
        )

    print(results)
    res = knitter.knit(results)

    # for sample in knitter.samples()[1]:
    #     print(sample.decompose())

    # dist_circ = VirtualCircuit.from_circuit(cut_circ)
    # print(dist_circ)

    # result = execute_virtual_circuit(dist_circ, 1000)
    # print(result)

    from vqc.bench.fidelity import fidelity

    fid = fidelity(circuit, res.counts(10000))
    print(fid)
