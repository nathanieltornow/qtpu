from random import sample
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from vqc import Knitter, cut, Bisection
from vqc.bench.fidelity import fidelity

if __name__ == "__main__":
    circuit = QuantumCircuit.from_qasm_file("examples/qasm/circuit1.qasm")

    # Cut the circuit using the Bisection cutter
    virt_circ = cut(circuit, Bisection())
    # print the virtual circuit, virtual gates are Barriers
    print(virt_circ)

    # print both fragments as circuits
    for fragment in virt_circ.fragments:
        print(virt_circ.fragment_as_circuit(fragment))

    # define a knitter the virtualization process
    knitter = Knitter(virt_circ)
    # get all circuit samples needed for the virtualization
    # the samples are a dictionary with keys being the fragment ids and values being
    # the circuits to execute for each fragment
    samples: dict[str, list[QuantumCircuit]] = knitter.samples()

    # execute the samples and store them accoring to the samples dictionary
    # with the fragment ids as keys and the list of counts as values
    results = {}
    for name, frag_circs in samples.items():
        results[name] = (
            AerSimulator()
            .run([circ.decompose() for circ in frag_circs])
            .result()
            .get_counts()
        )

    # knit the results together to get the final probability distribution
    prob_distr = knitter.knit(results)

    # store the probability distribution as counts and get the hellinger fidelity
    counts = prob_distr.counts(10000)
    fid = fidelity(circuit, counts)
    print(fid)
