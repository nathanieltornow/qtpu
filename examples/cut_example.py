from qvm.compiler import cut, virtualize_optimal_gates

from _example_circuit import example_circuit


def num_cnots(circuit):
    return sum(1 for instr in circuit.data if instr.operation.name == "cx")

def main():
    
    QPU_SIZE = 3
    
    circuit = example_circuit(8, 2, "circular")
    print(circuit)
    
    print("Virtualizing gates:\n")
    cut_circ = virtualize_optimal_gates(circuit, 3)
    
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeMontrealV2
    
    backend = FakeMontrealV2()
    t1 = transpile(circuit, backend=backend, optimization_level=3)
    t2 = transpile(cut_circ, backend=backend, optimization_level=3)
    
    
    
    print("CNOTs in original circuit:", num_cnots(t1))
    print("CNOTs in cut circuit:", num_cnots(t2))

    # print("Cutting wires:\n")
    # cut_circuit = cut(circuit, QPU_SIZE, technique="wire_optimal")
    # print(cut_circuit)

    # print("Cutting gates optimal:\n")
    # cut_circuit = cut(circuit, QPU_SIZE, technique="gate_optimal")
    # print(cut_circuit)

    # print("Cutting gates bisection:\n")
    # cut_circuit = cut(circuit, QPU_SIZE, technique="gate_bisection")
    # print(cut_circuit)

    # print("Qubit reuse:\n")
    # cut_circuit = cut(circuit, QPU_SIZE, technique="qubit_reuse")
    # print(cut_circuit)


if __name__ == "__main__":
    main()
