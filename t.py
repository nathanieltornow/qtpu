from bench.util.circuits import qaoa, hamsim, two_local

if __name__ == "__main__":
    circ = hamsim(4, 2)
    print(circ)

    from qvm.compiler.dag import DAG, get_qubit_dependencies
    from qvm.compiler.virtualization.reduce_deps import CircularDependencyBreaker, QubitDependencyMinimizer
    from qvm.compiler.virtualization.wire_decomp import OptimalWireCutter
    from qvm.compiler.qubit_reuse import QubitReuseCompiler
    from qvm.virtual_circuit import VirtualCircuit

    cut_circ = OptimalWireCutter(3).run(circ)

    print(cut_circ)    

    # virt = VirtualCircuit(cut_circ)

    # QubitReuseCompiler(3).run(virt)


    # for fragcirc in virt.fragment_circuits.values():
    #     dag = DAG(fragcirc)
    #     print(dag.to_circuit())


    # from qvm.run import run_virtualizer
    # from qvm.qvm_runner import LocalBackendRunner

    # runner = LocalBackendRunner()

    # res, _ = run_virtualizer(virt, runner)

    # from bench.util._util import compute_fidelity

    # fid1, fid2 = compute_fidelity(circ, res, runner)
    # print(fid1, fid2)

#     # dag = DAG(circ)
#     # qdg = get_qubit_dependencies(dag)
#     # from pprint import pprint

#     # pprint(qdg)
#     # import networkx as nx
#     # import matplotlib.pyplot as plt

#     # print(qdg.number_of_edges())

#     # nx.draw(qdg)
#     # plt.show()

