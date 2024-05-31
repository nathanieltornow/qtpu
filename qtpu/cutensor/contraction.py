from concurrent.futures import ThreadPoolExecutor


from qtpu.contraction import evaluate_quantum_tensor
from qtpu.tensor import HybridTensorNetwork
from qtpu.qinterface import QuantumInterface, BackendInterface

import cupy as cp
from cuquantum import Network


def contract_gpu(
    hybrid_tn: HybridTensorNetwork,
    shots: int,
    qiface: QuantumInterface | None = None,
) -> float:
    if qiface is None:
        qiface = BackendInterface()

    # load classical tensors to GPU
    classical_tensors = [
        cp.array(tens.data, dtype=cp.float32) for tens in hybrid_tn.classical_tensors
    ]

    # evaluate quantum tensors
    quantum_tensors = hybrid_tn.quantum_tensors
    with ThreadPoolExecutor() as executor:
        futs = [
            executor.submit(evaluate_quantum_tensor, qtens, qiface, shots, "expval")
            for qtens in quantum_tensors
        ]
        # load results to GPU
        eval_tensors = [cp.array(fut.result().data, dtype=cp.float32) for fut in futs]

    # contract tensors
    with Network(hybrid_tn.equation(), *(eval_tensors + classical_tensors)) as tn:
        path, info = tn.contract_path()
        tn.autotune(iterations=5)
        result = tn.contract()

    return result


# def graph_to_einsum_eq(graph: nx.Graph) -> tuple[str, dict[str, int]]:
#     graph = graph.to_undirected(as_view=True)
#     edge_to_char_size = {
#         (u, v): (chr(200 + i), size)
#         for i, (u, v, size) in enumerate(graph.edges(data="weight"))
#     }

#     terms = []
#     for u in sorted(graph.nodes):
#         term = []
#         for v in graph.neighbors(u):
#             if (u, v) in edge_to_char_size:
#                 term.append(str(edge_to_char_size[(u, v)][0]))
#             else:
#                 term.append(str(edge_to_char_size[(v, u)][0]))
#         terms.append("".join(term))

#     return ",".join(terms) + "->", {
#         char: size for char, size in edge_to_char_size.values()
#     }


# def contract_path_from_graph(
#     graph: nx.Graph,
# ) -> tuple[list[tuple[int, int]], OptimizerInfo]:
#     eq, size_dict = graph_to_einsum_eq(graph)

#     ops = []
#     for tensor_descr in eq[:-2].split(","):
#         shape = []
#         for char in tensor_descr:
#             shape += [size_dict[char]]
#         ops.append(np.zeros(*shape))
#     return contract_path(eq, *ops)


# def contraction_tree_from_graph(graph: nx.Graph) -> ctg.ContractionTree:
#     eq, size_dict = graph_to_einsum_eq(graph)
#     return ctg.ContractionTree.from_path(
#         inputs=list(eq[:-2].split(",")),
#         output=tuple(),
#         size_dict=size_dict,
#         path=contract_path_from_graph(graph)[0],
#     )


# def traverse_tree(tree: ctg.ContractionTree) -> Iterator[frozenset[int]]:
#     queue = [tree.root]
#     while queue:
#         node = queue.pop(0)
#         yield node
#         if node in tree.children:
#             queue.append(tree.children[node][0])
#             queue.append(tree.children[node][1])
