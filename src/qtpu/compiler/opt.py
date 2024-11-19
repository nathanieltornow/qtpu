import logging
from typing import Callable

import optuna
import numpy as np
import cotengra as ctg
from qiskit.circuit import QuantumCircuit

from qtpu.transforms import wire_cuts_to_moves, remove_operations_by_name
from qtpu.compiler.ir import HybridCircuitIR
from qtpu.compiler.compress import CompressedIR, compress_2q_gates, compress_qubits
from qtpu.compiler.util import (
    get_leafs,
    sampling_overhead_tree,
    partition_girvan_newman,
)
from qtpu.compiler.success import estimated_error


logger = logging.getLogger("qtpu.compiler")


LOGGING = False

if not LOGGING:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


try:
    import kahypar

    partition_fn = ctg.pathfinders.path_kahypar.kahypar_subgraph_find_membership
except ImportError:
    print("KAHYPAR not found, using Girvan-Newman partitioning")
    partition_fn = partition_girvan_newman


def optimize(
    circuit: QuantumCircuit,
    # optimization arguments
    max_overhead: float = np.inf,
    terminate_fn: Callable[[CompressedIR, ctg.ContractionTree], bool] | None = None,
    choose_leaf: str = "nodes",  # {"qubits", "nodes", "random"}
    compress: str = "none",  # {"2q", "qubits", "none"}
    # partition arguments
    random_strength: float = 0.01,
    parts: int = 2,
    parts_decay: float = 0.5,
    super_optimize: str = "auto-hq",
    seed: int | None = None,
    **partition_opts,
):
    if terminate_fn is None and max_overhead == np.inf:
        raise ValueError("No stopping condition provided")

    circuit = remove_operations_by_name(circuit, {"barrier"}, inplace=False)
    ir = HybridCircuitIR(circuit)

    match compress:
        case "2q":
            ir = compress_2q_gates(ir)
        case "qubits":
            ir = compress_qubits(ir)
        case "none":
            ir = CompressedIR(ir, set())
        case _:
            raise ValueError(f"Unknown compression method: {compress}")

    match choose_leaf:
        case "qubits":
            choose_leaf_fn = max_qubits_leaf
        case "nodes":
            choose_leaf_fn = max_nodes_leaf
        case _:
            raise ValueError(f"Unknown leaf selection method: {choose_leaf}")

    tree = ir.contraction_tree()

    rng = ctg.core.get_rng(seed)
    rand_size_dict = ctg.core.jitter_dict(tree.size_dict.copy(), random_strength, rng)

    dynamic_imbalance = ("imbalance" in partition_opts) and (
        "imbalance_decay" in partition_opts
    )
    if dynamic_imbalance:
        imbalance = partition_opts.pop("imbalance")
        imbalance_decay = partition_opts.pop("imbalance_decay")
    else:
        imbalance = imbalance_decay = None

    dynamic_fix = partition_opts.get("fix_output_nodes", None) == "auto"

    while terminate_fn is None or not terminate_fn(ir, tree):
        if tree.is_complete():
            break
        tree_node = choose_leaf_fn(ir, tree)
        if tree_node is None:
            break

        ref_tree = tree.copy() if max_overhead < np.inf else None

        if tree_node is None:
            break
        subgraph = tuple(tree_node)
        subsize = len(subgraph)

        # relative subgraph size
        s = subsize / tree.N

        # let the target number of communities depend on subgraph size
        parts_s = max(int(s**parts_decay * parts), 2)

        # let the imbalance either rise or fall
        if dynamic_imbalance:
            if imbalance_decay >= 0:
                imbalance_s = s**imbalance_decay * imbalance
            else:
                imbalance_s = 1 - s**-imbalance_decay * (1 - imbalance)
            partition_opts["imbalance"] = imbalance_s

        if dynamic_fix:
            # for the top level subtree (s==1.0) we partition the outputs
            # nodes first into their own bi-partition
            parts_s = 2
            partition_opts["fix_output_nodes"] = s == 1.0

        # partition! get community membership list e.g.
        # [0, 0, 1, 0, 1, 0, 0, 2, 2, ...]
        inputs = tuple(map(tuple, tree.node_to_terms(subgraph)))
        output = tuple(tree.get_legs(tree_node))
        membership = partition_fn(
            inputs,
            output,
            rand_size_dict,
            parts=parts_s,
            seed=rng,
            **partition_opts,
        )

        # divide subgraph up e.g. if we enumerate the subgraph index sets
        # (0, 1, 2, 3, 4, 5, 6, 7, 8, ...) ->
        # ({0, 1, 3, 5, 6}, {2, 4}, {7, 8})
        new_subgs = tuple(
            map(ctg.core.node_from_seq, ctg.core.separate(subgraph, membership))
        )

        if len(new_subgs) == 1:
            continue

        # update tree structure with newly contracted subgraphs
        tree.contract_nodes(new_subgs, optimize=super_optimize, check=None)

        if sampling_overhead_tree(tree) > max_overhead:
            tree = ref_tree
            break

    sampling_overhead = sampling_overhead_tree(tree)

    if terminate_fn is not None and not terminate_fn(ir, tree):
        sampling_overhead = np.inf

    return wire_cuts_to_moves(ir.cut_circuit(get_leafs(tree))), {
        "sampling_overhead": sampling_overhead,
        "post_overhead": tree.contraction_cost(),
    }


def objective(
    trial: optuna.Trial,
    circuit: QuantumCircuit,
    error_fn: Callable[[QuantumCircuit], float],
    overhead_type: str = "sampling",  # {"sampling", "post"}
    max_overhead: int | tuple[int, int] | list[int] = np.inf,
    terminate_fn: Callable[[CompressedIR, ctg.ContractionTree], bool] | None = None,
    choose_leaf_methods: list[str] | None = None,
    compression_methods: list[str] | None = None,
) -> float:

    if isinstance(max_overhead, tuple):
        assert max_overhead[0] < max_overhead[1]
        max_overhead = trial.suggest_int("max_overhead", *max_overhead)
    elif isinstance(max_overhead, list):
        max_overhead = trial.suggest_categorical("max_overhead", max_overhead)

    compress = trial.suggest_categorical("compress", compression_methods)
    choose_leaf = trial.suggest_categorical("choose_leaf", choose_leaf_methods)

    # partition arguments
    random_strength = trial.suggest_float("random_strength", 0.01, 10.0)
    weight_edges = trial.suggest_categorical("weight_edges", ["const", "log"])
    imbalance = trial.suggest_float("imbalance", 0.01, 1.0)
    imbalance_decay = trial.suggest_float("imbalance_decay", -5, 5)
    parts = trial.suggest_int("parts", 2, 10)
    parts_decay = trial.suggest_float("parts_decay", 0.0, 1.0)
    mode = trial.suggest_categorical("mode", ["direct", "recursive"])
    objective = trial.suggest_categorical("objective", ["cut", "km1"])
    fix_output_nodes = trial.suggest_categorical("fix_output_nodes", ["auto", ""])

    cut_circuit, meta = optimize(
        circuit,
        # optimizer arguments
        terminate_fn=terminate_fn,
        max_overhead=max_overhead,
        choose_leaf=choose_leaf,
        compress=compress,
        # partition arguments
        random_strength=random_strength,
        weight_edges=weight_edges,
        imbalance=imbalance,
        imbalance_decay=imbalance_decay,
        parts=parts,
        parts_decay=parts_decay,
        mode=mode,
        objective=objective,
        fix_output_nodes=fix_output_nodes,
    )

    trial.set_user_attr("circuit", cut_circuit)

    overhead = (
        meta["sampling_overhead"]
        if overhead_type == "sampling"
        else meta["post_overhead"]
    )

    return overhead, error_fn(cut_circuit)


def hyper_optimize(
    circuit: QuantumCircuit,
    error_fn: Callable[[QuantumCircuit], float] | None = None,
    max_overhead: int | tuple[int, int] | list[int] = np.inf,
    terminate_fn: Callable[[CompressedIR, ctg.ContractionTree], bool] | None = None,
    choose_leaf_methods: list[str] | None = None,
    compression_methods: list[str] | None = None,
    # optuna args
    n_trials: int = 100,
    show_progress_bar: bool = False,
) -> list[QuantumCircuit]:
    if error_fn is None:
        error_fn = estimated_error

    if compression_methods is None:
        compression_methods = ["qubits", "2q", "none"]

    if choose_leaf_methods is None:
        choose_leaf_methods = ["qubits", "nodes"]

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(
        lambda trial: objective(
            trial,
            circuit=circuit,
            error_fn=error_fn,
            max_overhead=max_overhead,
            terminate_fn=terminate_fn,
            choose_leaf_methods=choose_leaf_methods,
            compression_methods=compression_methods,
        ),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    return study


def max_qubits_leaf(
    ir: CompressedIR, tree: ctg.ContractionTree
) -> frozenset[int] | None:
    if len(tree.childless) == 1:
        return next(iter(tree.childless))
    if len(tree.childless) == 0:
        return None
    return max(tree.childless, key=lambda x: ir.num_qubits(x))


def max_nodes_leaf(
    ir: CompressedIR, tree: ctg.ContractionTree
) -> frozenset[int] | None:
    if len(tree.childless) == 1:
        return next(iter(tree.childless))
    if len(tree.childless) == 0:
        return None
    return max(tree.childless, key=lambda x: len(ir.decompress_nodes(x)))
