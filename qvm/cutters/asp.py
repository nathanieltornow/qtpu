from pathlib import Path

import networkx as nx

from ._graphs import TNGraph
from ._cutter import Cutter


class ASPCutter(Cutter):
    def __init__(self, max_cost: int) -> None:
        self._max_cost = max_cost
        super().__init__()

    def _cut(self, tn_graph: TNGraph) -> set[tuple[int, int]]:
        asp = self._tn_graph_to_asp(tn_graph)
        with open(str(str(Path(__file__).parent / "opt_cut.lp")), "r") as f:
            asp += f.read()

        symbols = self.get_result_symbols(asp, 0)

        cut_edges = set()

        for symbol in symbols:
            if symbol.name == "vertex_cost":
                print(symbol.arguments)
            if symbol.name == "cut_edge":
                cut_edges.add((symbol.arguments[0].number, symbol.arguments[1].number))
        return cut_edges

    @staticmethod
    def _tn_graph_to_asp(tn_graph: TNGraph) -> str:
        asp = ""
        for node, qubit_idx in tn_graph.nodes.data("qubit_idx"):
            asp += f"vertex({node}, {qubit_idx}).\n"
        for node1, node2 in tn_graph.edges:
            asp += f"edge({node1}, {node2}).\n"
        return asp

    @staticmethod
    def get_result_symbols(asp: str, num_answers: int = 0) -> list:
        try:
            from clingo.control import Control
        except ImportError as e:
            raise ImportError("ASPCutter requires clingo") from e

        control = Control()
        control.configuration.solve.models = num_answers  # type: ignore
        control.add("base", [], asp)
        control.ground([("base", [])])
        solve_result = control.solve(yield_=True)  # type: ignore
        opt_model = None
        for model in solve_result:  # type: ignore
            opt_model = model

        if opt_model is None:
            raise ValueError("No solution found.")
        return opt_model.symbols(shown=True)  # type: ignore
