from pathlib import Path

import networkx as nx

from .cutter import TNCutter


class ASPCutter(TNCutter):
    def __init__(self, max_cost: int) -> None:
        self._max_cost = max_cost
        super().__init__()

    def _cut_tn(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
        asp = self._cut_graph_to_asp(cut_graph)
        with open(str(str(Path(__file__).parent / "opt_cut.ll")), "r") as f:
            asp += f.read()
        symbols = self.get_optimal_symbols(asp)
        print(symbols)

    @staticmethod
    def _cut_graph_to_asp(cut_graph: nx.Graph) -> str:
        asp = ""
        for node, qubit_idx in cut_graph.nodes.data("qubit_idx"):
            asp += f"node({node}, {qubit_idx}).\n"
        for node1, node2 in cut_graph.edges:
            asp += f"edge({node1}, {node2}).\n"
        return asp

    @staticmethod
    def get_optimal_symbols(asp: str) -> list["clingo.Symbol"]:
        try:
            from clingo.control import Control
        except ImportError as e:
            raise ImportError("ASPCutter requires clingo") from e

        control = Control()
        control.configuration.solve.models = 0  # type: ignore
        control.add("base", [], asp)
        control.ground([("base", [])])
        solve_result = control.solve(yield_=True)  # type: ignore
        opt_model = None
        for model in solve_result:  # type: ignore
            opt_model = model

        if opt_model is None:
            raise ValueError("No solution found.")

        return list(opt_model.symbols(shown=True))  # type: ignore
