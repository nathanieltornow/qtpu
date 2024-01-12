from pathlib import Path

import networkx as nx

from ._graphs import TNGraph
from ._cutter import Cutter


class ASPCutter(Cutter):
    def __init__(self) -> None:
        super().__init__()

    def _cut(self, tn_graph: TNGraph) -> set[tuple[int, int]]:
        return super()._cut(tn_graph)

    # def __init__(self, max_cost: int) -> None:
    #     self._max_cost = max_cost
    #     super().__init__()

    # def _cut_tn(self, tn_graph: nx.Graph) -> list[tuple[int, int]]:
    #     asp = self._cut_graph_to_asp(tn_graph)
    #     with open(str(str(Path(__file__).parent / "opt_cut.ll")), "r") as f:
    #         asp += f.read()

    #     num_partitions = 2
    #     asp += f"#const num_partitions = {num_partitions}.\n"

    #     print(asp)

    #     symbols = self.get_result_symbols(asp)
    #     print(symbols)

    # def _cut_tn(self, cut_graph: nx.Graph) -> list[tuple[int, int]]:
    #     asp = self._cut_graph_to_asp(cut_graph)
    #     with open(str(str(Path(__file__).parent / "opt_cut.ll")), "r") as f:
    #         asp += f.read()
    #     symbols = self.get_optimal_symbols(asp)
    #     print(symbols)

    # @staticmethod
    # def _cut_graph_to_asp(cut_graph: nx.Graph) -> str:
    #     asp = ""
    #     for node, qubit_idx in cut_graph.nodes.data("qubit_idx"):
    #         asp += f"node({node}, {qubit_idx}).\n"
    #     for node1, node2 in cut_graph.edges:
    #         asp += f"edge({node1}, {node2}).\n"
    #     return asp

    # @staticmethod
    # def get_result_symbols(asp: str, num_answers: int = 0) -> list:
    #     try:
    #         from clingo.control import Control
    #     except ImportError as e:
    #         raise ImportError("ASPCutter requires clingo") from e

    #     control = Control()
    #     control.configuration.solve.models = num_answers  # type: ignore
    #     control.add("base", [], asp)
    #     control.ground([("base", [])])
    #     solve_result = control.solve(yield_=True)  # type: ignore
    #     opt_model = None
    #     for model in solve_result:  # type: ignore
    #         opt_model = model

    #     if opt_model is None:
    #         raise ValueError("No solution found.")

    #     return list(opt_model.symbols(shown=True))  # type: ignore
