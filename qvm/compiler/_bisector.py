import abc

import networkx as nx


class Bisector(abc.ABC):
    @abc.abstractmethod
    def bisect(self, graph: nx.Graph) -> tuple[set, set]:
        pass


class GirvanNewmanBisector(Bisector):
    def bisect(self, graph: nx.Graph) -> tuple[set, set]:
        components = next(nx.community.girvan_newman(graph))
        assert len(components) == 2
        return (set(components[0]), set(components[1]))
