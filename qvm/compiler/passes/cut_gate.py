from ._qvm_pass import QVMPass


class GateBisectionPass(QVMPass):
    def __init__(self, fragment_sizes: set[int]):
        self._fragment_sizes = fragment_sizes

    def run(self, dag: DAG):
        pass


