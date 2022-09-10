
from qvm.circuit import DistributedCircuit
from .compiler import Compiler

class MapomaticCompiler(Compiler):
    
    def run(self, vc: DistributedCircuit) -> DistributedCircuit:
        return super().run(vc)
