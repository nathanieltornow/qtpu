
from qvm.circuit import VirtualCircuit
from .compiler import Compiler

class MapomaticCompiler(Compiler):
    
    def run(self, vc: VirtualCircuit) -> VirtualCircuit:
        return super().run(vc)
