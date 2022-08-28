# from typing import Any, Dict
# from qiskit.providers import Backend

# from qvm.circuit import VirtualCircuit
# from qvm.execution.exec import execute_fragmented_circuit
# from qvm.transpiler.transpiled_circuit import TranspiledVirtualCircuit


# def execute(virtual_circuit: VirtualCircuit, shots: int = 10000) -> Dict[str, int]:
#     if not isinstance(virtual_circuit, TranspiledVirtualCircuit):
#         virtual_circuit = TranspiledVirtualCircuit(virtual_circuit)
#     res = execute_fragmented_circuit(virtual_circuit)
#     return res.counts(shots)
