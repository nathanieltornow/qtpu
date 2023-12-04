from qiskit.circuit import QuantumCircuit

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg meas[16];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
cz q[0],q[1];
cz q[6],q[7];
cz q[8],q[9];
cz q[14],q[15];
t q[2];
t q[3];
t q[4];
t q[5];
t q[10];
t q[11];
t q[12];
t q[13];
cz q[4],q[8];
cz q[6],q[10];
ry(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[7];
ry(pi/2) q[9];
rx(pi/2) q[14];
rx(pi/2) q[15];
cz q[1],q[2];
cz q[9],q[10];
t q[0];
t q[7];
t q[14];
t q[15];
ry(pi/2) q[4];
rx(pi/2) q[8];
rx(pi/2) q[6];
cz q[0],q[4];
cz q[9],q[13];
cz q[2],q[6];
cz q[11],q[15];
t q[8];
ry(pi/2) q[1];
ry(pi/2) q[10];
cz q[4],q[5];
cz q[2],q[3];
cz q[12],q[13];
cz q[10],q[11];
t q[1];
rx(pi/2) q[0];
ry(pi/2) q[9];
ry(pi/2) q[6];
ry(pi/2) q[15];
cz q[5],q[9];
cz q[7],q[11];
t q[0];
t q[6];
t q[15];
rx(pi/2) q[4];
rx(pi/2) q[2];
ry(pi/2) q[3];
ry(pi/2) q[12];
ry(pi/2) q[13];
rx(pi/2) q[10];
cz q[5],q[6];
cz q[13],q[14];
t q[4];
t q[2];
t q[3];
t q[12];
t q[10];
rx(pi/2) q[9];
rx(pi/2) q[7];
rx(pi/2) q[11];
cz q[1],q[5];
cz q[8],q[12];
cz q[3],q[7];
cz q[10],q[14];
t q[9];
t q[11];
rx(pi/2) q[6];
rx(pi/2) q[13];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
"""

circuit = QuantumCircuit.from_qasm_str(qasm)

from qvm.compiler import QVMCompiler, CutterCompiler
from qvm.compiler.virtualization import BisectionPass

compiler = QVMCompiler(virt_passes=[BisectionPass(size_to_reach=8)])
# compiler = CutterCompiler(8)
virt_circuit = compiler.run(circuit, 4)

print(virt_circuit.circuit)

from qvm import run

res, _ = run(virt_circuit)
print(res)