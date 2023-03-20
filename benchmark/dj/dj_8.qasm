// Benchmark was created by MQT Bench on 2022-12-15
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 0.2.2
// Qiskit version: {'qiskit-terra': '0.22.3', 'qiskit-aer': '0.11.1', 'qiskit-ignis': '0.7.0', 'qiskit-ibmq-provider': '0.19.2', 'qiskit': '0.39.3', 'qiskit-nature': '0.5.1', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.4.0', 'qiskit-machine-learning': '0.5.0'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[7];
h q[0];
u2(0,0) q[1];
u2(0,0) q[2];
u2(0,0) q[3];
u2(0,0) q[4];
h q[5];
h q[6];
u2(-pi,-pi) q[7];
cx q[0],q[7];
h q[0];
cx q[1],q[7];
u2(-pi,-pi) q[1];
cx q[2],q[7];
u2(-pi,-pi) q[2];
cx q[3],q[7];
u2(-pi,-pi) q[3];
cx q[4],q[7];
u2(-pi,-pi) q[4];
cx q[5],q[7];
h q[5];
cx q[6],q[7];
h q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
