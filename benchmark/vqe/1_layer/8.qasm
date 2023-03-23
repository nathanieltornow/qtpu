// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7)]
qreg q[8];
creg m0[8];  // Measurement: q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7)


ry(pi*2.9409441193) q[0];
ry(pi*0.7818834827) q[1];
ry(pi*2.4029996324) q[2];
ry(pi*1.5622247443) q[3];
ry(pi*2.5460571535) q[4];
ry(pi*0.4531739287) q[5];
ry(pi*3.5441084813) q[6];
ry(pi*0.4589909468) q[7];
rz(pi*2.5192394488) q[0];
rz(pi*-0.5508863379) q[1];
rz(pi*2.0108676208) q[2];
rz(pi*2.9981633678) q[3];
rz(pi*2.9934989722) q[4];
rz(pi*0.9989932312) q[5];
rz(pi*4.0055827108) q[6];
rz(pi*2.9956457856) q[7];
cx q[0],q[1];
cx q[1],q[2];
ry(pi*0.1911315611) q[0];
cx q[2],q[3];
ry(pi*1.0452222719) q[1];
rz(pi*2.9695876613) q[0];
cx q[3],q[4];
ry(pi*0.8021666141) q[2];
rz(pi*1.4914056701) q[1];
cx q[4],q[5];
ry(pi*1.2725370977) q[3];
rz(pi*3.0096418402) q[2];
cx q[5],q[6];
ry(pi*0.299199954) q[4];
rz(pi*-0.0103557372) q[3];
cx q[6],q[7];
ry(pi*2.3014057337) q[5];
rz(pi*4.0090073899) q[4];
ry(pi*0.298123905) q[6];
ry(pi*-0.313921201) q[7];
rz(pi*0.0094568749) q[5];
rz(pi*3.9958757905) q[6];
rz(pi*0.9979778936) q[7];

// Gate: cirq.MeasurementGate(8, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7)'), ())
measure q[0] -> m0[0];
measure q[1] -> m0[1];
measure q[2] -> m0[2];
measure q[3] -> m0[3];
measure q[4] -> m0[4];
measure q[5] -> m0[5];
measure q[6] -> m0[6];
measure q[7] -> m0[7];
