// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];
creg m0[4];  // Measurement: q(0),q(1),q(2),q(3)


h q[0];
h q[1];
h q[2];
h q[3];
rz(pi*-0.75) q[0];
rz(pi*-0.75) q[1];
rz(pi*-0.75) q[2];
rz(pi*-0.75) q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(pi*-0.5) q[1];
cx q[0],q[1];
cx q[1],q[2];
h q[0];
rz(pi*-0.5) q[2];
rz(pi*-0.7493178647) q[0];
cx q[1],q[2];
h q[0];
cx q[2],q[3];
h q[1];
rz(pi*-0.5) q[3];
rz(pi*-0.7493178647) q[1];
cx q[2],q[3];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(pi*-0.7493178647) q[2];
rz(pi*-0.7493178647) q[3];
rz(pi*-0.5) q[1];
h q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
h q[0];
rz(pi*-0.5) q[2];
rz(pi*-0.7479542144) q[0];
cx q[1],q[2];
h q[0];
cx q[2],q[3];
h q[1];
rz(pi*-0.5) q[3];
rz(pi*-0.7479542144) q[1];
cx q[2],q[3];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(pi*-0.7479542144) q[2];
rz(pi*-0.7479542144) q[3];
rz(pi*-0.5) q[1];
h q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
h q[0];
rz(pi*-0.5) q[2];
rz(pi*-0.7459102894) q[0];
cx q[1],q[2];
h q[0];
cx q[2],q[3];
h q[1];
rz(pi*-0.5) q[3];
rz(pi*-0.7459102894) q[1];
cx q[2],q[3];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(pi*-0.7459102894) q[2];
rz(pi*-0.7459102894) q[3];
rz(pi*-0.5) q[1];
h q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.5) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(pi*-0.5) q[3];
cx q[2],q[3];

// Gate: cirq.MeasurementGate(4, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3)'), ())
measure q[0] -> m0[0];
measure q[1] -> m0[1];
measure q[2] -> m0[2];
measure q[3] -> m0[3];
