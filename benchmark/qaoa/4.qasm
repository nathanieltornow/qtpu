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
cx q[0],q[2];
rz(pi*0.1959530952) q[2];
cx q[0],q[2];
cx q[2],q[3];
cx q[0],q[1];
rz(pi*-0.1959530952) q[3];
rz(pi*0.1959530952) q[1];
cx q[2],q[3];
cx q[0],q[1];
cx q[1],q[3];
rz(pi*-0.1959530952) q[3];
cx q[1],q[3];
cx q[1],q[2];
cx q[0],q[3];
rz(pi*-0.1959530952) q[2];
rz(pi*-0.1959530952) q[3];
cx q[1],q[2];
cx q[0],q[3];
rx(pi*0.7500217822) q[0];
rx(pi*0.7500217822) q[1];
rx(pi*0.7500217822) q[2];
rx(pi*0.7500217822) q[3];

// Gate: cirq.MeasurementGate(4, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3)'), ())
measure q[0] -> m0[0];
measure q[1] -> m0[1];
measure q[2] -> m0[2];
measure q[3] -> m0[3];
