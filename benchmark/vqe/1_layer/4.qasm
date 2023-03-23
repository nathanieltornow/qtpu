// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];
creg m0[4];  // Measurement: q(0),q(1),q(2),q(3)


ry(pi*1.0146834587) q[0];
ry(pi*-0.0293393862) q[1];
ry(pi*0.9460210148) q[2];
ry(pi*0.7606804865) q[3];
rz(pi*4.6437505466) q[0];
rz(pi*4.2375150459) q[1];
rz(pi*2.7052922659) q[2];
rz(pi*1.8837783767) q[3];
cx q[0],q[1];
cx q[1],q[2];
ry(pi*0.1703807852) q[0];
cx q[2],q[3];
ry(pi*1.8282051949) q[1];
rz(pi*3.0010762779) q[0];
ry(pi*0.8440484616) q[2];
ry(pi*4.0686435408) q[3];
rz(pi*4.0044689995) q[1];
rz(pi*0.0512254487) q[2];
rz(pi*2.1463263935) q[3];

// Gate: cirq.MeasurementGate(4, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3)'), ())
measure q[0] -> m0[0];
measure q[1] -> m0[1];
measure q[2] -> m0[2];
measure q[3] -> m0[3];
