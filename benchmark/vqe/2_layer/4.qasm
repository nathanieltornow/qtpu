// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3)]
qreg q[4];
creg m0[4];  // Measurement: q(0),q(1),q(2),q(3)


ry(pi*2.6769341457) q[0];
ry(pi*3.5696454153) q[1];
ry(pi*1.5052359548) q[2];
ry(pi*4.4955908467) q[3];
rz(pi*2.3340435781) q[0];
rz(pi*4.2531677517) q[1];
rz(pi*1.0140527059) q[2];
rz(pi*0.0132446239) q[3];
cx q[0],q[1];
cx q[1],q[2];
ry(pi*2.9317676013) q[0];
cx q[2],q[3];
ry(pi*0.8440951374) q[1];
rz(pi*1.5273499017) q[0];
ry(pi*2.9164988898) q[2];
ry(pi*3.5798871312) q[3];
rz(pi*0.0934943772) q[1];
ry(pi*2.4237359664) q[0];
rz(pi*2.3948767678) q[2];
rz(pi*1.4024493118) q[3];
ry(pi*2.5980418505) q[1];
rz(pi*1.3013983728) q[0];
ry(pi*1.999241539) q[2];
ry(pi*4.5743250762) q[3];
rz(pi*3.7976394972) q[1];
rz(pi*2.5927488381) q[2];
rz(pi*3.0787360569) q[3];
cx q[0],q[1];
cx q[1],q[2];
ry(pi*1.1428824265) q[0];
cx q[2],q[3];
ry(pi*0.0601113771) q[1];
rz(pi*0.3185880251) q[0];
ry(pi*1.2357728408) q[2];
ry(pi*1.7078764194) q[3];
rz(pi*1.727631561) q[1];
rz(pi*2.012451851) q[2];
rz(pi*3.0006975588) q[3];

// Gate: cirq.MeasurementGate(4, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3)'), ())
measure q[0] -> m0[0];
measure q[1] -> m0[1];
measure q[2] -> m0[2];
measure q[3] -> m0[3];
