// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5)]
qreg q[6];
creg m0[6];  // Measurement: q(0),q(1),q(2),q(3),q(4),q(5)


ry(pi*1.9999052023) q[0];
ry(pi*3.0002403444) q[1];
ry(pi*1.0025162662) q[2];
ry(pi*2.0020103479) q[3];
ry(pi*0.958548921) q[4];
ry(pi*1.1831429484) q[5];
rz(pi*1.2131861559) q[0];
rz(pi*3.2923175669) q[1];
rz(pi*1.2279998501) q[2];
rz(pi*3.4764334118) q[3];
rz(pi*0.9797040017) q[4];
rz(pi*2.9986232898) q[5];
cx q[0],q[1];
cx q[1],q[2];
ry(pi*4.1684126904) q[0];
cx q[2],q[3];
ry(pi*0.8329547441) q[1];
rz(pi*1.9996546246) q[0];
cx q[3],q[4];
ry(pi*1.8329302786) q[2];
rz(pi*1.0002377031) q[1];
cx q[4],q[5];
ry(pi*0.1684015663) q[3];
rz(pi*3.0001889495) q[2];
ry(pi*3.1529263039) q[4];
ry(pi*-0.0091071324) q[5];
rz(pi*3.9990601039) q[3];
rz(pi*1.9974823698) q[4];
rz(pi*1.9993996457) q[5];

// Gate: cirq.MeasurementGate(6, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3),q(4),q(5)'), ())
measure q[0] -> m0[0];
measure q[1] -> m0[1];
measure q[2] -> m0[2];
measure q[3] -> m0[3];
measure q[4] -> m0[4];
measure q[5] -> m0[5];
