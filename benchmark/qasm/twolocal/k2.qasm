OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(5.66985611832717,-pi/2,pi/2) q0; }
qreg q[2];
creg meas[2];
u2(0,pi) q[0];
u2(0,pi) q[1];
cx q[0],q[1];
rz(pi/2) q[1];
cx q[0],q[1];
r(5.66985611832717,0) q[0];
r(5.66985611832717,0) q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
