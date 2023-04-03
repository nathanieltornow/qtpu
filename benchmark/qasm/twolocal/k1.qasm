OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(1.9507176521076317,-pi/2,pi/2) q0; }
qreg q[1];
creg meas[1];
u2(0,pi) q[0];
r(1.9507176521076317,0) q[0];
barrier q[0];
measure q[0] -> meas[0];
