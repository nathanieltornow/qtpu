OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
u3(pi,0,pi) q[0];
u2(0,pi) q[0];
barrier q[0];
