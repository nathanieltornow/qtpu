OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c7[18];
h q[0];
rz(-3*pi/4) q[0];
h q[0];
h q[1];
rz(-3*pi/4) q[1];
h q[1];
h q[2];
rz(-3*pi/4) q[2];
h q[2];
h q[3];
rz(-3*pi/4) q[3];
h q[3];
h q[4];
rz(-3*pi/4) q[4];
h q[4];
h q[5];
rz(-3*pi/4) q[5];
h q[5];
h q[6];
rz(-3*pi/4) q[6];
h q[6];
h q[7];
rz(-3*pi/4) q[7];
h q[7];
h q[8];
rz(-3*pi/4) q[8];
h q[8];
h q[9];
rz(-3*pi/4) q[9];
h q[9];
h q[10];
rz(-3*pi/4) q[10];
h q[10];
h q[11];
rz(-3*pi/4) q[11];
h q[11];
h q[12];
rz(-3*pi/4) q[12];
h q[12];
h q[13];
rz(-3*pi/4) q[13];
h q[13];
h q[14];
rz(-3*pi/4) q[14];
h q[14];
h q[15];
rz(-3*pi/4) q[15];
h q[15];
h q[16];
rz(-3*pi/4) q[16];
h q[16];
h q[17];
rz(-3*pi/4) q[17];
h q[17];
rzz(-pi/2) q[0],q[1];
rzz(-pi/2) q[1],q[2];
rzz(-pi/2) q[2],q[3];
rzz(-pi/2) q[3],q[4];
rzz(-pi/2) q[4],q[5];
rzz(-pi/2) q[5],q[6];
rzz(-pi/2) q[6],q[7];
rzz(-pi/2) q[7],q[8];
rzz(-pi/2) q[8],q[9];
rzz(-pi/2) q[9],q[10];
rzz(-pi/2) q[10],q[11];
rzz(-pi/2) q[11],q[12];
rzz(-pi/2) q[12],q[13];
rzz(-pi/2) q[13],q[14];
rzz(-pi/2) q[14],q[15];
rzz(-pi/2) q[15],q[16];
rzz(-pi/2) q[16],q[17];
measure q[0] -> c7[0];
measure q[1] -> c7[1];
measure q[2] -> c7[2];
measure q[3] -> c7[3];
measure q[4] -> c7[4];
measure q[5] -> c7[5];
measure q[6] -> c7[6];
measure q[7] -> c7[7];
measure q[8] -> c7[8];
measure q[9] -> c7[9];
measure q[10] -> c7[10];
measure q[11] -> c7[11];
measure q[12] -> c7[12];
measure q[13] -> c7[13];
measure q[14] -> c7[14];
measure q[15] -> c7[15];
measure q[16] -> c7[16];
measure q[17] -> c7[17];