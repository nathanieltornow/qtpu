OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c11[28];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[8];
cx q[8],q[9];
cx q[9],q[10];
cx q[10],q[11];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[15],q[16];
cx q[16],q[17];
cx q[17],q[18];
cx q[18],q[19];
cx q[19],q[20];
cx q[20],q[21];
cx q[21],q[22];
cx q[22],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[26],q[27];
measure q[0] -> c11[0];
measure q[1] -> c11[1];
measure q[2] -> c11[2];
measure q[3] -> c11[3];
measure q[4] -> c11[4];
measure q[5] -> c11[5];
measure q[6] -> c11[6];
measure q[7] -> c11[7];
measure q[8] -> c11[8];
measure q[9] -> c11[9];
measure q[10] -> c11[10];
measure q[11] -> c11[11];
measure q[12] -> c11[12];
measure q[13] -> c11[13];
measure q[14] -> c11[14];
measure q[15] -> c11[15];
measure q[16] -> c11[16];
measure q[17] -> c11[17];
measure q[18] -> c11[18];
measure q[19] -> c11[19];
measure q[20] -> c11[20];
measure q[21] -> c11[21];
measure q[22] -> c11[22];
measure q[23] -> c11[23];
measure q[24] -> c11[24];
measure q[25] -> c11[25];
measure q[26] -> c11[26];
measure q[27] -> c11[27];