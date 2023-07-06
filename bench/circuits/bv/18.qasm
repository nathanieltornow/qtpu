OPENQASM 2.0;
include "qelib1.inc";
qreg q7[18];
creg c7[18];
h q7[0];
h q7[1];
h q7[2];
h q7[3];
h q7[4];
h q7[5];
h q7[6];
h q7[7];
h q7[8];
h q7[9];
h q7[10];
h q7[11];
h q7[12];
h q7[13];
h q7[14];
h q7[15];
h q7[16];
x q7[17];
h q7[17];
barrier q7[0],q7[1],q7[2],q7[3],q7[4],q7[5],q7[6],q7[7],q7[8],q7[9],q7[10],q7[11],q7[12],q7[13],q7[14],q7[15],q7[16],q7[17];
cx q7[0],q7[17];
cx q7[4],q7[17];
cx q7[7],q7[17];
cx q7[8],q7[17];
cx q7[10],q7[17];
cx q7[11],q7[17];
cx q7[15],q7[17];
cx q7[16],q7[17];
barrier q7[0],q7[1],q7[2],q7[3],q7[4],q7[5],q7[6],q7[7],q7[8],q7[9],q7[10],q7[11],q7[12],q7[13],q7[14],q7[15],q7[16],q7[17];
h q7[0];
h q7[1];
h q7[2];
h q7[3];
h q7[4];
h q7[5];
h q7[6];
h q7[7];
h q7[8];
h q7[9];
h q7[10];
h q7[11];
h q7[12];
h q7[13];
h q7[14];
h q7[15];
h q7[16];
measure q7[0] -> c7[0];
measure q7[1] -> c7[1];
measure q7[2] -> c7[2];
measure q7[3] -> c7[3];
measure q7[4] -> c7[4];
measure q7[5] -> c7[5];
measure q7[6] -> c7[6];
measure q7[7] -> c7[7];
measure q7[8] -> c7[8];
measure q7[9] -> c7[9];
measure q7[10] -> c7[10];
measure q7[11] -> c7[11];
measure q7[12] -> c7[12];
measure q7[13] -> c7[13];
measure q7[14] -> c7[14];
measure q7[15] -> c7[15];
measure q7[16] -> c7[16];
