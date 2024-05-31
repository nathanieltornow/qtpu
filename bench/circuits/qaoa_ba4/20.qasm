OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c7[20];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
rzz(6.042691605014292) q[0],q[1];
rzz(6.042691605014292) q[0],q[2];
rzz(6.042691605014292) q[0],q[3];
rzz(6.042691605014292) q[0],q[4];
rzz(6.042691605014292) q[0],q[5];
rzz(6.042691605014292) q[0],q[6];
rzz(6.042691605014292) q[0],q[7];
rzz(6.042691605014292) q[0],q[8];
rzz(6.042691605014292) q[0],q[9];
rzz(6.042691605014292) q[0],q[11];
rzz(6.042691605014292) q[0],q[12];
rzz(6.042691605014292) q[0],q[14];
rzz(6.042691605014292) q[0],q[15];
rzz(6.042691605014292) q[0],q[17];
rzz(6.042691605014292) q[1],q[7];
rzz(6.042691605014292) q[1],q[9];
rzz(6.042691605014292) q[1],q[16];
rzz(6.042691605014292) q[1],q[19];
rzz(6.042691605014292) q[2],q[5];
rzz(6.042691605014292) q[2],q[6];
rzz(6.042691605014292) q[2],q[13];
rzz(6.042691605014292) q[2],q[17];
rzz(6.042691605014292) q[2],q[19];
rzz(6.042691605014292) q[3],q[5];
rzz(6.042691605014292) q[3],q[6];
rzz(6.042691605014292) q[3],q[11];
rzz(6.042691605014292) q[3],q[12];
rzz(6.042691605014292) q[4],q[5];
rzz(6.042691605014292) q[4],q[18];
rzz(6.042691605014292) q[5],q[6];
rzz(6.042691605014292) q[5],q[7];
rzz(6.042691605014292) q[5],q[8];
rzz(6.042691605014292) q[5],q[10];
rzz(6.042691605014292) q[5],q[15];
rzz(6.042691605014292) q[5],q[18];
rzz(6.042691605014292) q[6],q[7];
rzz(6.042691605014292) q[6],q[8];
rzz(6.042691605014292) q[6],q[9];
rzz(6.042691605014292) q[6],q[10];
rzz(6.042691605014292) q[6],q[13];
rzz(6.042691605014292) q[6],q[15];
rzz(6.042691605014292) q[6],q[19];
rzz(6.042691605014292) q[7],q[8];
rzz(6.042691605014292) q[7],q[18];
rzz(6.042691605014292) q[7],q[19];
rzz(6.042691605014292) q[8],q[9];
rzz(6.042691605014292) q[8],q[10];
rzz(6.042691605014292) q[8],q[11];
rzz(6.042691605014292) q[8],q[12];
rzz(6.042691605014292) q[8],q[14];
rzz(6.042691605014292) q[8],q[16];
rzz(6.042691605014292) q[9],q[10];
rzz(6.042691605014292) q[9],q[11];
rzz(6.042691605014292) q[9],q[14];
rzz(6.042691605014292) q[9],q[15];
rzz(6.042691605014292) q[9],q[16];
rzz(6.042691605014292) q[10],q[12];
rzz(6.042691605014292) q[10],q[18];
rzz(6.042691605014292) q[11],q[13];
rzz(6.042691605014292) q[11],q[14];
rzz(6.042691605014292) q[11],q[16];
rzz(6.042691605014292) q[11],q[17];
rzz(6.042691605014292) q[12],q[13];
rzz(6.042691605014292) q[14],q[17];
rx(1.1692757762334427) q[0];
rx(1.1692757762334427) q[1];
rx(1.1692757762334427) q[2];
rx(1.1692757762334427) q[3];
rx(1.1692757762334427) q[4];
rx(1.1692757762334427) q[5];
rx(1.1692757762334427) q[6];
rx(1.1692757762334427) q[7];
rx(1.1692757762334427) q[8];
rx(1.1692757762334427) q[9];
rx(1.1692757762334427) q[10];
rx(1.1692757762334427) q[11];
rx(1.1692757762334427) q[12];
rx(1.1692757762334427) q[13];
rx(1.1692757762334427) q[14];
rx(1.1692757762334427) q[15];
rx(1.1692757762334427) q[16];
rx(1.1692757762334427) q[17];
rx(1.1692757762334427) q[18];
rx(1.1692757762334427) q[19];
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
measure q[18] -> c7[18];
measure q[19] -> c7[19];
