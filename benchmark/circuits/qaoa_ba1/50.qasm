OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
creg c14[50];
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
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
h q[30];
h q[31];
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
h q[39];
h q[40];
h q[41];
h q[42];
h q[43];
h q[44];
h q[45];
h q[46];
h q[47];
h q[48];
h q[49];
rzz(2.0477782182850186) q[0],q[1];
rzz(2.0477782182850186) q[0],q[12];
rzz(2.0477782182850186) q[0],q[18];
rzz(2.0477782182850186) q[0],q[24];
rzz(2.0477782182850186) q[0],q[25];
rzz(2.0477782182850186) q[0],q[32];
rzz(2.0477782182850186) q[0],q[33];
rzz(2.0477782182850186) q[0],q[40];
rzz(2.0477782182850186) q[1],q[2];
rzz(2.0477782182850186) q[1],q[3];
rzz(2.0477782182850186) q[1],q[4];
rzz(2.0477782182850186) q[1],q[6];
rzz(2.0477782182850186) q[1],q[21];
rzz(2.0477782182850186) q[2],q[5];
rzz(2.0477782182850186) q[2],q[7];
rzz(2.0477782182850186) q[2],q[17];
rzz(2.0477782182850186) q[2],q[22];
rzz(2.0477782182850186) q[2],q[27];
rzz(2.0477782182850186) q[2],q[41];
rzz(2.0477782182850186) q[3],q[20];
rzz(2.0477782182850186) q[4],q[9];
rzz(2.0477782182850186) q[5],q[14];
rzz(2.0477782182850186) q[5],q[30];
rzz(2.0477782182850186) q[5],q[38];
rzz(2.0477782182850186) q[6],q[10];
rzz(2.0477782182850186) q[6],q[11];
rzz(2.0477782182850186) q[6],q[26];
rzz(2.0477782182850186) q[7],q[8];
rzz(2.0477782182850186) q[7],q[13];
rzz(2.0477782182850186) q[8],q[16];
rzz(2.0477782182850186) q[8],q[43];
rzz(2.0477782182850186) q[9],q[37];
rzz(2.0477782182850186) q[10],q[15];
rzz(2.0477782182850186) q[16],q[23];
rzz(2.0477782182850186) q[18],q[19];
rzz(2.0477782182850186) q[18],q[28];
rzz(2.0477782182850186) q[18],q[31];
rzz(2.0477782182850186) q[19],q[35];
rzz(2.0477782182850186) q[23],q[29];
rzz(2.0477782182850186) q[29],q[42];
rzz(2.0477782182850186) q[29],q[48];
rzz(2.0477782182850186) q[30],q[39];
rzz(2.0477782182850186) q[33],q[34];
rzz(2.0477782182850186) q[34],q[36];
rzz(2.0477782182850186) q[34],q[47];
rzz(2.0477782182850186) q[37],q[45];
rzz(2.0477782182850186) q[37],q[46];
rzz(2.0477782182850186) q[37],q[49];
rzz(2.0477782182850186) q[43],q[44];
rx(1.500883274850303) q[0];
rx(1.500883274850303) q[1];
rx(1.500883274850303) q[2];
rx(1.500883274850303) q[3];
rx(1.500883274850303) q[4];
rx(1.500883274850303) q[5];
rx(1.500883274850303) q[6];
rx(1.500883274850303) q[7];
rx(1.500883274850303) q[8];
rx(1.500883274850303) q[9];
rx(1.500883274850303) q[10];
rx(1.500883274850303) q[11];
rx(1.500883274850303) q[12];
rx(1.500883274850303) q[13];
rx(1.500883274850303) q[14];
rx(1.500883274850303) q[15];
rx(1.500883274850303) q[16];
rx(1.500883274850303) q[17];
rx(1.500883274850303) q[18];
rx(1.500883274850303) q[19];
rx(1.500883274850303) q[20];
rx(1.500883274850303) q[21];
rx(1.500883274850303) q[22];
rx(1.500883274850303) q[23];
rx(1.500883274850303) q[24];
rx(1.500883274850303) q[25];
rx(1.500883274850303) q[26];
rx(1.500883274850303) q[27];
rx(1.500883274850303) q[28];
rx(1.500883274850303) q[29];
rx(1.500883274850303) q[30];
rx(1.500883274850303) q[31];
rx(1.500883274850303) q[32];
rx(1.500883274850303) q[33];
rx(1.500883274850303) q[34];
rx(1.500883274850303) q[35];
rx(1.500883274850303) q[36];
rx(1.500883274850303) q[37];
rx(1.500883274850303) q[38];
rx(1.500883274850303) q[39];
rx(1.500883274850303) q[40];
rx(1.500883274850303) q[41];
rx(1.500883274850303) q[42];
rx(1.500883274850303) q[43];
rx(1.500883274850303) q[44];
rx(1.500883274850303) q[45];
rx(1.500883274850303) q[46];
rx(1.500883274850303) q[47];
rx(1.500883274850303) q[48];
rx(1.500883274850303) q[49];
measure q[0] -> c14[0];
measure q[1] -> c14[1];
measure q[2] -> c14[2];
measure q[3] -> c14[3];
measure q[4] -> c14[4];
measure q[5] -> c14[5];
measure q[6] -> c14[6];
measure q[7] -> c14[7];
measure q[8] -> c14[8];
measure q[9] -> c14[9];
measure q[10] -> c14[10];
measure q[11] -> c14[11];
measure q[12] -> c14[12];
measure q[13] -> c14[13];
measure q[14] -> c14[14];
measure q[15] -> c14[15];
measure q[16] -> c14[16];
measure q[17] -> c14[17];
measure q[18] -> c14[18];
measure q[19] -> c14[19];
measure q[20] -> c14[20];
measure q[21] -> c14[21];
measure q[22] -> c14[22];
measure q[23] -> c14[23];
measure q[24] -> c14[24];
measure q[25] -> c14[25];
measure q[26] -> c14[26];
measure q[27] -> c14[27];
measure q[28] -> c14[28];
measure q[29] -> c14[29];
measure q[30] -> c14[30];
measure q[31] -> c14[31];
measure q[32] -> c14[32];
measure q[33] -> c14[33];
measure q[34] -> c14[34];
measure q[35] -> c14[35];
measure q[36] -> c14[36];
measure q[37] -> c14[37];
measure q[38] -> c14[38];
measure q[39] -> c14[39];
measure q[40] -> c14[40];
measure q[41] -> c14[41];
measure q[42] -> c14[42];
measure q[43] -> c14[43];
measure q[44] -> c14[44];
measure q[45] -> c14[45];
measure q[46] -> c14[46];
measure q[47] -> c14[47];
measure q[48] -> c14[48];
measure q[49] -> c14[49];