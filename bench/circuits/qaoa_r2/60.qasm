OPENQASM 2.0;
include "qelib1.inc";
qreg q[60];
creg c15[60];
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
h q[50];
h q[51];
h q[52];
h q[53];
h q[54];
h q[55];
h q[56];
h q[57];
h q[58];
h q[59];
rzz(4.574278633304488) q[1],q[36];
rzz(4.574278633304488) q[3],q[0];
rzz(4.574278633304488) q[3],q[31];
rzz(4.574278633304488) q[4],q[27];
rzz(4.574278633304488) q[4],q[52];
rzz(4.574278633304488) q[5],q[17];
rzz(4.574278633304488) q[6],q[53];
rzz(4.574278633304488) q[7],q[32];
rzz(4.574278633304488) q[7],q[38];
rzz(4.574278633304488) q[8],q[2];
rzz(4.574278633304488) q[8],q[43];
rzz(4.574278633304488) q[9],q[48];
rzz(4.574278633304488) q[9],q[56];
rzz(4.574278633304488) q[12],q[11];
rzz(4.574278633304488) q[12],q[25];
rzz(4.574278633304488) q[13],q[5];
rzz(4.574278633304488) q[13],q[33];
rzz(4.574278633304488) q[14],q[6];
rzz(4.574278633304488) q[14],q[34];
rzz(4.574278633304488) q[15],q[11];
rzz(4.574278633304488) q[16],q[58];
rzz(4.574278633304488) q[18],q[23];
rzz(4.574278633304488) q[18],q[24];
rzz(4.574278633304488) q[19],q[29];
rzz(4.574278633304488) q[21],q[1];
rzz(4.574278633304488) q[22],q[36];
rzz(4.574278633304488) q[22],q[47];
rzz(4.574278633304488) q[23],q[49];
rzz(4.574278633304488) q[24],q[39];
rzz(4.574278633304488) q[25],q[16];
rzz(4.574278633304488) q[26],q[17];
rzz(4.574278633304488) q[26],q[51];
rzz(4.574278633304488) q[27],q[57];
rzz(4.574278633304488) q[28],q[20];
rzz(4.574278633304488) q[28],q[46];
rzz(4.574278633304488) q[30],q[29];
rzz(4.574278633304488) q[30],q[42];
rzz(4.574278633304488) q[31],q[56];
rzz(4.574278633304488) q[33],q[0];
rzz(4.574278633304488) q[34],q[53];
rzz(4.574278633304488) q[35],q[54];
rzz(4.574278633304488) q[37],q[55];
rzz(4.574278633304488) q[37],q[59];
rzz(4.574278633304488) q[38],q[41];
rzz(4.574278633304488) q[39],q[20];
rzz(4.574278633304488) q[40],q[50];
rzz(4.574278633304488) q[41],q[42];
rzz(4.574278633304488) q[43],q[48];
rzz(4.574278633304488) q[44],q[32];
rzz(4.574278633304488) q[44],q[45];
rzz(4.574278633304488) q[45],q[35];
rzz(4.574278633304488) q[46],q[21];
rzz(4.574278633304488) q[47],q[10];
rzz(4.574278633304488) q[49],q[52];
rzz(4.574278633304488) q[50],q[15];
rzz(4.574278633304488) q[51],q[40];
rzz(4.574278633304488) q[54],q[19];
rzz(4.574278633304488) q[55],q[2];
rzz(4.574278633304488) q[58],q[57];
rzz(4.574278633304488) q[59],q[10];
rx(1.2763942087541607) q[0];
rx(1.2763942087541607) q[1];
rx(1.2763942087541607) q[2];
rx(1.2763942087541607) q[3];
rx(1.2763942087541607) q[4];
rx(1.2763942087541607) q[5];
rx(1.2763942087541607) q[6];
rx(1.2763942087541607) q[7];
rx(1.2763942087541607) q[8];
rx(1.2763942087541607) q[9];
rx(1.2763942087541607) q[10];
rx(1.2763942087541607) q[11];
rx(1.2763942087541607) q[12];
rx(1.2763942087541607) q[13];
rx(1.2763942087541607) q[14];
rx(1.2763942087541607) q[15];
rx(1.2763942087541607) q[16];
rx(1.2763942087541607) q[17];
rx(1.2763942087541607) q[18];
rx(1.2763942087541607) q[19];
rx(1.2763942087541607) q[20];
rx(1.2763942087541607) q[21];
rx(1.2763942087541607) q[22];
rx(1.2763942087541607) q[23];
rx(1.2763942087541607) q[24];
rx(1.2763942087541607) q[25];
rx(1.2763942087541607) q[26];
rx(1.2763942087541607) q[27];
rx(1.2763942087541607) q[28];
rx(1.2763942087541607) q[29];
rx(1.2763942087541607) q[30];
rx(1.2763942087541607) q[31];
rx(1.2763942087541607) q[32];
rx(1.2763942087541607) q[33];
rx(1.2763942087541607) q[34];
rx(1.2763942087541607) q[35];
rx(1.2763942087541607) q[36];
rx(1.2763942087541607) q[37];
rx(1.2763942087541607) q[38];
rx(1.2763942087541607) q[39];
rx(1.2763942087541607) q[40];
rx(1.2763942087541607) q[41];
rx(1.2763942087541607) q[42];
rx(1.2763942087541607) q[43];
rx(1.2763942087541607) q[44];
rx(1.2763942087541607) q[45];
rx(1.2763942087541607) q[46];
rx(1.2763942087541607) q[47];
rx(1.2763942087541607) q[48];
rx(1.2763942087541607) q[49];
rx(1.2763942087541607) q[50];
rx(1.2763942087541607) q[51];
rx(1.2763942087541607) q[52];
rx(1.2763942087541607) q[53];
rx(1.2763942087541607) q[54];
rx(1.2763942087541607) q[55];
rx(1.2763942087541607) q[56];
rx(1.2763942087541607) q[57];
rx(1.2763942087541607) q[58];
rx(1.2763942087541607) q[59];
measure q[0] -> c15[0];
measure q[1] -> c15[1];
measure q[2] -> c15[2];
measure q[3] -> c15[3];
measure q[4] -> c15[4];
measure q[5] -> c15[5];
measure q[6] -> c15[6];
measure q[7] -> c15[7];
measure q[8] -> c15[8];
measure q[9] -> c15[9];
measure q[10] -> c15[10];
measure q[11] -> c15[11];
measure q[12] -> c15[12];
measure q[13] -> c15[13];
measure q[14] -> c15[14];
measure q[15] -> c15[15];
measure q[16] -> c15[16];
measure q[17] -> c15[17];
measure q[18] -> c15[18];
measure q[19] -> c15[19];
measure q[20] -> c15[20];
measure q[21] -> c15[21];
measure q[22] -> c15[22];
measure q[23] -> c15[23];
measure q[24] -> c15[24];
measure q[25] -> c15[25];
measure q[26] -> c15[26];
measure q[27] -> c15[27];
measure q[28] -> c15[28];
measure q[29] -> c15[29];
measure q[30] -> c15[30];
measure q[31] -> c15[31];
measure q[32] -> c15[32];
measure q[33] -> c15[33];
measure q[34] -> c15[34];
measure q[35] -> c15[35];
measure q[36] -> c15[36];
measure q[37] -> c15[37];
measure q[38] -> c15[38];
measure q[39] -> c15[39];
measure q[40] -> c15[40];
measure q[41] -> c15[41];
measure q[42] -> c15[42];
measure q[43] -> c15[43];
measure q[44] -> c15[44];
measure q[45] -> c15[45];
measure q[46] -> c15[46];
measure q[47] -> c15[47];
measure q[48] -> c15[48];
measure q[49] -> c15[49];
measure q[50] -> c15[50];
measure q[51] -> c15[51];
measure q[52] -> c15[52];
measure q[53] -> c15[53];
measure q[54] -> c15[54];
measure q[55] -> c15[55];
measure q[56] -> c15[56];
measure q[57] -> c15[57];
measure q[58] -> c15[58];
measure q[59] -> c15[59];
