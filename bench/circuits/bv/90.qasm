OPENQASM 2.0;
include "qelib1.inc";
qreg q19[90];
creg c19[90];
h q19[0];
h q19[1];
h q19[2];
h q19[3];
h q19[4];
h q19[5];
h q19[6];
h q19[7];
h q19[8];
h q19[9];
h q19[10];
h q19[11];
h q19[12];
h q19[13];
h q19[14];
h q19[15];
h q19[16];
h q19[17];
h q19[18];
h q19[19];
h q19[20];
h q19[21];
h q19[22];
h q19[23];
h q19[24];
h q19[25];
h q19[26];
h q19[27];
h q19[28];
h q19[29];
h q19[30];
h q19[31];
h q19[32];
h q19[33];
h q19[34];
h q19[35];
h q19[36];
h q19[37];
h q19[38];
h q19[39];
h q19[40];
h q19[41];
h q19[42];
h q19[43];
h q19[44];
h q19[45];
h q19[46];
h q19[47];
h q19[48];
h q19[49];
h q19[50];
h q19[51];
h q19[52];
h q19[53];
h q19[54];
h q19[55];
h q19[56];
h q19[57];
h q19[58];
h q19[59];
h q19[60];
h q19[61];
h q19[62];
h q19[63];
h q19[64];
h q19[65];
h q19[66];
h q19[67];
h q19[68];
h q19[69];
h q19[70];
h q19[71];
h q19[72];
h q19[73];
h q19[74];
h q19[75];
h q19[76];
h q19[77];
h q19[78];
h q19[79];
h q19[80];
h q19[81];
h q19[82];
h q19[83];
h q19[84];
h q19[85];
h q19[86];
h q19[87];
h q19[88];
x q19[89];
h q19[89];
barrier q19[0],q19[1],q19[2],q19[3],q19[4],q19[5],q19[6],q19[7],q19[8],q19[9],q19[10],q19[11],q19[12],q19[13],q19[14],q19[15],q19[16],q19[17],q19[18],q19[19],q19[20],q19[21],q19[22],q19[23],q19[24],q19[25],q19[26],q19[27],q19[28],q19[29],q19[30],q19[31],q19[32],q19[33],q19[34],q19[35],q19[36],q19[37],q19[38],q19[39],q19[40],q19[41],q19[42],q19[43],q19[44],q19[45],q19[46],q19[47],q19[48],q19[49],q19[50],q19[51],q19[52],q19[53],q19[54],q19[55],q19[56],q19[57],q19[58],q19[59],q19[60],q19[61],q19[62],q19[63],q19[64],q19[65],q19[66],q19[67],q19[68],q19[69],q19[70],q19[71],q19[72],q19[73],q19[74],q19[75],q19[76],q19[77],q19[78],q19[79],q19[80],q19[81],q19[82],q19[83],q19[84],q19[85],q19[86],q19[87],q19[88],q19[89];
cx q19[0],q19[89];
cx q19[1],q19[89];
cx q19[3],q19[89];
cx q19[4],q19[89];
cx q19[6],q19[89];
cx q19[11],q19[89];
cx q19[12],q19[89];
cx q19[14],q19[89];
cx q19[22],q19[89];
cx q19[24],q19[89];
cx q19[25],q19[89];
cx q19[27],q19[89];
cx q19[30],q19[89];
cx q19[34],q19[89];
cx q19[35],q19[89];
cx q19[36],q19[89];
cx q19[38],q19[89];
cx q19[45],q19[89];
cx q19[46],q19[89];
cx q19[47],q19[89];
cx q19[49],q19[89];
cx q19[50],q19[89];
cx q19[51],q19[89];
cx q19[54],q19[89];
cx q19[56],q19[89];
cx q19[58],q19[89];
cx q19[61],q19[89];
cx q19[62],q19[89];
cx q19[63],q19[89];
cx q19[64],q19[89];
cx q19[65],q19[89];
cx q19[68],q19[89];
cx q19[69],q19[89];
cx q19[71],q19[89];
cx q19[73],q19[89];
cx q19[74],q19[89];
cx q19[75],q19[89];
cx q19[80],q19[89];
cx q19[81],q19[89];
cx q19[82],q19[89];
cx q19[83],q19[89];
cx q19[85],q19[89];
cx q19[87],q19[89];
barrier q19[0],q19[1],q19[2],q19[3],q19[4],q19[5],q19[6],q19[7],q19[8],q19[9],q19[10],q19[11],q19[12],q19[13],q19[14],q19[15],q19[16],q19[17],q19[18],q19[19],q19[20],q19[21],q19[22],q19[23],q19[24],q19[25],q19[26],q19[27],q19[28],q19[29],q19[30],q19[31],q19[32],q19[33],q19[34],q19[35],q19[36],q19[37],q19[38],q19[39],q19[40],q19[41],q19[42],q19[43],q19[44],q19[45],q19[46],q19[47],q19[48],q19[49],q19[50],q19[51],q19[52],q19[53],q19[54],q19[55],q19[56],q19[57],q19[58],q19[59],q19[60],q19[61],q19[62],q19[63],q19[64],q19[65],q19[66],q19[67],q19[68],q19[69],q19[70],q19[71],q19[72],q19[73],q19[74],q19[75],q19[76],q19[77],q19[78],q19[79],q19[80],q19[81],q19[82],q19[83],q19[84],q19[85],q19[86],q19[87],q19[88],q19[89];
h q19[0];
h q19[1];
h q19[2];
h q19[3];
h q19[4];
h q19[5];
h q19[6];
h q19[7];
h q19[8];
h q19[9];
h q19[10];
h q19[11];
h q19[12];
h q19[13];
h q19[14];
h q19[15];
h q19[16];
h q19[17];
h q19[18];
h q19[19];
h q19[20];
h q19[21];
h q19[22];
h q19[23];
h q19[24];
h q19[25];
h q19[26];
h q19[27];
h q19[28];
h q19[29];
h q19[30];
h q19[31];
h q19[32];
h q19[33];
h q19[34];
h q19[35];
h q19[36];
h q19[37];
h q19[38];
h q19[39];
h q19[40];
h q19[41];
h q19[42];
h q19[43];
h q19[44];
h q19[45];
h q19[46];
h q19[47];
h q19[48];
h q19[49];
h q19[50];
h q19[51];
h q19[52];
h q19[53];
h q19[54];
h q19[55];
h q19[56];
h q19[57];
h q19[58];
h q19[59];
h q19[60];
h q19[61];
h q19[62];
h q19[63];
h q19[64];
h q19[65];
h q19[66];
h q19[67];
h q19[68];
h q19[69];
h q19[70];
h q19[71];
h q19[72];
h q19[73];
h q19[74];
h q19[75];
h q19[76];
h q19[77];
h q19[78];
h q19[79];
h q19[80];
h q19[81];
h q19[82];
h q19[83];
h q19[84];
h q19[85];
h q19[86];
h q19[87];
h q19[88];
measure q19[0] -> c19[0];
measure q19[1] -> c19[1];
measure q19[2] -> c19[2];
measure q19[3] -> c19[3];
measure q19[4] -> c19[4];
measure q19[5] -> c19[5];
measure q19[6] -> c19[6];
measure q19[7] -> c19[7];
measure q19[8] -> c19[8];
measure q19[9] -> c19[9];
measure q19[10] -> c19[10];
measure q19[11] -> c19[11];
measure q19[12] -> c19[12];
measure q19[13] -> c19[13];
measure q19[14] -> c19[14];
measure q19[15] -> c19[15];
measure q19[16] -> c19[16];
measure q19[17] -> c19[17];
measure q19[18] -> c19[18];
measure q19[19] -> c19[19];
measure q19[20] -> c19[20];
measure q19[21] -> c19[21];
measure q19[22] -> c19[22];
measure q19[23] -> c19[23];
measure q19[24] -> c19[24];
measure q19[25] -> c19[25];
measure q19[26] -> c19[26];
measure q19[27] -> c19[27];
measure q19[28] -> c19[28];
measure q19[29] -> c19[29];
measure q19[30] -> c19[30];
measure q19[31] -> c19[31];
measure q19[32] -> c19[32];
measure q19[33] -> c19[33];
measure q19[34] -> c19[34];
measure q19[35] -> c19[35];
measure q19[36] -> c19[36];
measure q19[37] -> c19[37];
measure q19[38] -> c19[38];
measure q19[39] -> c19[39];
measure q19[40] -> c19[40];
measure q19[41] -> c19[41];
measure q19[42] -> c19[42];
measure q19[43] -> c19[43];
measure q19[44] -> c19[44];
measure q19[45] -> c19[45];
measure q19[46] -> c19[46];
measure q19[47] -> c19[47];
measure q19[48] -> c19[48];
measure q19[49] -> c19[49];
measure q19[50] -> c19[50];
measure q19[51] -> c19[51];
measure q19[52] -> c19[52];
measure q19[53] -> c19[53];
measure q19[54] -> c19[54];
measure q19[55] -> c19[55];
measure q19[56] -> c19[56];
measure q19[57] -> c19[57];
measure q19[58] -> c19[58];
measure q19[59] -> c19[59];
measure q19[60] -> c19[60];
measure q19[61] -> c19[61];
measure q19[62] -> c19[62];
measure q19[63] -> c19[63];
measure q19[64] -> c19[64];
measure q19[65] -> c19[65];
measure q19[66] -> c19[66];
measure q19[67] -> c19[67];
measure q19[68] -> c19[68];
measure q19[69] -> c19[69];
measure q19[70] -> c19[70];
measure q19[71] -> c19[71];
measure q19[72] -> c19[72];
measure q19[73] -> c19[73];
measure q19[74] -> c19[74];
measure q19[75] -> c19[75];
measure q19[76] -> c19[76];
measure q19[77] -> c19[77];
measure q19[78] -> c19[78];
measure q19[79] -> c19[79];
measure q19[80] -> c19[80];
measure q19[81] -> c19[81];
measure q19[82] -> c19[82];
measure q19[83] -> c19[83];
measure q19[84] -> c19[84];
measure q19[85] -> c19[85];
measure q19[86] -> c19[86];
measure q19[87] -> c19[87];
measure q19[88] -> c19[88];
