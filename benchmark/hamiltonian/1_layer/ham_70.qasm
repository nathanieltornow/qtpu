// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7), q(8), q(9), q(10), q(11), q(12), q(13), q(14), q(15), q(16), q(17), q(18), q(19), q(20), q(21), q(22), q(23), q(24), q(25), q(26), q(27), q(28), q(29), q(30), q(31), q(32), q(33), q(34), q(35), q(36), q(37), q(38), q(39), q(40), q(41), q(42), q(43), q(44), q(45), q(46), q(47), q(48), q(49), q(50), q(51), q(52), q(53), q(54), q(55), q(56), q(57), q(58), q(59), q(60), q(61), q(62), q(63), q(64), q(65), q(66), q(67), q(68), q(69)]
qreg q[70];
creg m0[70];  // Measurement: q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7),q(8),q(9),q(10),q(11),q(12),q(13),q(14),q(15),q(16),q(17),q(18),q(19),q(20),q(21),q(22),q(23),q(24),q(25),q(26),q(27),q(28),q(29),q(30),q(31),q(32),q(33),q(34),q(35),q(36),q(37),q(38),q(39),q(40),q(41),q(42),q(43),q(44),q(45),q(46),q(47),q(48),q(49),q(50),q(51),q(52),q(53),q(54),q(55),q(56),q(57),q(58),q(59),q(60),q(61),q(62),q(63),q(64),q(65),q(66),q(67),q(68),q(69)


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
h q[60];
h q[61];
h q[62];
h q[63];
h q[64];
h q[65];
h q[66];
h q[67];
h q[68];
h q[69];
rz(pi*-0.75) q[0];
rz(pi*-0.75) q[1];
rz(pi*-0.75) q[2];
rz(pi*-0.75) q[3];
rz(pi*-0.75) q[4];
rz(pi*-0.75) q[5];
rz(pi*-0.75) q[6];
rz(pi*-0.75) q[7];
rz(pi*-0.75) q[8];
rz(pi*-0.75) q[9];
rz(pi*-0.75) q[10];
rz(pi*-0.75) q[11];
rz(pi*-0.75) q[12];
rz(pi*-0.75) q[13];
rz(pi*-0.75) q[14];
rz(pi*-0.75) q[15];
rz(pi*-0.75) q[16];
rz(pi*-0.75) q[17];
rz(pi*-0.75) q[18];
rz(pi*-0.75) q[19];
rz(pi*-0.75) q[20];
rz(pi*-0.75) q[21];
rz(pi*-0.75) q[22];
rz(pi*-0.75) q[23];
rz(pi*-0.75) q[24];
rz(pi*-0.75) q[25];
rz(pi*-0.75) q[26];
rz(pi*-0.75) q[27];
rz(pi*-0.75) q[28];
rz(pi*-0.75) q[29];
rz(pi*-0.75) q[30];
rz(pi*-0.75) q[31];
rz(pi*-0.75) q[32];
rz(pi*-0.75) q[33];
rz(pi*-0.75) q[34];
rz(pi*-0.75) q[35];
rz(pi*-0.75) q[36];
rz(pi*-0.75) q[37];
rz(pi*-0.75) q[38];
rz(pi*-0.75) q[39];
rz(pi*-0.75) q[40];
rz(pi*-0.75) q[41];
rz(pi*-0.75) q[42];
rz(pi*-0.75) q[43];
rz(pi*-0.75) q[44];
rz(pi*-0.75) q[45];
rz(pi*-0.75) q[46];
rz(pi*-0.75) q[47];
rz(pi*-0.75) q[48];
rz(pi*-0.75) q[49];
rz(pi*-0.75) q[50];
rz(pi*-0.75) q[51];
rz(pi*-0.75) q[52];
rz(pi*-0.75) q[53];
rz(pi*-0.75) q[54];
rz(pi*-0.75) q[55];
rz(pi*-0.75) q[56];
rz(pi*-0.75) q[57];
rz(pi*-0.75) q[58];
rz(pi*-0.75) q[59];
rz(pi*-0.75) q[60];
rz(pi*-0.75) q[61];
rz(pi*-0.75) q[62];
rz(pi*-0.75) q[63];
rz(pi*-0.75) q[64];
rz(pi*-0.75) q[65];
rz(pi*-0.75) q[66];
rz(pi*-0.75) q[67];
rz(pi*-0.75) q[68];
rz(pi*-0.75) q[69];
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
h q[60];
h q[61];
h q[62];
h q[63];
h q[64];
h q[65];
h q[66];
h q[67];
h q[68];
h q[69];
cx q[0],q[1];
rz(pi*-0.5) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.5) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(pi*-0.5) q[3];
cx q[2],q[3];
cx q[3],q[4];
rz(pi*-0.5) q[4];
cx q[3],q[4];
cx q[4],q[5];
rz(pi*-0.5) q[5];
cx q[4],q[5];
cx q[5],q[6];
rz(pi*-0.5) q[6];
cx q[5],q[6];
cx q[6],q[7];
rz(pi*-0.5) q[7];
cx q[6],q[7];
cx q[7],q[8];
rz(pi*-0.5) q[8];
cx q[7],q[8];
cx q[8],q[9];
rz(pi*-0.5) q[9];
cx q[8],q[9];
cx q[9],q[10];
rz(pi*-0.5) q[10];
cx q[9],q[10];
cx q[10],q[11];
rz(pi*-0.5) q[11];
cx q[10],q[11];
cx q[11],q[12];
rz(pi*-0.5) q[12];
cx q[11],q[12];
cx q[12],q[13];
rz(pi*-0.5) q[13];
cx q[12],q[13];
cx q[13],q[14];
rz(pi*-0.5) q[14];
cx q[13],q[14];
cx q[14],q[15];
rz(pi*-0.5) q[15];
cx q[14],q[15];
cx q[15],q[16];
rz(pi*-0.5) q[16];
cx q[15],q[16];
cx q[16],q[17];
rz(pi*-0.5) q[17];
cx q[16],q[17];
cx q[17],q[18];
rz(pi*-0.5) q[18];
cx q[17],q[18];
cx q[18],q[19];
rz(pi*-0.5) q[19];
cx q[18],q[19];
cx q[19],q[20];
rz(pi*-0.5) q[20];
cx q[19],q[20];
cx q[20],q[21];
rz(pi*-0.5) q[21];
cx q[20],q[21];
cx q[21],q[22];
rz(pi*-0.5) q[22];
cx q[21],q[22];
cx q[22],q[23];
rz(pi*-0.5) q[23];
cx q[22],q[23];
cx q[23],q[24];
rz(pi*-0.5) q[24];
cx q[23],q[24];
cx q[24],q[25];
rz(pi*-0.5) q[25];
cx q[24],q[25];
cx q[25],q[26];
rz(pi*-0.5) q[26];
cx q[25],q[26];
cx q[26],q[27];
rz(pi*-0.5) q[27];
cx q[26],q[27];
cx q[27],q[28];
rz(pi*-0.5) q[28];
cx q[27],q[28];
cx q[28],q[29];
rz(pi*-0.5) q[29];
cx q[28],q[29];
cx q[29],q[30];
rz(pi*-0.5) q[30];
cx q[29],q[30];
cx q[30],q[31];
rz(pi*-0.5) q[31];
cx q[30],q[31];
cx q[31],q[32];
rz(pi*-0.5) q[32];
cx q[31],q[32];
cx q[32],q[33];
rz(pi*-0.5) q[33];
cx q[32],q[33];
cx q[33],q[34];
rz(pi*-0.5) q[34];
cx q[33],q[34];
cx q[34],q[35];
rz(pi*-0.5) q[35];
cx q[34],q[35];
cx q[35],q[36];
rz(pi*-0.5) q[36];
cx q[35],q[36];
cx q[36],q[37];
rz(pi*-0.5) q[37];
cx q[36],q[37];
cx q[37],q[38];
rz(pi*-0.5) q[38];
cx q[37],q[38];
cx q[38],q[39];
rz(pi*-0.5) q[39];
cx q[38],q[39];
cx q[39],q[40];
rz(pi*-0.5) q[40];
cx q[39],q[40];
cx q[40],q[41];
rz(pi*-0.5) q[41];
cx q[40],q[41];
cx q[41],q[42];
rz(pi*-0.5) q[42];
cx q[41],q[42];
cx q[42],q[43];
rz(pi*-0.5) q[43];
cx q[42],q[43];
cx q[43],q[44];
rz(pi*-0.5) q[44];
cx q[43],q[44];
cx q[44],q[45];
rz(pi*-0.5) q[45];
cx q[44],q[45];
cx q[45],q[46];
rz(pi*-0.5) q[46];
cx q[45],q[46];
cx q[46],q[47];
rz(pi*-0.5) q[47];
cx q[46],q[47];
cx q[47],q[48];
rz(pi*-0.5) q[48];
cx q[47],q[48];
cx q[48],q[49];
rz(pi*-0.5) q[49];
cx q[48],q[49];
cx q[49],q[50];
rz(pi*-0.5) q[50];
cx q[49],q[50];
cx q[50],q[51];
rz(pi*-0.5) q[51];
cx q[50],q[51];
cx q[51],q[52];
rz(pi*-0.5) q[52];
cx q[51],q[52];
cx q[52],q[53];
rz(pi*-0.5) q[53];
cx q[52],q[53];
cx q[53],q[54];
rz(pi*-0.5) q[54];
cx q[53],q[54];
cx q[54],q[55];
rz(pi*-0.5) q[55];
cx q[54],q[55];
cx q[55],q[56];
rz(pi*-0.5) q[56];
cx q[55],q[56];
cx q[56],q[57];
rz(pi*-0.5) q[57];
cx q[56],q[57];
cx q[57],q[58];
rz(pi*-0.5) q[58];
cx q[57],q[58];
cx q[58],q[59];
rz(pi*-0.5) q[59];
cx q[58],q[59];
cx q[59],q[60];
rz(pi*-0.5) q[60];
cx q[59],q[60];
cx q[60],q[61];
rz(pi*-0.5) q[61];
cx q[60],q[61];
cx q[61],q[62];
rz(pi*-0.5) q[62];
cx q[61],q[62];
cx q[62],q[63];
rz(pi*-0.5) q[63];
cx q[62],q[63];
cx q[63],q[64];
rz(pi*-0.5) q[64];
cx q[63],q[64];
cx q[64],q[65];
rz(pi*-0.5) q[65];
cx q[64],q[65];
cx q[65],q[66];
rz(pi*-0.5) q[66];
cx q[65],q[66];
cx q[66],q[67];
rz(pi*-0.5) q[67];
cx q[66],q[67];
cx q[67],q[68];
rz(pi*-0.5) q[68];
cx q[67],q[68];
cx q[68],q[69];
rz(pi*-0.5) q[69];
cx q[68],q[69];

// Gate: cirq.MeasurementGate(70, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7),q(8),q(9),q(10),q(11),q(12),q(13),q(14),q(15),q(16),q(17),q(18),q(19),q(20),q(21),q(22),q(23),q(24),q(25),q(26),q(27),q(28),q(29),q(30),q(31),q(32),q(33),q(34),q(35),q(36),q(37),q(38),q(39),q(40),q(41),q(42),q(43),q(44),q(45),q(46),q(47),q(48),q(49),q(50),q(51),q(52),q(53),q(54),q(55),q(56),q(57),q(58),q(59),q(60),q(61),q(62),q(63),q(64),q(65),q(66),q(67),q(68),q(69)'), ())
measure q[0] -> m0[0];
measure q[1] -> m0[1];
measure q[2] -> m0[2];
measure q[3] -> m0[3];
measure q[4] -> m0[4];
measure q[5] -> m0[5];
measure q[6] -> m0[6];
measure q[7] -> m0[7];
measure q[8] -> m0[8];
measure q[9] -> m0[9];
measure q[10] -> m0[10];
measure q[11] -> m0[11];
measure q[12] -> m0[12];
measure q[13] -> m0[13];
measure q[14] -> m0[14];
measure q[15] -> m0[15];
measure q[16] -> m0[16];
measure q[17] -> m0[17];
measure q[18] -> m0[18];
measure q[19] -> m0[19];
measure q[20] -> m0[20];
measure q[21] -> m0[21];
measure q[22] -> m0[22];
measure q[23] -> m0[23];
measure q[24] -> m0[24];
measure q[25] -> m0[25];
measure q[26] -> m0[26];
measure q[27] -> m0[27];
measure q[28] -> m0[28];
measure q[29] -> m0[29];
measure q[30] -> m0[30];
measure q[31] -> m0[31];
measure q[32] -> m0[32];
measure q[33] -> m0[33];
measure q[34] -> m0[34];
measure q[35] -> m0[35];
measure q[36] -> m0[36];
measure q[37] -> m0[37];
measure q[38] -> m0[38];
measure q[39] -> m0[39];
measure q[40] -> m0[40];
measure q[41] -> m0[41];
measure q[42] -> m0[42];
measure q[43] -> m0[43];
measure q[44] -> m0[44];
measure q[45] -> m0[45];
measure q[46] -> m0[46];
measure q[47] -> m0[47];
measure q[48] -> m0[48];
measure q[49] -> m0[49];
measure q[50] -> m0[50];
measure q[51] -> m0[51];
measure q[52] -> m0[52];
measure q[53] -> m0[53];
measure q[54] -> m0[54];
measure q[55] -> m0[55];
measure q[56] -> m0[56];
measure q[57] -> m0[57];
measure q[58] -> m0[58];
measure q[59] -> m0[59];
measure q[60] -> m0[60];
measure q[61] -> m0[61];
measure q[62] -> m0[62];
measure q[63] -> m0[63];
measure q[64] -> m0[64];
measure q[65] -> m0[65];
measure q[66] -> m0[66];
measure q[67] -> m0[67];
measure q[68] -> m0[68];
measure q[69] -> m0[69];