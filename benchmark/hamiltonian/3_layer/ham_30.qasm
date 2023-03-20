// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7), q(8), q(9), q(10), q(11), q(12), q(13), q(14), q(15), q(16), q(17), q(18), q(19), q(20), q(21), q(22), q(23), q(24), q(25), q(26), q(27), q(28), q(29)]
qreg q[30];
creg m0[30];  // Measurement: q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7),q(8),q(9),q(10),q(11),q(12),q(13),q(14),q(15),q(16),q(17),q(18),q(19),q(20),q(21),q(22),q(23),q(24),q(25),q(26),q(27),q(28),q(29)


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
cx q[0],q[1];
rz(pi*-0.5) q[1];
cx q[0],q[1];
cx q[1],q[2];
h q[0];
rz(pi*-0.5) q[2];
rz(pi*-0.7493178647) q[0];
cx q[1],q[2];
h q[0];
cx q[2],q[3];
h q[1];
rz(pi*-0.5) q[3];
rz(pi*-0.7493178647) q[1];
cx q[2],q[3];
h q[1];
cx q[3],q[4];
h q[2];
cx q[0],q[1];
rz(pi*-0.5) q[4];
rz(pi*-0.7493178647) q[2];
rz(pi*-0.5) q[1];
cx q[3],q[4];
h q[2];
cx q[0],q[1];
cx q[4],q[5];
h q[3];
cx q[1],q[2];
h q[0];
rz(pi*-0.5) q[5];
rz(pi*-0.7493178647) q[3];
rz(pi*-0.5) q[2];
rz(pi*-0.7479542144) q[0];
cx q[4],q[5];
h q[3];
cx q[1],q[2];
h q[0];
cx q[5],q[6];
h q[4];
cx q[2],q[3];
h q[1];
rz(pi*-0.5) q[6];
rz(pi*-0.7493178647) q[4];
rz(pi*-0.5) q[3];
rz(pi*-0.7479542144) q[1];
cx q[5],q[6];
h q[4];
cx q[2],q[3];
h q[1];
cx q[6],q[7];
h q[5];
cx q[3],q[4];
h q[2];
cx q[0],q[1];
rz(pi*-0.5) q[7];
rz(pi*-0.7493178647) q[5];
rz(pi*-0.5) q[4];
rz(pi*-0.7479542144) q[2];
rz(pi*-0.5) q[1];
cx q[6],q[7];
h q[5];
cx q[3],q[4];
h q[2];
cx q[0],q[1];
cx q[7],q[8];
h q[6];
cx q[4],q[5];
h q[3];
cx q[1],q[2];
rz(pi*-0.5) q[8];
rz(pi*-0.7493178647) q[6];
rz(pi*-0.5) q[5];
rz(pi*-0.7479542144) q[3];
rz(pi*-0.5) q[2];
cx q[7],q[8];
h q[6];
cx q[4],q[5];
h q[3];
cx q[1],q[2];
cx q[8],q[9];
h q[7];
cx q[5],q[6];
h q[4];
cx q[2],q[3];
rz(pi*-0.5) q[9];
rz(pi*-0.7493178647) q[7];
rz(pi*-0.5) q[6];
rz(pi*-0.7479542144) q[4];
rz(pi*-0.5) q[3];
cx q[8],q[9];
h q[7];
cx q[5],q[6];
h q[4];
cx q[2],q[3];
cx q[9],q[10];
h q[8];
cx q[6],q[7];
h q[5];
cx q[3],q[4];
rz(pi*-0.5) q[10];
rz(pi*-0.7493178647) q[8];
rz(pi*-0.5) q[7];
rz(pi*-0.7479542144) q[5];
rz(pi*-0.5) q[4];
cx q[9],q[10];
h q[8];
cx q[6],q[7];
h q[5];
cx q[3],q[4];
cx q[10],q[11];
h q[9];
cx q[7],q[8];
h q[6];
cx q[4],q[5];
rz(pi*-0.5) q[11];
rz(pi*-0.7493178647) q[9];
rz(pi*-0.5) q[8];
rz(pi*-0.7479542144) q[6];
rz(pi*-0.5) q[5];
cx q[10],q[11];
h q[9];
cx q[7],q[8];
h q[6];
cx q[4],q[5];
cx q[11],q[12];
h q[10];
cx q[8],q[9];
h q[7];
cx q[5],q[6];
rz(pi*-0.5) q[12];
rz(pi*-0.7493178647) q[10];
rz(pi*-0.5) q[9];
rz(pi*-0.7479542144) q[7];
rz(pi*-0.5) q[6];
cx q[11],q[12];
h q[10];
cx q[8],q[9];
h q[7];
cx q[5],q[6];
cx q[12],q[13];
h q[11];
cx q[9],q[10];
h q[8];
cx q[6],q[7];
rz(pi*-0.5) q[13];
rz(pi*-0.7493178647) q[11];
rz(pi*-0.5) q[10];
rz(pi*-0.7479542144) q[8];
rz(pi*-0.5) q[7];
cx q[12],q[13];
h q[11];
cx q[9],q[10];
h q[8];
cx q[6],q[7];
cx q[13],q[14];
h q[12];
cx q[10],q[11];
h q[9];
cx q[7],q[8];
rz(pi*-0.5) q[14];
rz(pi*-0.7493178647) q[12];
rz(pi*-0.5) q[11];
rz(pi*-0.7479542144) q[9];
rz(pi*-0.5) q[8];
cx q[13],q[14];
h q[12];
cx q[10],q[11];
h q[9];
cx q[7],q[8];
cx q[14],q[15];
h q[13];
cx q[11],q[12];
h q[10];
cx q[8],q[9];
rz(pi*-0.5) q[15];
rz(pi*-0.7493178647) q[13];
rz(pi*-0.5) q[12];
rz(pi*-0.7479542144) q[10];
rz(pi*-0.5) q[9];
cx q[14],q[15];
h q[13];
cx q[11],q[12];
h q[10];
cx q[8],q[9];
cx q[15],q[16];
h q[14];
cx q[12],q[13];
h q[11];
cx q[9],q[10];
rz(pi*-0.5) q[16];
rz(pi*-0.7493178647) q[14];
rz(pi*-0.5) q[13];
rz(pi*-0.7479542144) q[11];
rz(pi*-0.5) q[10];
cx q[15],q[16];
h q[14];
cx q[12],q[13];
h q[11];
cx q[9],q[10];
cx q[16],q[17];
h q[15];
cx q[13],q[14];
h q[12];
cx q[10],q[11];
rz(pi*-0.5) q[17];
rz(pi*-0.7493178647) q[15];
rz(pi*-0.5) q[14];
rz(pi*-0.7479542144) q[12];
rz(pi*-0.5) q[11];
cx q[16],q[17];
h q[15];
cx q[13],q[14];
h q[12];
cx q[10],q[11];
cx q[17],q[18];
h q[16];
cx q[14],q[15];
h q[13];
cx q[11],q[12];
rz(pi*-0.5) q[18];
rz(pi*-0.7493178647) q[16];
rz(pi*-0.5) q[15];
rz(pi*-0.7479542144) q[13];
rz(pi*-0.5) q[12];
cx q[17],q[18];
h q[16];
cx q[14],q[15];
h q[13];
cx q[11],q[12];
cx q[18],q[19];
h q[17];
cx q[15],q[16];
h q[14];
cx q[12],q[13];
rz(pi*-0.5) q[19];
rz(pi*-0.7493178647) q[17];
rz(pi*-0.5) q[16];
rz(pi*-0.7479542144) q[14];
rz(pi*-0.5) q[13];
cx q[18],q[19];
h q[17];
cx q[15],q[16];
h q[14];
cx q[12],q[13];
cx q[19],q[20];
h q[18];
cx q[16],q[17];
h q[15];
cx q[13],q[14];
rz(pi*-0.5) q[20];
rz(pi*-0.7493178647) q[18];
rz(pi*-0.5) q[17];
rz(pi*-0.7479542144) q[15];
rz(pi*-0.5) q[14];
cx q[19],q[20];
h q[18];
cx q[16],q[17];
h q[15];
cx q[13],q[14];
cx q[20],q[21];
h q[19];
cx q[17],q[18];
h q[16];
cx q[14],q[15];
rz(pi*-0.5) q[21];
rz(pi*-0.7493178647) q[19];
rz(pi*-0.5) q[18];
rz(pi*-0.7479542144) q[16];
rz(pi*-0.5) q[15];
cx q[20],q[21];
h q[19];
cx q[17],q[18];
h q[16];
cx q[14],q[15];
cx q[21],q[22];
h q[20];
cx q[18],q[19];
h q[17];
cx q[15],q[16];
rz(pi*-0.5) q[22];
rz(pi*-0.7493178647) q[20];
rz(pi*-0.5) q[19];
rz(pi*-0.7479542144) q[17];
rz(pi*-0.5) q[16];
cx q[21],q[22];
h q[20];
cx q[18],q[19];
h q[17];
cx q[15],q[16];
cx q[22],q[23];
h q[21];
cx q[19],q[20];
h q[18];
cx q[16],q[17];
rz(pi*-0.5) q[23];
rz(pi*-0.7493178647) q[21];
rz(pi*-0.5) q[20];
rz(pi*-0.7479542144) q[18];
rz(pi*-0.5) q[17];
cx q[22],q[23];
h q[21];
cx q[19],q[20];
h q[18];
cx q[16],q[17];
cx q[23],q[24];
h q[22];
cx q[20],q[21];
h q[19];
cx q[17],q[18];
rz(pi*-0.5) q[24];
rz(pi*-0.7493178647) q[22];
rz(pi*-0.5) q[21];
rz(pi*-0.7479542144) q[19];
rz(pi*-0.5) q[18];
cx q[23],q[24];
h q[22];
cx q[20],q[21];
h q[19];
cx q[17],q[18];
cx q[24],q[25];
h q[23];
cx q[21],q[22];
h q[20];
cx q[18],q[19];
rz(pi*-0.5) q[25];
rz(pi*-0.7493178647) q[23];
rz(pi*-0.5) q[22];
rz(pi*-0.7479542144) q[20];
rz(pi*-0.5) q[19];
cx q[24],q[25];
h q[23];
cx q[21],q[22];
h q[20];
cx q[18],q[19];
cx q[25],q[26];
h q[24];
cx q[22],q[23];
h q[21];
cx q[19],q[20];
rz(pi*-0.5) q[26];
rz(pi*-0.7493178647) q[24];
rz(pi*-0.5) q[23];
rz(pi*-0.7479542144) q[21];
rz(pi*-0.5) q[20];
cx q[25],q[26];
h q[24];
cx q[22],q[23];
h q[21];
cx q[19],q[20];
cx q[26],q[27];
h q[25];
cx q[23],q[24];
h q[22];
cx q[20],q[21];
rz(pi*-0.5) q[27];
rz(pi*-0.7493178647) q[25];
rz(pi*-0.5) q[24];
rz(pi*-0.7479542144) q[22];
rz(pi*-0.5) q[21];
cx q[26],q[27];
h q[25];
cx q[23],q[24];
h q[22];
cx q[20],q[21];
cx q[27],q[28];
h q[26];
cx q[24],q[25];
h q[23];
cx q[21],q[22];
rz(pi*-0.5) q[28];
rz(pi*-0.7493178647) q[26];
rz(pi*-0.5) q[25];
rz(pi*-0.7479542144) q[23];
rz(pi*-0.5) q[22];
cx q[27],q[28];
h q[26];
cx q[24],q[25];
h q[23];
cx q[21],q[22];
cx q[28],q[29];
h q[27];
cx q[25],q[26];
h q[24];
cx q[22],q[23];
rz(pi*-0.5) q[29];
rz(pi*-0.7493178647) q[27];
rz(pi*-0.5) q[26];
rz(pi*-0.7479542144) q[24];
rz(pi*-0.5) q[23];
cx q[28],q[29];
h q[27];
cx q[25],q[26];
h q[24];
cx q[22],q[23];
h q[28];
h q[29];
cx q[26],q[27];
h q[25];
cx q[23],q[24];
rz(pi*-0.7493178647) q[28];
rz(pi*-0.7493178647) q[29];
rz(pi*-0.5) q[27];
rz(pi*-0.7479542144) q[25];
rz(pi*-0.5) q[24];
h q[28];
h q[29];
cx q[26],q[27];
h q[25];
cx q[23],q[24];
cx q[27],q[28];
h q[26];
cx q[24],q[25];
rz(pi*-0.5) q[28];
rz(pi*-0.7479542144) q[26];
rz(pi*-0.5) q[25];
cx q[27],q[28];
h q[26];
cx q[24],q[25];
cx q[28],q[29];
h q[27];
cx q[25],q[26];
rz(pi*-0.5) q[29];
rz(pi*-0.7479542144) q[27];
rz(pi*-0.5) q[26];
cx q[28],q[29];
h q[27];
cx q[25],q[26];
h q[28];
h q[29];
cx q[26],q[27];
rz(pi*-0.7479542144) q[28];
rz(pi*-0.7479542144) q[29];
rz(pi*-0.5) q[27];
h q[28];
h q[29];
cx q[26],q[27];
cx q[27],q[28];
rz(pi*-0.5) q[28];
cx q[27],q[28];
cx q[28],q[29];
rz(pi*-0.5) q[29];
cx q[28],q[29];

// Gate: cirq.MeasurementGate(30, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7),q(8),q(9),q(10),q(11),q(12),q(13),q(14),q(15),q(16),q(17),q(18),q(19),q(20),q(21),q(22),q(23),q(24),q(25),q(26),q(27),q(28),q(29)'), ())
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
