// Generated from Cirq v1.1.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7), q(8), q(9), q(10), q(11), q(12), q(13), q(14), q(15), q(16), q(17), q(18), q(19), q(20), q(21), q(22), q(23), q(24), q(25), q(26), q(27), q(28), q(29), q(30), q(31), q(32), q(33), q(34), q(35), q(36), q(37), q(38), q(39)]
qreg q[40];
creg m0[40];  // Measurement: q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7),q(8),q(9),q(10),q(11),q(12),q(13),q(14),q(15),q(16),q(17),q(18),q(19),q(20),q(21),q(22),q(23),q(24),q(25),q(26),q(27),q(28),q(29),q(30),q(31),q(32),q(33),q(34),q(35),q(36),q(37),q(38),q(39)


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
cx q[27],q[28];
cx q[28],q[29];
cx q[29],q[30];
cx q[30],q[31];
cx q[31],q[32];
cx q[32],q[33];
cx q[33],q[34];
cx q[34],q[35];
cx q[35],q[36];
cx q[36],q[37];
cx q[37],q[38];
cx q[38],q[39];

// Gate: cirq.MeasurementGate(40, cirq.MeasurementKey(name='q(0),q(1),q(2),q(3),q(4),q(5),q(6),q(7),q(8),q(9),q(10),q(11),q(12),q(13),q(14),q(15),q(16),q(17),q(18),q(19),q(20),q(21),q(22),q(23),q(24),q(25),q(26),q(27),q(28),q(29),q(30),q(31),q(32),q(33),q(34),q(35),q(36),q(37),q(38),q(39)'), ())
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
