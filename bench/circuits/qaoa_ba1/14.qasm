OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c4[14];
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
rzz(1.4560235639482728) q[0],q[1];
rzz(1.4560235639482728) q[0],q[9];
rzz(1.4560235639482728) q[0],q[11];
rzz(1.4560235639482728) q[1],q[2];
rzz(1.4560235639482728) q[1],q[3];
rzz(1.4560235639482728) q[1],q[4];
rzz(1.4560235639482728) q[1],q[6];
rzz(1.4560235639482728) q[1],q[7];
rzz(1.4560235639482728) q[1],q[8];
rzz(1.4560235639482728) q[1],q[10];
rzz(1.4560235639482728) q[1],q[13];
rzz(1.4560235639482728) q[3],q[5];
rzz(1.4560235639482728) q[4],q[12];
rx(2.09975898657441) q[0];
rx(2.09975898657441) q[1];
rx(2.09975898657441) q[2];
rx(2.09975898657441) q[3];
rx(2.09975898657441) q[4];
rx(2.09975898657441) q[5];
rx(2.09975898657441) q[6];
rx(2.09975898657441) q[7];
rx(2.09975898657441) q[8];
rx(2.09975898657441) q[9];
rx(2.09975898657441) q[10];
rx(2.09975898657441) q[11];
rx(2.09975898657441) q[12];
rx(2.09975898657441) q[13];
measure q[0] -> c4[0];
measure q[1] -> c4[1];
measure q[2] -> c4[2];
measure q[3] -> c4[3];
measure q[4] -> c4[4];
measure q[5] -> c4[5];
measure q[6] -> c4[6];
measure q[7] -> c4[7];
measure q[8] -> c4[8];
measure q[9] -> c4[9];
measure q[10] -> c4[10];
measure q[11] -> c4[11];
measure q[12] -> c4[12];
measure q[13] -> c4[13];
