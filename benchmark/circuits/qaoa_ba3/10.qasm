OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c24[10];
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
rzz(6.148667335926924) q[0],q[1];
rzz(6.148667335926924) q[0],q[2];
rzz(6.148667335926924) q[0],q[3];
rzz(6.148667335926924) q[0],q[4];
rzz(6.148667335926924) q[0],q[5];
rzz(6.148667335926924) q[0],q[6];
rzz(6.148667335926924) q[0],q[7];
rzz(6.148667335926924) q[1],q[4];
rzz(6.148667335926924) q[2],q[8];
rzz(6.148667335926924) q[3],q[4];
rzz(6.148667335926924) q[3],q[5];
rzz(6.148667335926924) q[3],q[6];
rzz(6.148667335926924) q[3],q[7];
rzz(6.148667335926924) q[3],q[9];
rzz(6.148667335926924) q[4],q[5];
rzz(6.148667335926924) q[4],q[6];
rzz(6.148667335926924) q[4],q[8];
rzz(6.148667335926924) q[4],q[9];
rzz(6.148667335926924) q[5],q[7];
rzz(6.148667335926924) q[6],q[8];
rzz(6.148667335926924) q[6],q[9];
rx(0.5177959161738424) q[0];
rx(0.5177959161738424) q[1];
rx(0.5177959161738424) q[2];
rx(0.5177959161738424) q[3];
rx(0.5177959161738424) q[4];
rx(0.5177959161738424) q[5];
rx(0.5177959161738424) q[6];
rx(0.5177959161738424) q[7];
rx(0.5177959161738424) q[8];
rx(0.5177959161738424) q[9];
measure q[0] -> c24[0];
measure q[1] -> c24[1];
measure q[2] -> c24[2];
measure q[3] -> c24[3];
measure q[4] -> c24[4];
measure q[5] -> c24[5];
measure q[6] -> c24[6];
measure q[7] -> c24[7];
measure q[8] -> c24[8];
measure q[9] -> c24[9];