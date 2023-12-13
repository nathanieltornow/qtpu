OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - 1.0*param1) q0; }
qreg q[28];
creg c12[28];
r(0.00659209691347318,pi/2) q[0];
r(0.181601579706238,pi/2) q[1];
r(0.146473252899436,pi/2) q[2];
r(0.72978828248247,pi/2) q[3];
r(0.915670158076121,pi/2) q[4];
r(0.323069325343832,pi/2) q[5];
r(0.635754973523144,pi/2) q[6];
r(0.509345331511018,pi/2) q[7];
r(0.385274545205406,pi/2) q[8];
r(0.978084326241331,pi/2) q[9];
r(0.208080685193579,pi/2) q[10];
r(0.19367692174476,pi/2) q[11];
r(0.851114245352766,pi/2) q[12];
r(0.960641809794693,pi/2) q[13];
r(0.453878121685104,pi/2) q[14];
r(0.534923146428902,pi/2) q[15];
r(0.70005008664867,pi/2) q[16];
r(0.0926701311641666,pi/2) q[17];
r(0.0885557773246924,pi/2) q[18];
r(0.236651902428483,pi/2) q[19];
r(0.131838994374047,pi/2) q[20];
r(0.296694403144428,pi/2) q[21];
r(0.87555108487878,pi/2) q[22];
r(0.887817490837993,pi/2) q[23];
r(0.547337193029149,pi/2) q[24];
r(0.773313682419525,pi/2) q[25];
r(0.0568608021764123,pi/2) q[26];
r(0.936425206297213,pi/2) q[27];
cx q[26],q[27];
cx q[25],q[26];
cx q[24],q[25];
cx q[23],q[24];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
cx q[10],q[11];
r(0.849817094410823,pi/2) q[11];
r(0.313816551282973,pi/2) q[12];
r(0.537282524935241,pi/2) q[13];
r(0.107807446133362,pi/2) q[14];
r(0.594735053644414,pi/2) q[15];
r(0.538137009982558,pi/2) q[16];
r(0.120535747888713,pi/2) q[17];
r(0.675572296324131,pi/2) q[18];
r(0.988118836405598,pi/2) q[19];
r(0.148141654318317,pi/2) q[20];
r(0.368524106664281,pi/2) q[21];
r(0.383857683531223,pi/2) q[22];
r(0.420961775670824,pi/2) q[23];
r(0.750396333927169,pi/2) q[24];
r(0.105837774689863,pi/2) q[25];
r(0.199180069849121,pi/2) q[26];
r(0.592579448853595,pi/2) q[27];
cx q[9],q[10];
r(0.37372211838695,pi/2) q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
r(0.304320070671607,pi/2) q[0];
r(0.853554696235669,pi/2) q[1];
r(0.113845313721084,pi/2) q[2];
r(0.312697148140104,pi/2) q[3];
r(0.779974740881008,pi/2) q[4];
r(0.584226193911913,pi/2) q[5];
r(0.0949740159530466,pi/2) q[6];
r(0.779995568740889,pi/2) q[7];
r(0.546112278023644,pi/2) q[8];
r(0.529182669988731,pi/2) q[9];
measure q[0] -> c12[0];
measure q[1] -> c12[1];
measure q[2] -> c12[2];
measure q[3] -> c12[3];
measure q[4] -> c12[4];
measure q[5] -> c12[5];
measure q[6] -> c12[6];
measure q[7] -> c12[7];
measure q[8] -> c12[8];
measure q[9] -> c12[9];
measure q[10] -> c12[10];
measure q[11] -> c12[11];
measure q[12] -> c12[12];
measure q[13] -> c12[13];
measure q[14] -> c12[14];
measure q[15] -> c12[15];
measure q[16] -> c12[16];
measure q[17] -> c12[17];
measure q[18] -> c12[18];
measure q[19] -> c12[19];
measure q[20] -> c12[20];
measure q[21] -> c12[21];
measure q[22] -> c12[22];
measure q[23] -> c12[23];
measure q[24] -> c12[24];
measure q[25] -> c12[25];
measure q[26] -> c12[26];
measure q[27] -> c12[27];