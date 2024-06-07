OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c1[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
rzz(5.718295091431391) q[0],q[1];
rzz(5.718295091431391) q[0],q[2];
rzz(5.718295091431391) q[0],q[3];
rzz(5.718295091431391) q[1],q[2];
rzz(5.718295091431391) q[1],q[3];
rzz(5.718295091431391) q[2],q[3];
rzz(5.718295091431391) q[3],q[4];
rzz(5.718295091431391) q[4],q[5];
rzz(5.718295091431391) q[4],q[6];
rzz(5.718295091431391) q[4],q[7];
rzz(5.718295091431391) q[5],q[6];
rzz(5.718295091431391) q[5],q[7];
rzz(5.718295091431391) q[6],q[7];
rx(0.6256299943754943) q[0];
rx(0.6256299943754943) q[1];
rx(0.6256299943754943) q[2];
rx(0.6256299943754943) q[3];
rx(0.6256299943754943) q[4];
rx(0.6256299943754943) q[5];
rx(0.6256299943754943) q[6];
rx(0.6256299943754943) q[7];
measure q[0] -> c1[0];
measure q[1] -> c1[1];
measure q[2] -> c1[2];
measure q[3] -> c1[3];
measure q[4] -> c1[4];
measure q[5] -> c1[5];
measure q[6] -> c1[6];
measure q[7] -> c1[7];
