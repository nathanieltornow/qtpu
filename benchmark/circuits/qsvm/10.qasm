OPENQASM 2.0;
include "qelib1.inc";
qreg q3[10];
creg meas[10];
h q3[0];
h q3[1];
h q3[2];
h q3[3];
h q3[4];
h q3[5];
h q3[6];
h q3[7];
h q3[8];
h q3[9];
p(2.9364269032675208) q3[0];
p(1.7740050144833013) q3[1];
p(1.9740269623214228) q3[2];
p(0.8749570369566105) q3[3];
p(2.817364192528738) q3[4];
p(1.7393872517043498) q3[5];
p(3.0165840949609133) q3[6];
p(0.8177165417299094) q3[7];
p(0.8798071914300016) q3[8];
p(1.772170314774745) q3[9];
rzz(1.4186024723170325) q3[0],q3[1];
rzz(1.9699506195403034) q3[1],q3[2];
rzz(2.778493103040092) q3[2],q3[3];
rzz(1.322893027767465) q3[3],q3[4];
rzz(1.2748223737889846) q3[4],q3[5];
rzz(2.235159365990628) q3[5],q3[6];
rzz(0.3078992345432299) q3[6],q3[7];
rzz(0.48339346578883263) q3[7],q3[8];
rzz(0.5006621164570546) q3[8],q3[9];
rzz(1.4873235115776973) q3[8],q3[9];
rzz(0.7800548497374964) q3[7],q3[8];
rzz(2.2377216225869767) q3[6],q3[7];
rzz(2.504786233993481) q3[5],q3[6];
rzz(0.12326960685040611) q3[4],q3[5];
rzz(0.9605357904830759) q3[3],q3[4];
rzz(1.3557901897816065) q3[2],q3[3];
rzz(0.9696413261199067) q3[1],q3[2];
rzz(1.3326339495984905) q3[0],q3[1];
rz(1.8833879551416697) q3[0];
rz(1.0113025181598114) q3[1];
rz(2.260219705789528) q3[2];
rz(2.9956125964963545) q3[3];
rz(1.2056198291179059) q3[4];
rz(1.772478855567131) q3[5];
rz(0.40106017322590354) q3[6];
rz(2.746320695988531) q3[7];
rz(1.8884742272217985) q3[8];
rz(0.5766486083531173) q3[9];
h q3[0];
h q3[1];
h q3[2];
h q3[3];
h q3[4];
h q3[5];
h q3[6];
h q3[7];
h q3[8];
h q3[9];
barrier q3[0],q3[1],q3[2],q3[3],q3[4],q3[5],q3[6],q3[7],q3[8],q3[9];
measure q3[0] -> meas[0];
measure q3[1] -> meas[1];
measure q3[2] -> meas[2];
measure q3[3] -> meas[3];
measure q3[4] -> meas[4];
measure q3[5] -> meas[5];
measure q3[6] -> meas[6];
measure q3[7] -> meas[7];
measure q3[8] -> meas[8];
measure q3[9] -> meas[9];