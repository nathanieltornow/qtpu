OPENQASM 2.0;
include "qelib1.inc";
qreg q2[8];
creg meas[8];
h q2[0];
h q2[1];
h q2[2];
h q2[3];
h q2[4];
h q2[5];
h q2[6];
h q2[7];
p(1.5449765346220716) q2[0];
p(1.3918046736957774) q2[1];
p(0.2044921632645535) q2[2];
p(0.37912380644654686) q2[3];
p(1.5294995567414276) q2[4];
p(2.1899284936404517) q2[5];
p(0.7394364078920904) q2[6];
p(0.9562371971415632) q2[7];
rzz(1.3933352697087904) q2[0],q2[1];
rzz(1.9793234413668677) q2[1],q2[2];
rzz(2.455996129131364) q2[2],q2[3];
rzz(1.0176943531678753) q2[3],q2[4];
rzz(0.7835716572222406) q2[4],q2[5];
rzz(2.4184290842263567) q2[5],q2[6];
rzz(2.9369661404423772) q2[6],q2[7];
rzz(0.4008178012614665) q2[6],q2[7];
rzz(0.24607947663505106) q2[5],q2[6];
rzz(1.3825816515296097) q2[4],q2[5];
rzz(2.5612103261998915) q2[3],q2[4];
rzz(0.8998900725482402) q2[2],q2[3];
rzz(0.6251160490856928) q2[1],q2[2];
rzz(2.5242779985665202) q2[0],q2[1];
rz(0.20493151684526426) q2[0];
rz(1.7249084291574461) q2[1];
rz(1.57386379808984) q2[2];
rz(1.135880062790035) q2[3];
rz(0.0073672969450650815) q2[4];
rz(1.4291483030666703) q2[5];
rz(0.7311175607464303) q2[6];
rz(2.4040483586849453) q2[7];
h q2[0];
h q2[1];
h q2[2];
h q2[3];
h q2[4];
h q2[5];
h q2[6];
h q2[7];
barrier q2[0],q2[1],q2[2],q2[3],q2[4],q2[5],q2[6],q2[7];
measure q2[0] -> meas[0];
measure q2[1] -> meas[1];
measure q2[2] -> meas[2];
measure q2[3] -> meas[3];
measure q2[4] -> meas[4];
measure q2[5] -> meas[5];
measure q2[6] -> meas[6];
measure q2[7] -> meas[7];