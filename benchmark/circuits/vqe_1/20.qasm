OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - 1.0*param1) q0; }
qreg q[20];
creg c8[20];
r(0.729226721705983,pi/2) q[0];
r(0.951017884292869,pi/2) q[1];
r(0.0180162577851812,pi/2) q[2];
r(0.609115461389735,pi/2) q[3];
r(0.810941392107341,pi/2) q[4];
r(0.247366314117304,pi/2) q[5];
r(0.676602616103103,pi/2) q[6];
r(0.871223180468552,pi/2) q[7];
r(0.296398265156817,pi/2) q[8];
r(0.935969810323574,pi/2) q[9];
r(0.554540802195912,pi/2) q[10];
r(0.0271318505899996,pi/2) q[11];
r(0.157954398816439,pi/2) q[12];
r(0.732062176516144,pi/2) q[13];
r(0.670873810581533,pi/2) q[14];
r(0.882203794668489,pi/2) q[15];
r(0.0506580672783744,pi/2) q[16];
r(0.112463036778132,pi/2) q[17];
r(0.973180721644559,pi/2) q[18];
r(0.481594005180084,pi/2) q[19];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
cx q[10],q[11];
r(0.79127923564534,pi/2) q[11];
r(0.566265225736129,pi/2) q[12];
r(0.176051601141793,pi/2) q[13];
r(0.701442924215899,pi/2) q[14];
r(0.688901800317266,pi/2) q[15];
r(0.468168382243196,pi/2) q[16];
r(0.752732921928717,pi/2) q[17];
r(0.218026967726162,pi/2) q[18];
r(0.575803515779057,pi/2) q[19];
cx q[9],q[10];
r(0.294892568585493,pi/2) q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
r(0.78885869558988,pi/2) q[0];
r(0.0276210119126035,pi/2) q[1];
r(0.319812814818763,pi/2) q[2];
r(0.671417875979944,pi/2) q[3];
r(0.830539111691441,pi/2) q[4];
r(0.654163828624933,pi/2) q[5];
r(0.367750580666104,pi/2) q[6];
r(0.327884441766744,pi/2) q[7];
r(0.410597137951406,pi/2) q[8];
r(0.699119967608693,pi/2) q[9];
measure q[0] -> c8[0];
measure q[1] -> c8[1];
measure q[2] -> c8[2];
measure q[3] -> c8[3];
measure q[4] -> c8[4];
measure q[5] -> c8[5];
measure q[6] -> c8[6];
measure q[7] -> c8[7];
measure q[8] -> c8[8];
measure q[9] -> c8[9];
measure q[10] -> c8[10];
measure q[11] -> c8[11];
measure q[12] -> c8[12];
measure q[13] -> c8[13];
measure q[14] -> c8[14];
measure q[15] -> c8[15];
measure q[16] -> c8[16];
measure q[17] -> c8[17];
measure q[18] -> c8[18];
measure q[19] -> c8[19];
