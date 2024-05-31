OPENQASM 2.0;
include "qelib1.inc";
gate r(param0,param1) q0 { u3(param0,param1 - pi/2,pi/2 - 1.0*param1) q0; }
qreg q[150];
creg c1[150];
r(0.014900002870114726,pi/2) q[0];
r(0.8758465031012945,pi/2) q[1];
r(0.7659367419587109,pi/2) q[2];
r(0.5797720123467673,pi/2) q[3];
r(0.5618660410023801,pi/2) q[4];
r(0.7157182966767196,pi/2) q[5];
r(0.9396528311383066,pi/2) q[6];
r(0.5551835294619394,pi/2) q[7];
r(0.6117414536813146,pi/2) q[8];
r(0.7641151636694832,pi/2) q[9];
r(0.06042696359191335,pi/2) q[10];
r(0.8217809627802435,pi/2) q[11];
r(0.05354259689396357,pi/2) q[12];
r(0.2520487952918872,pi/2) q[13];
r(0.7816935085697548,pi/2) q[14];
r(0.3395447041973888,pi/2) q[15];
r(0.5585110979647803,pi/2) q[16];
r(0.10471547714380258,pi/2) q[17];
r(0.7881684612132025,pi/2) q[18];
r(0.15497898587831382,pi/2) q[19];
r(0.73174313270274,pi/2) q[20];
r(0.13732573904213818,pi/2) q[21];
r(0.6945579190526509,pi/2) q[22];
r(0.3438004470778482,pi/2) q[23];
r(0.3103593790687902,pi/2) q[24];
r(0.20000898591599858,pi/2) q[25];
r(0.6360585311405639,pi/2) q[26];
r(0.5523219173704745,pi/2) q[27];
r(0.7401963630682088,pi/2) q[28];
r(0.4460228494477707,pi/2) q[29];
r(0.44187566425785096,pi/2) q[30];
r(0.35239905659332316,pi/2) q[31];
r(0.8036891357849684,pi/2) q[32];
r(0.5728689305720256,pi/2) q[33];
r(0.8043100704136811,pi/2) q[34];
r(0.8428591946167732,pi/2) q[35];
r(0.1336399389630758,pi/2) q[36];
r(0.3174914694975657,pi/2) q[37];
r(0.6572435387795484,pi/2) q[38];
r(0.16504769295456112,pi/2) q[39];
r(0.48770136812321374,pi/2) q[40];
r(0.7842279640316046,pi/2) q[41];
r(0.3542034589379709,pi/2) q[42];
r(0.6205071920926171,pi/2) q[43];
r(0.6773916040366835,pi/2) q[44];
r(0.819691061717887,pi/2) q[45];
r(0.14421860652078167,pi/2) q[46];
r(0.09442887904145048,pi/2) q[47];
r(0.145459108930856,pi/2) q[48];
r(0.8414941659631533,pi/2) q[49];
r(0.538386362962058,pi/2) q[50];
r(0.9071836924732671,pi/2) q[51];
r(0.6301687716617852,pi/2) q[52];
r(0.03685675394858379,pi/2) q[53];
r(0.0294437440500942,pi/2) q[54];
r(0.687975504852256,pi/2) q[55];
r(0.5398778404375277,pi/2) q[56];
r(0.5125877727563957,pi/2) q[57];
r(0.3675349889614503,pi/2) q[58];
r(0.2329497590392663,pi/2) q[59];
r(0.4581539669838429,pi/2) q[60];
r(0.8254690939640003,pi/2) q[61];
r(0.504070921213597,pi/2) q[62];
r(0.9053357215719245,pi/2) q[63];
r(0.9045753342033576,pi/2) q[64];
r(0.6693565083325751,pi/2) q[65];
r(0.4300911103797256,pi/2) q[66];
r(0.24829252283631598,pi/2) q[67];
r(0.4807279620348165,pi/2) q[68];
r(0.06630108876062224,pi/2) q[69];
r(0.3656443647609895,pi/2) q[70];
r(0.4370530168673278,pi/2) q[71];
r(0.967291933995769,pi/2) q[72];
r(0.37489016479042114,pi/2) q[73];
r(0.2682611481305811,pi/2) q[74];
r(0.7388445295926517,pi/2) q[75];
r(0.15947515116515898,pi/2) q[76];
r(0.8382380817080375,pi/2) q[77];
r(0.09990517140971733,pi/2) q[78];
r(0.29770230943161613,pi/2) q[79];
r(0.15423007706763447,pi/2) q[80];
r(0.9369218585189717,pi/2) q[81];
r(0.33204960287703666,pi/2) q[82];
r(0.477718697748154,pi/2) q[83];
r(0.32797939138564447,pi/2) q[84];
r(0.25730817990218713,pi/2) q[85];
r(0.4598964575269714,pi/2) q[86];
r(0.8223824275314663,pi/2) q[87];
r(0.634703954312423,pi/2) q[88];
r(0.5251970004476016,pi/2) q[89];
r(0.3325234748740893,pi/2) q[90];
r(0.820063652516098,pi/2) q[91];
r(0.4549824131979726,pi/2) q[92];
r(0.19098466588208673,pi/2) q[93];
r(0.5038453410110879,pi/2) q[94];
r(0.7840154924053994,pi/2) q[95];
r(0.7737751512796646,pi/2) q[96];
r(0.40666219120037017,pi/2) q[97];
r(0.6053687092253206,pi/2) q[98];
r(0.4146420130607089,pi/2) q[99];
r(0.7087122307973002,pi/2) q[100];
r(0.9095759757574067,pi/2) q[101];
r(0.5806241034177986,pi/2) q[102];
r(0.6166729898504377,pi/2) q[103];
r(0.3404170679118853,pi/2) q[104];
r(0.6837695319378115,pi/2) q[105];
r(0.12428342194917852,pi/2) q[106];
r(0.1309820542462511,pi/2) q[107];
r(0.8596905771591481,pi/2) q[108];
r(0.860677525848213,pi/2) q[109];
r(0.9829503281626185,pi/2) q[110];
r(0.3574542568220318,pi/2) q[111];
r(0.9765281856261697,pi/2) q[112];
r(0.6397051598387591,pi/2) q[113];
r(0.9508144077824389,pi/2) q[114];
r(0.8891564388100027,pi/2) q[115];
r(0.04719747156974341,pi/2) q[116];
r(0.9050665682179521,pi/2) q[117];
r(0.14861562762369873,pi/2) q[118];
r(0.30836938492406407,pi/2) q[119];
r(0.6580714880593006,pi/2) q[120];
r(0.08062102926135073,pi/2) q[121];
r(0.5726743333266583,pi/2) q[122];
r(0.11080400359823006,pi/2) q[123];
r(0.5598240274092575,pi/2) q[124];
r(0.4651567798130489,pi/2) q[125];
r(0.009644156288395433,pi/2) q[126];
r(0.4884816436534718,pi/2) q[127];
r(0.5237665995206737,pi/2) q[128];
r(0.767295980435833,pi/2) q[129];
r(0.23780095546370628,pi/2) q[130];
r(0.8962296451080891,pi/2) q[131];
r(0.5515477959385404,pi/2) q[132];
r(0.8063515601002375,pi/2) q[133];
r(0.8594644252225274,pi/2) q[134];
r(0.8960091647042641,pi/2) q[135];
r(0.8011648376272881,pi/2) q[136];
r(0.7923857568009407,pi/2) q[137];
r(0.818418409444944,pi/2) q[138];
r(0.5814988369858347,pi/2) q[139];
r(0.6756031307226793,pi/2) q[140];
r(0.6320199758025382,pi/2) q[141];
r(0.5961373930785527,pi/2) q[142];
r(0.7461373360874678,pi/2) q[143];
r(0.9927565851270798,pi/2) q[144];
r(0.44425734388326765,pi/2) q[145];
r(0.5428395095729719,pi/2) q[146];
r(0.25588457991741376,pi/2) q[147];
r(0.7305220481522069,pi/2) q[148];
r(0.05214683737937997,pi/2) q[149];
cx q[148],q[149];
cx q[147],q[148];
cx q[146],q[147];
cx q[145],q[146];
cx q[144],q[145];
cx q[143],q[144];
cx q[142],q[143];
cx q[141],q[142];
cx q[140],q[141];
cx q[139],q[140];
cx q[138],q[139];
cx q[137],q[138];
cx q[136],q[137];
cx q[135],q[136];
cx q[134],q[135];
cx q[133],q[134];
cx q[132],q[133];
cx q[131],q[132];
cx q[130],q[131];
cx q[129],q[130];
cx q[128],q[129];
cx q[127],q[128];
cx q[126],q[127];
cx q[125],q[126];
cx q[124],q[125];
cx q[123],q[124];
cx q[122],q[123];
cx q[121],q[122];
cx q[120],q[121];
cx q[119],q[120];
cx q[118],q[119];
cx q[117],q[118];
cx q[116],q[117];
cx q[115],q[116];
cx q[114],q[115];
cx q[113],q[114];
cx q[112],q[113];
cx q[111],q[112];
cx q[110],q[111];
cx q[109],q[110];
cx q[108],q[109];
cx q[107],q[108];
cx q[106],q[107];
cx q[105],q[106];
cx q[104],q[105];
cx q[103],q[104];
cx q[102],q[103];
cx q[101],q[102];
cx q[100],q[101];
cx q[99],q[100];
cx q[98],q[99];
cx q[97],q[98];
cx q[96],q[97];
cx q[95],q[96];
cx q[94],q[95];
cx q[93],q[94];
cx q[92],q[93];
cx q[91],q[92];
cx q[90],q[91];
cx q[89],q[90];
cx q[88],q[89];
cx q[87],q[88];
cx q[86],q[87];
cx q[85],q[86];
cx q[84],q[85];
cx q[83],q[84];
cx q[82],q[83];
cx q[81],q[82];
cx q[80],q[81];
cx q[79],q[80];
cx q[78],q[79];
cx q[77],q[78];
cx q[76],q[77];
cx q[75],q[76];
cx q[74],q[75];
cx q[73],q[74];
cx q[72],q[73];
cx q[71],q[72];
cx q[70],q[71];
cx q[69],q[70];
cx q[68],q[69];
cx q[67],q[68];
cx q[66],q[67];
cx q[65],q[66];
cx q[64],q[65];
cx q[63],q[64];
cx q[62],q[63];
cx q[61],q[62];
cx q[60],q[61];
cx q[59],q[60];
cx q[58],q[59];
cx q[57],q[58];
cx q[56],q[57];
cx q[55],q[56];
cx q[54],q[55];
cx q[53],q[54];
cx q[52],q[53];
cx q[51],q[52];
cx q[50],q[51];
cx q[49],q[50];
cx q[48],q[49];
cx q[47],q[48];
cx q[46],q[47];
cx q[45],q[46];
cx q[44],q[45];
cx q[43],q[44];
cx q[42],q[43];
cx q[41],q[42];
cx q[40],q[41];
cx q[39],q[40];
cx q[38],q[39];
cx q[37],q[38];
cx q[36],q[37];
cx q[35],q[36];
cx q[34],q[35];
cx q[33],q[34];
cx q[32],q[33];
cx q[31],q[32];
cx q[30],q[31];
cx q[29],q[30];
cx q[28],q[29];
cx q[27],q[28];
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
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
r(0.09873339730257757,pi/2) q[0];
r(0.1832950480254718,pi/2) q[1];
r(0.0496231237678364,pi/2) q[2];
r(0.9462210181302712,pi/2) q[3];
r(0.6595548105622746,pi/2) q[4];
r(0.9163896965675733,pi/2) q[5];
r(0.1200033097709593,pi/2) q[6];
r(0.19475883051302445,pi/2) q[7];
r(0.21896508235636403,pi/2) q[8];
r(0.9639030456546508,pi/2) q[9];
r(0.853896188576983,pi/2) q[10];
r(0.14804852739361818,pi/2) q[11];
r(0.6936794718929055,pi/2) q[12];
r(0.0449201109012799,pi/2) q[13];
r(0.5683921311314396,pi/2) q[14];
r(0.7778858527790241,pi/2) q[15];
r(0.3985668902078312,pi/2) q[16];
r(0.875236701988423,pi/2) q[17];
r(0.475436903625195,pi/2) q[18];
r(0.6718221320186336,pi/2) q[19];
r(0.47060285263557877,pi/2) q[20];
r(0.3645268408967809,pi/2) q[21];
r(0.34202004612112813,pi/2) q[22];
r(0.5361289631031145,pi/2) q[23];
r(0.053982430096398915,pi/2) q[24];
r(0.5106631333814453,pi/2) q[25];
r(0.7552556737163653,pi/2) q[26];
r(0.7552264754851143,pi/2) q[27];
r(0.7074542857184398,pi/2) q[28];
r(0.5257161258394957,pi/2) q[29];
r(0.9858311393897325,pi/2) q[30];
r(0.15600141918054145,pi/2) q[31];
r(0.8400312988175287,pi/2) q[32];
r(0.435140481355214,pi/2) q[33];
r(0.2989049381599789,pi/2) q[34];
r(0.2567250587971559,pi/2) q[35];
r(0.061117139949289245,pi/2) q[36];
r(0.5307370005948051,pi/2) q[37];
r(0.998191289711318,pi/2) q[38];
r(0.36118174623612587,pi/2) q[39];
r(0.4254263248509351,pi/2) q[40];
r(0.08356723861452586,pi/2) q[41];
r(0.8248040005984,pi/2) q[42];
r(0.8068313593553766,pi/2) q[43];
r(0.26020209493702895,pi/2) q[44];
r(0.8356447013997143,pi/2) q[45];
r(0.6636282717237438,pi/2) q[46];
r(0.8013579074812631,pi/2) q[47];
r(0.5673941841588358,pi/2) q[48];
r(0.4236608023414341,pi/2) q[49];
r(0.1646482936444683,pi/2) q[50];
r(0.44681723744291146,pi/2) q[51];
r(0.7635509264699124,pi/2) q[52];
r(0.9397758376186947,pi/2) q[53];
r(0.1211250030381692,pi/2) q[54];
r(0.14257213273326774,pi/2) q[55];
r(0.4214143822659161,pi/2) q[56];
r(0.9118852449517301,pi/2) q[57];
r(0.4173531685229527,pi/2) q[58];
r(0.0635548985386204,pi/2) q[59];
r(0.5219097036613903,pi/2) q[60];
r(0.2859388134705707,pi/2) q[61];
r(0.8674049389454828,pi/2) q[62];
r(0.8497020164891022,pi/2) q[63];
r(0.8216766835065124,pi/2) q[64];
r(0.3202959334408504,pi/2) q[65];
r(0.006585006493745493,pi/2) q[66];
r(0.6721262639102588,pi/2) q[67];
r(0.2614053644612463,pi/2) q[68];
r(0.060367561672476144,pi/2) q[69];
r(0.03418206367077081,pi/2) q[70];
r(0.1838034597383038,pi/2) q[71];
r(0.9060244469808286,pi/2) q[72];
r(0.022092808381622864,pi/2) q[73];
r(0.045650988667746084,pi/2) q[74];
r(0.934349525664337,pi/2) q[75];
r(0.4586307147018486,pi/2) q[76];
r(0.035684849262410556,pi/2) q[77];
r(0.05306927621509849,pi/2) q[78];
r(0.20519844852685387,pi/2) q[79];
r(0.6209344517395402,pi/2) q[80];
r(0.24797261943768922,pi/2) q[81];
r(0.7273307714591284,pi/2) q[82];
r(0.5370473694606184,pi/2) q[83];
r(0.10569689558771822,pi/2) q[84];
r(0.46497961979519864,pi/2) q[85];
r(0.8749076767904036,pi/2) q[86];
r(0.14552551499203725,pi/2) q[87];
r(0.8125854614470024,pi/2) q[88];
r(0.518187090544985,pi/2) q[89];
r(0.5157126658284121,pi/2) q[90];
r(0.22652682425942705,pi/2) q[91];
r(0.9830615114190441,pi/2) q[92];
r(0.05244984333145841,pi/2) q[93];
r(0.4583851324123889,pi/2) q[94];
r(0.4503023362364821,pi/2) q[95];
r(0.0989293293125092,pi/2) q[96];
r(0.060740310684762155,pi/2) q[97];
r(0.6512703651156296,pi/2) q[98];
r(0.19043387696981562,pi/2) q[99];
r(0.6443762651331532,pi/2) q[100];
r(0.5346966582289581,pi/2) q[101];
r(0.11582385039557286,pi/2) q[102];
r(0.024296077420768003,pi/2) q[103];
r(0.8617520005235307,pi/2) q[104];
r(0.8883052162295314,pi/2) q[105];
r(0.7007769677261062,pi/2) q[106];
r(0.8492934150274537,pi/2) q[107];
r(0.46283266549540303,pi/2) q[108];
r(0.7957264832192869,pi/2) q[109];
r(0.8854888632814636,pi/2) q[110];
r(0.8679547574652152,pi/2) q[111];
r(0.2172853852328358,pi/2) q[112];
r(0.03000527752982307,pi/2) q[113];
r(0.8513419391233757,pi/2) q[114];
r(0.5932602590583842,pi/2) q[115];
r(0.3337540137469457,pi/2) q[116];
r(0.9834319097514098,pi/2) q[117];
r(0.5868060341266531,pi/2) q[118];
r(0.8425033974016451,pi/2) q[119];
r(0.923053908814685,pi/2) q[120];
r(0.49951958114242223,pi/2) q[121];
r(0.6671241847530804,pi/2) q[122];
r(0.9114832845291218,pi/2) q[123];
r(0.23444484109639208,pi/2) q[124];
r(0.6572896045008536,pi/2) q[125];
r(0.4650510312906122,pi/2) q[126];
r(0.25262222859237715,pi/2) q[127];
r(0.6464189908786132,pi/2) q[128];
r(0.4235787330004831,pi/2) q[129];
r(0.46552648655942386,pi/2) q[130];
r(0.4836733630876223,pi/2) q[131];
r(0.7743311758736406,pi/2) q[132];
r(0.4258017609487368,pi/2) q[133];
r(0.8410959739799919,pi/2) q[134];
r(0.7180123536471447,pi/2) q[135];
r(0.31029637511977326,pi/2) q[136];
r(0.22344277834821646,pi/2) q[137];
r(0.4035539851613318,pi/2) q[138];
r(0.7030688304895142,pi/2) q[139];
r(0.4171174193169084,pi/2) q[140];
r(0.5187494996284043,pi/2) q[141];
r(0.1434792229441122,pi/2) q[142];
r(0.0008760376591476771,pi/2) q[143];
r(0.4023264281736326,pi/2) q[144];
r(0.9701010711884048,pi/2) q[145];
r(0.8420092315194045,pi/2) q[146];
r(0.6168128803022725,pi/2) q[147];
r(0.3204677630075722,pi/2) q[148];
r(0.015340353454702527,pi/2) q[149];
measure q[0] -> c1[0];
measure q[1] -> c1[1];
measure q[2] -> c1[2];
measure q[3] -> c1[3];
measure q[4] -> c1[4];
measure q[5] -> c1[5];
measure q[6] -> c1[6];
measure q[7] -> c1[7];
measure q[8] -> c1[8];
measure q[9] -> c1[9];
measure q[10] -> c1[10];
measure q[11] -> c1[11];
measure q[12] -> c1[12];
measure q[13] -> c1[13];
measure q[14] -> c1[14];
measure q[15] -> c1[15];
measure q[16] -> c1[16];
measure q[17] -> c1[17];
measure q[18] -> c1[18];
measure q[19] -> c1[19];
measure q[20] -> c1[20];
measure q[21] -> c1[21];
measure q[22] -> c1[22];
measure q[23] -> c1[23];
measure q[24] -> c1[24];
measure q[25] -> c1[25];
measure q[26] -> c1[26];
measure q[27] -> c1[27];
measure q[28] -> c1[28];
measure q[29] -> c1[29];
measure q[30] -> c1[30];
measure q[31] -> c1[31];
measure q[32] -> c1[32];
measure q[33] -> c1[33];
measure q[34] -> c1[34];
measure q[35] -> c1[35];
measure q[36] -> c1[36];
measure q[37] -> c1[37];
measure q[38] -> c1[38];
measure q[39] -> c1[39];
measure q[40] -> c1[40];
measure q[41] -> c1[41];
measure q[42] -> c1[42];
measure q[43] -> c1[43];
measure q[44] -> c1[44];
measure q[45] -> c1[45];
measure q[46] -> c1[46];
measure q[47] -> c1[47];
measure q[48] -> c1[48];
measure q[49] -> c1[49];
measure q[50] -> c1[50];
measure q[51] -> c1[51];
measure q[52] -> c1[52];
measure q[53] -> c1[53];
measure q[54] -> c1[54];
measure q[55] -> c1[55];
measure q[56] -> c1[56];
measure q[57] -> c1[57];
measure q[58] -> c1[58];
measure q[59] -> c1[59];
measure q[60] -> c1[60];
measure q[61] -> c1[61];
measure q[62] -> c1[62];
measure q[63] -> c1[63];
measure q[64] -> c1[64];
measure q[65] -> c1[65];
measure q[66] -> c1[66];
measure q[67] -> c1[67];
measure q[68] -> c1[68];
measure q[69] -> c1[69];
measure q[70] -> c1[70];
measure q[71] -> c1[71];
measure q[72] -> c1[72];
measure q[73] -> c1[73];
measure q[74] -> c1[74];
measure q[75] -> c1[75];
measure q[76] -> c1[76];
measure q[77] -> c1[77];
measure q[78] -> c1[78];
measure q[79] -> c1[79];
measure q[80] -> c1[80];
measure q[81] -> c1[81];
measure q[82] -> c1[82];
measure q[83] -> c1[83];
measure q[84] -> c1[84];
measure q[85] -> c1[85];
measure q[86] -> c1[86];
measure q[87] -> c1[87];
measure q[88] -> c1[88];
measure q[89] -> c1[89];
measure q[90] -> c1[90];
measure q[91] -> c1[91];
measure q[92] -> c1[92];
measure q[93] -> c1[93];
measure q[94] -> c1[94];
measure q[95] -> c1[95];
measure q[96] -> c1[96];
measure q[97] -> c1[97];
measure q[98] -> c1[98];
measure q[99] -> c1[99];
measure q[100] -> c1[100];
measure q[101] -> c1[101];
measure q[102] -> c1[102];
measure q[103] -> c1[103];
measure q[104] -> c1[104];
measure q[105] -> c1[105];
measure q[106] -> c1[106];
measure q[107] -> c1[107];
measure q[108] -> c1[108];
measure q[109] -> c1[109];
measure q[110] -> c1[110];
measure q[111] -> c1[111];
measure q[112] -> c1[112];
measure q[113] -> c1[113];
measure q[114] -> c1[114];
measure q[115] -> c1[115];
measure q[116] -> c1[116];
measure q[117] -> c1[117];
measure q[118] -> c1[118];
measure q[119] -> c1[119];
measure q[120] -> c1[120];
measure q[121] -> c1[121];
measure q[122] -> c1[122];
measure q[123] -> c1[123];
measure q[124] -> c1[124];
measure q[125] -> c1[125];
measure q[126] -> c1[126];
measure q[127] -> c1[127];
measure q[128] -> c1[128];
measure q[129] -> c1[129];
measure q[130] -> c1[130];
measure q[131] -> c1[131];
measure q[132] -> c1[132];
measure q[133] -> c1[133];
measure q[134] -> c1[134];
measure q[135] -> c1[135];
measure q[136] -> c1[136];
measure q[137] -> c1[137];
measure q[138] -> c1[138];
measure q[139] -> c1[139];
measure q[140] -> c1[140];
measure q[141] -> c1[141];
measure q[142] -> c1[142];
measure q[143] -> c1[143];
measure q[144] -> c1[144];
measure q[145] -> c1[145];
measure q[146] -> c1[146];
measure q[147] -> c1[147];
measure q[148] -> c1[148];
measure q[149] -> c1[149];
