OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg c39[80];
ry(0.543404941790965) q[0];
ry(0.278369385093796) q[1];
ry(0.424517590749133) q[2];
ry(0.844776132319904) q[3];
ry(0.00471885619097256) q[4];
ry(0.121569120783114) q[5];
ry(0.670749084726779) q[6];
ry(0.825852755105048) q[7];
ry(0.136706589684953) q[8];
ry(0.57509332942725) q[9];
ry(0.891321954312264) q[10];
ry(0.20920212211719) q[11];
ry(0.185328219550075) q[12];
ry(0.108376890464255) q[13];
ry(0.219697492624992) q[14];
ry(0.97862378470737) q[15];
ry(0.811683149089323) q[16];
ry(0.171941012732594) q[17];
ry(0.81622474872584) q[18];
ry(0.274073747041699) q[19];
ry(0.431704183663122) q[20];
ry(0.940029819622375) q[21];
ry(0.817649378776727) q[22];
ry(0.336111950120899) q[23];
ry(0.175410453742337) q[24];
ry(0.372832046289923) q[25];
ry(0.00568850735257342) q[26];
ry(0.25242635344484) q[27];
ry(0.795662508473287) q[28];
ry(0.015254971246339) q[29];
ry(0.598843376928493) q[30];
ry(0.603804539042854) q[31];
ry(0.105147685412056) q[32];
ry(0.38194344494311) q[33];
ry(0.0364760565925689) q[34];
ry(0.890411563442076) q[35];
ry(0.980920857012311) q[36];
ry(0.0599419888180373) q[37];
ry(0.890545944728504) q[38];
ry(0.576901499400033) q[39];
ry(0.742479689097977) q[40];
ry(0.630183936475376) q[41];
ry(0.581842192398778) q[42];
ry(0.0204391320269232) q[43];
ry(0.210026577672861) q[44];
ry(0.544684878178648) q[45];
ry(0.769115171105652) q[46];
ry(0.250695229138396) q[47];
ry(0.285895690406865) q[48];
ry(0.852395087841306) q[49];
ry(0.975006493606588) q[50];
ry(0.884853293491106) q[51];
ry(0.359507843936902) q[52];
ry(0.598858945875747) q[53];
ry(0.3547956116573) q[54];
ry(0.340190215370646) q[55];
ry(0.178080989505805) q[56];
ry(0.23769420862405) q[57];
ry(0.0448622824607753) q[58];
ry(0.505431429635789) q[59];
ry(0.376252454297363) q[60];
ry(0.592805400975887) q[61];
ry(0.629941875587497) q[62];
ry(0.142600314446284) q[63];
ry(0.933841299466419) q[64];
ry(0.946379880809101) q[65];
ry(0.602296657730866) q[66];
ry(0.387766280326631) q[67];
ry(0.36318800410935) q[68];
ry(0.204345276868644) q[69];
ry(0.276765061396335) q[70];
ry(0.24653588120355) q[71];
ry(0.173608001740205) q[72];
ry(0.966609694487324) q[73];
ry(0.957012600352798) q[74];
ry(0.597973684328921) q[75];
ry(0.731300753059923) q[76];
ry(0.340385222837436) q[77];
ry(0.0920556033772386) q[78];
ry(0.463498018937148) q[79];
rzz(0.508698893238194) q[79],q[0];
rzz(0.0884601730028908) q[0],q[1];
ry(0.208207240196023) q[0];
rzz(0.528035223318047) q[1],q[2];
ry(0.451258624061834) q[1];
rzz(0.992158036510528) q[2],q[3];
ry(0.491842910264054) q[2];
rzz(0.39503593175823) q[3],q[4];
ry(0.899076314793711) q[3];
rzz(0.335596441718568) q[4],q[5];
ry(0.729360461029441) q[4];
rzz(0.80545053732928) q[5],q[6];
ry(0.770089772919695) q[5];
rzz(0.754348994582354) q[6],q[7];
ry(0.375439247561988) q[6];
rzz(0.31306644158851) q[7],q[8];
ry(0.343739535235384) q[7];
rzz(0.634036682962275) q[8],q[9];
ry(0.655035205999322) q[8];
rzz(0.540404575300716) q[9],q[10];
rzz(0.296793750880015) q[10],q[11];
ry(0.113537575218676) q[10];
rzz(0.110787901182446) q[11],q[12];
ry(0.133028689373575) q[11];
rzz(0.312640297875743) q[12],q[13];
ry(0.456039057606124) q[12];
rzz(0.456979130049266) q[13],q[14];
ry(0.15973623015851) q[13];
rzz(0.658940070226197) q[14],q[15];
ry(0.961641903774646) q[14];
rzz(0.254257517817718) q[15],q[16];
ry(0.83761574486181) q[15];
rzz(0.641101258700702) q[16],q[17];
ry(0.520160687037923) q[16];
rzz(0.200123607218403) q[17],q[18];
ry(0.218272257728154) q[17];
rzz(0.657624805528984) q[18],q[19];
ry(0.134918722532399) q[18];
rzz(0.778289215449849) q[19],q[20];
ry(0.979070345483869) q[19];
rzz(0.77959839861075) q[20],q[21];
ry(0.707043495689143) q[20];
rzz(0.610328153209394) q[21],q[22];
ry(0.859975556945663) q[21];
rzz(0.309000348524402) q[22],q[23];
ry(0.387172627828639) q[22];
rzz(0.697734907512956) q[23],q[24];
ry(0.250834019831725) q[23];
rzz(0.859618295729065) q[24],q[25];
ry(0.299438018944702) q[24];
rzz(0.625323757756808) q[25],q[26];
ry(0.856895528405016) q[25];
rzz(0.98240782960955) q[26],q[27];
ry(0.472983990568221) q[26];
rzz(0.976500127015855) q[27],q[28];
ry(0.663277047016128) q[27];
rzz(0.166694131198858) q[28],q[29];
ry(0.805728607436785) q[28];
rzz(0.0231781364784036) q[29],q[30];
ry(0.252980504649724) q[29];
rzz(0.160744548507082) q[30],q[31];
ry(0.0795734389703249) q[30];
rzz(0.923496825259087) q[31],q[32];
ry(0.732760605015715) q[31];
rzz(0.953549849879534) q[32],q[33];
ry(0.961397477503605) q[32];
rzz(0.210978418718446) q[33],q[34];
ry(0.953804734167663) q[33];
rzz(0.360525250814608) q[34],q[35];
ry(0.4904990518839) q[34];
rzz(0.549375261627672) q[35],q[36];
ry(0.632192064432755) q[35];
rzz(0.271830849176972) q[36],q[37];
ry(0.732995019837992) q[36];
rzz(0.46060162107485) q[37],q[38];
ry(0.902409503247966) q[37];
rzz(0.696161564823385) q[38],q[39];
ry(0.1622469187482) q[38];
rzz(0.500355896674865) q[39],q[40];
ry(0.405881322367561) q[39];
rzz(0.716070990564336) q[40],q[41];
ry(0.417090735583661) q[40];
rzz(0.525955936229779) q[41],q[42];
ry(0.695591028292074) q[41];
rzz(0.00139902311904383) q[42],q[43];
ry(0.424847237924832) q[42];
rzz(0.394700286689835) q[43],q[44];
ry(0.85811422605143) q[43];
rzz(0.49216696990115) q[44],q[45];
ry(0.846932479609419) q[44];
rzz(0.402880331379142) q[45],q[46];
ry(0.0701991139086888) q[45];
rzz(0.354298300106321) q[46],q[47];
ry(0.301752413484148) q[46];
rzz(0.500614319442953) q[47],q[48];
ry(0.97962368103017) q[47];
rzz(0.445176628831138) q[48],q[49];
ry(0.0356269965530348) q[48];
rzz(0.0904327881964359) q[49],q[50];
ry(0.492392646998582) q[49];
rzz(0.273562920027441) q[50],q[51];
ry(0.952376853013546) q[50];
rzz(0.943477097742727) q[51],q[52];
ry(0.810573758529471) q[51];
rzz(0.026544641333942) q[52],q[53];
ry(0.294330441296371) q[52];
rzz(0.0399986896406508) q[53],q[54];
ry(0.596233518518341) q[53];
rzz(0.28314035971982) q[54],q[55];
ry(0.43117785229973) q[54];
rzz(0.582344170216769) q[55],q[56];
ry(0.592397502988986) q[55];
rzz(0.990892802924827) q[56],q[57];
ry(0.893752104720206) q[56];
rzz(0.992642237402968) q[57],q[58];
ry(0.554021189771706) q[57];
rzz(0.993117372481045) q[58],q[59];
ry(0.492866507345274) q[58];
rzz(0.110048330966563) q[59],q[60];
ry(0.319270457189501) q[59];
rzz(0.66448144596394) q[60],q[61];
ry(0.263365783050724) q[60];
rzz(0.523986834488313) q[61],q[62];
ry(0.542280613535796) q[61];
rzz(0.173149909808731) q[62],q[63];
ry(0.082264523932024) q[62];
rzz(0.942960244915026) q[63],q[64];
ry(0.635636709825399) q[63];
rzz(0.241860085976252) q[64],q[65];
ry(0.796405225186208) q[64];
rzz(0.998932268843212) q[65],q[66];
ry(0.954747505430809) q[65];
rzz(0.582693815149899) q[66],q[67];
ry(0.684624271692713) q[66];
rzz(0.183279000630576) q[67],q[68];
ry(0.488293166805099) q[67];
rzz(0.38684542191779) q[68],q[69];
ry(0.485414310184367) q[68];
rzz(0.18967352891215) q[69],q[70];
ry(0.966692920582968) q[69];
rzz(0.41077067302531) q[70],q[71];
ry(0.211347887497122) q[70];
rzz(0.594680068901705) q[71],q[72];
ry(0.411648138177833) q[71];
rzz(0.71658609312834) q[72],q[73];
ry(0.989665576779283) q[72];
rzz(0.486891482369123) q[73],q[74];
ry(0.0284118567132517) q[73];
rzz(0.309589817766705) q[74],q[75];
ry(0.701326514093516) q[74];
rzz(0.577441372827847) q[75],q[76];
ry(0.0251715638848119) q[75];
rzz(0.441707819568743) q[76],q[77];
ry(0.320881726086536) q[76];
rzz(0.359678102600536) q[77],q[78];
ry(0.0735270618655741) q[77];
rzz(0.321331932008814) q[78],q[79];
ry(0.0608845643466355) q[78];
ry(0.111406316704053) q[79];
rzz(0.169268908145431) q[79],q[0];
rzz(0.627686279501054) q[0],q[1];
ry(0.0604379321355856) q[0];
rzz(0.438393094639843) q[1],q[2];
ry(0.0909511672008748) q[1];
rzz(0.830903764603975) q[2],q[3];
ry(0.682706456428307) q[2];
rzz(0.239792189564472) q[3],q[4];
ry(0.680735767230638) q[3];
rzz(0.190052707919737) q[4],q[5];
ry(0.243174165887425) q[4];
rzz(0.711899658582916) q[5],q[6];
ry(0.64046144215603) q[5];
rzz(0.858294925326778) q[6],q[7];
ry(0.0691391831115514) q[6];
rzz(0.559055885596019) q[7],q[8];
ry(0.872919961746265) q[7];
ry(0.711037993210498) q[9];
rzz(0.704420408288857) q[8],q[9];
ry(0.109606949832513) q[8];
rzz(0.605112035518179) q[9],q[10];
rzz(0.55921728326804) q[10],q[11];
ry(0.467377991445348) q[10];
rzz(0.860394190907587) q[11],q[12];
ry(0.775949219403373) q[11];
rzz(0.91975535915007) q[12],q[13];
ry(0.854444515705057) q[12];
rzz(0.849607325758981) q[13],q[14];
ry(0.210386444767159) q[13];
rzz(0.254466535494455) q[14],q[15];
ry(0.076641869268903) q[14];
rzz(0.877555542286771) q[15],q[16];
ry(0.788914797103218) q[15];
rzz(0.435130190092226) q[16],q[17];
ry(0.547500001149302) q[16];
rzz(0.729494343964508) q[17],q[18];
ry(0.786254862815412) q[17];
rzz(0.412640767538794) q[18],q[19];
ry(0.92004704285871) q[18];
rzz(0.190836045811222) q[19],q[20];
ry(0.480972765927881) q[19];
rzz(0.706019519956162) q[20],q[21];
ry(0.459553670010184) q[20];
rzz(0.240632820929853) q[21],q[22];
ry(0.598979155480953) q[21];
rzz(0.851324426832995) q[22],q[23];
ry(0.599318780724284) q[22];
rzz(0.824102289258587) q[23],q[24];
ry(0.504373451233758) q[23];
rzz(0.52521178661397) q[24],q[25];
ry(0.306878529743451) q[24];
rzz(0.386340794306168) q[25],q[26];
ry(0.541352980349113) q[25];
rzz(0.590880790734925) q[26],q[27];
ry(0.924926943402792) q[26];
rzz(0.137523614907826) q[27],q[28];
ry(0.970550802030219) q[27];
rzz(0.808270407891608) q[28],q[29];
ry(0.395794609896996) q[28];
rzz(0.965825815244485) q[29],q[30];
ry(0.798745272555339) q[29];
rzz(0.779795804232935) q[30],q[31];
ry(0.635088147750926) q[30];
rzz(0.23933508209582) q[31],q[32];
ry(0.229969165249033) q[31];
rzz(0.867260413108431) q[32],q[33];
ry(0.0512070928629372) q[32];
rzz(0.8081150128937) q[33],q[34];
ry(0.0284638061713518) q[33];
rzz(0.0636811242204685) q[34],q[35];
ry(0.122847751903769) q[34];
rzz(0.231228304048803) q[35],q[36];
ry(0.220212517833639) q[35];
rzz(0.589685448703527) q[36],q[37];
ry(0.829022753700823) q[36];
rzz(0.137486947977774) q[37],q[38];
ry(0.285491827700039) q[37];
rzz(0.678440704371486) q[38],q[39];
ry(0.781064082631092) q[38];
rzz(0.992190689515247) q[39],q[40];
ry(0.504665812596679) q[39];
rzz(0.285751984815572) q[40],q[41];
ry(0.138448923776585) q[40];
rzz(0.76091275955885) q[41],q[42];
ry(0.778036552333978) q[41];
rzz(0.0465271676661386) q[42],q[43];
ry(0.921331793520843) q[42];
rzz(0.332535906522207) q[43],q[44];
ry(0.943018632036191) q[43];
rzz(0.944552791027057) q[44],q[45];
ry(0.704435797263943) q[44];
rzz(0.636517041254779) q[45],q[46];
ry(0.693916445475909) q[45];
rzz(0.601848606131893) q[46],q[47];
ry(0.546551813367495) q[46];
rzz(0.928184681463646) q[47],q[48];
ry(0.369217229737883) q[47];
rzz(0.181679410796613) q[48],q[49];
ry(0.982467574726993) q[48];
rzz(0.0178231840265211) q[49],q[50];
ry(0.0656092254275048) q[49];
rzz(0.190072176134172) q[50],q[51];
ry(0.897678310742266) q[50];
rzz(0.52187179782459) q[51],q[52];
ry(0.263930989094871) q[51];
rzz(0.49582198590367) q[52],q[53];
ry(0.574475842023338) q[52];
rzz(0.800491205569153) q[53],q[54];
ry(0.512866268093788) q[53];
rzz(0.859436311444911) q[54],q[55];
ry(0.554476808071934) q[54];
rzz(0.212956032240345) q[55],q[56];
ry(0.647167331828399) q[55];
rzz(0.437268841447664) q[56],q[57];
ry(0.185474159054481) q[56];
rzz(0.42161750906259) q[57],q[58];
ry(0.271978204309022) q[57];
rzz(0.0547173770819418) q[58],q[59];
ry(0.148438657742784) q[58];
rzz(0.00993369367098296) q[59],q[60];
ry(0.030304170594405) q[59];
rzz(0.789765679336007) q[60],q[61];
ry(0.939255584152139) q[60];
rzz(0.275313218794898) q[61],q[62];
ry(0.346754130176204) q[61];
rzz(0.717740004543336) q[62],q[63];
ry(0.10956460430561) q[62];
rzz(0.421355921756556) q[63],q[64];
ry(0.37832699566147) q[63];
rzz(0.143335872169) q[64],q[65];
ry(0.384079459041414) q[64];
rzz(0.192521680251979) q[65],q[66];
ry(0.665423687919461) q[65];
rzz(0.313815233317635) q[66],q[67];
ry(0.24449092930616) q[66];
rzz(0.805170167367297) q[67],q[68];
ry(0.661480973773685) q[67];
rzz(0.01262583035748) q[68],q[69];
ry(0.0984928837918622) q[68];
rzz(0.0491059726975559) q[69],q[70];
ry(0.580862747666852) q[69];
rzz(0.566000384985094) q[70],q[71];
ry(0.106865504668551) q[70];
rzz(0.686810696154396) q[71],q[72];
ry(0.548254507463988) q[71];
rzz(0.726810905051707) q[72],q[73];
ry(0.519751707711207) q[72];
rzz(0.479693761301677) q[73],q[74];
ry(0.296234117865758) q[73];
rzz(0.367656721778528) q[74],q[75];
ry(0.455729050998098) q[74];
rzz(0.839970099201751) q[75],q[76];
ry(0.0386665201295668) q[75];
rzz(0.454163552451005) q[76],q[77];
ry(0.599003024888501) q[76];
rzz(0.321365838782199) q[77],q[78];
ry(0.00048675947525989) q[77];
rzz(0.0927198671710415) q[78],q[79];
ry(0.501813593685531) q[78];
ry(0.501726121691614) q[79];
ry(0.169055765119435) q[9];
measure q[0] -> c39[0];
measure q[1] -> c39[1];
measure q[2] -> c39[2];
measure q[3] -> c39[3];
measure q[4] -> c39[4];
measure q[5] -> c39[5];
measure q[6] -> c39[6];
measure q[7] -> c39[7];
measure q[8] -> c39[8];
measure q[9] -> c39[9];
measure q[10] -> c39[10];
measure q[11] -> c39[11];
measure q[12] -> c39[12];
measure q[13] -> c39[13];
measure q[14] -> c39[14];
measure q[15] -> c39[15];
measure q[16] -> c39[16];
measure q[17] -> c39[17];
measure q[18] -> c39[18];
measure q[19] -> c39[19];
measure q[20] -> c39[20];
measure q[21] -> c39[21];
measure q[22] -> c39[22];
measure q[23] -> c39[23];
measure q[24] -> c39[24];
measure q[25] -> c39[25];
measure q[26] -> c39[26];
measure q[27] -> c39[27];
measure q[28] -> c39[28];
measure q[29] -> c39[29];
measure q[30] -> c39[30];
measure q[31] -> c39[31];
measure q[32] -> c39[32];
measure q[33] -> c39[33];
measure q[34] -> c39[34];
measure q[35] -> c39[35];
measure q[36] -> c39[36];
measure q[37] -> c39[37];
measure q[38] -> c39[38];
measure q[39] -> c39[39];
measure q[40] -> c39[40];
measure q[41] -> c39[41];
measure q[42] -> c39[42];
measure q[43] -> c39[43];
measure q[44] -> c39[44];
measure q[45] -> c39[45];
measure q[46] -> c39[46];
measure q[47] -> c39[47];
measure q[48] -> c39[48];
measure q[49] -> c39[49];
measure q[50] -> c39[50];
measure q[51] -> c39[51];
measure q[52] -> c39[52];
measure q[53] -> c39[53];
measure q[54] -> c39[54];
measure q[55] -> c39[55];
measure q[56] -> c39[56];
measure q[57] -> c39[57];
measure q[58] -> c39[58];
measure q[59] -> c39[59];
measure q[60] -> c39[60];
measure q[61] -> c39[61];
measure q[62] -> c39[62];
measure q[63] -> c39[63];
measure q[64] -> c39[64];
measure q[65] -> c39[65];
measure q[66] -> c39[66];
measure q[67] -> c39[67];
measure q[68] -> c39[68];
measure q[69] -> c39[69];
measure q[70] -> c39[70];
measure q[71] -> c39[71];
measure q[72] -> c39[72];
measure q[73] -> c39[73];
measure q[74] -> c39[74];
measure q[75] -> c39[75];
measure q[76] -> c39[76];
measure q[77] -> c39[77];
measure q[78] -> c39[78];
measure q[79] -> c39[79];
