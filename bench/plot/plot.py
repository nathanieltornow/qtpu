from bench.plot.util import *
import os
from bench.plot.bar_plot import *
from bench.plot.data import *
HIGHERISBETTER = "Higher is better ↑"
LOWERISBETTER = "Lower is better ↓"

from bench.plot.get_average import get_average

def plot_fidelities():
	FID_DATA = {
        "BV": "bench/results/scale_sim/bv_100.csv",
		"GHZ": "bench/results/scale_sim/ghz_100.csv",
		"HS-3": "bench/results/scale_sim/hamsim_3_100.csv",
		"QAOA-B": "bench/results/scale_sim/qaoa_ba3_100.csv",
        "QAOA-R": "bench/results/scale_sim/qaoa_r2_100.csv",
		"TL-3": "bench/results/scale_sim/twolocal_3_100.csv",
		"VQE-2": "bench/results/scale_sim/vqe_2_100.csv",
		"WSTATE": "bench/results/scale_sim/wstate_100.csv",
    }

	dfs = [pd.read_csv(file) for file in FID_DATA.values()]
	labels = list(FID_DATA.keys())

	fig, ax = plt.subplots(ncols=1, figsize=WIDE_FIGSIZE)
	xvalues = [4, 6, 8, 10, 12]

	ax.set_ylim(0, 1.0)

	y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "h_fid")
	
	print(y)
	grouped_bar_plot(ax, y.T, yerr.T, labels, show_average_text=True)

	ax.set_ylabel("Fidelity")
	ax.set_title("", fontweight="bold", fontsize=FONTSIZE)
	ax.set_xlabel("Number of Qubits")
	ax.set_xticklabels(xvalues)
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=14)
	handles, labels = ax.get_legend_handles_labels()

	fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=10,
        frameon=False,
    )

	output_file = "figures/scale_sim/scalability_results.pdf"
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")

def plot_cutting():
	CUT_DATA = {
		"20": "bench/results/scale_sim/ghz_20.csv",
		"10": "bench/results/scale_sim/ghz_10.csv",        
		"8": "bench/results/scale_sim/ghz_8.csv",
		"5": "bench/results/scale_sim/ghz_5.csv",
		"3": "bench/results/scale_sim/ghz_3.csv",
    }

	dfs = [pd.read_csv(file) for file in CUT_DATA.values()]
	labels = list(CUT_DATA.keys())

	fig, ax = plt.subplots(ncols=1, figsize=COLUMN_FIGSIZE)
	xvalues = [20, 10, 8, 5, 3]

	ax.set_ylim(0, 0.5)

	y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array([20]), "h_fid")
	
	#print(y, yerr)
	#plot_abs22(ax, dfs, ["h_fid"], labels, "num_qubits", xvalues)
	custom_bar_plot(ax, y.T, yerr.T, labels, spacing=3)

	ax.set_ylabel("Fidelity")
	ax.set_title("", fontweight="bold", fontsize=FONTSIZE)
	ax.set_xlabel("Maximum Fragment Size [Number of Qubits]")
	#ax.set_xticklabels(xvalues)
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=14)
	handles, labels = ax.get_legend_handles_labels()
	#fig.legend()

	#fig.legend(
       # handles,
       # labels,
       # loc="lower center",
       # bbox_to_anchor=(0.5, -0.1),
       # ncol=10,
       # frameon=False,
    #)

	output_file = "figures/scale_sim/scalability_proposal.pdf"
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")


def plot_utilization():
	data = {
		"127": {
			"hamsim_3" : 11.8,
			"ghz" : 12.5,
			"bv": 11,
			"qaoa-b": 9.4,
			"qaoa-r": 10.2,
			"tl-3": 10.2,
			"vqe-2": 9.4,
			"wstate": 11.8,
		},
		"65" : {
			"hamsim_3": 20.9,
			"ghz" : 14.15,
			"bv": 21.3,
			"qaoa-b": 15.8,
			"qaoa-r": 27.3,
			"tl-3": 29.7,
			"vqe-2": 11.8,
			"wstate": 13.5, 
		},
		"27" : {
			"hamsim_3": 25.9,
			"ghz" : 18.5,
			"bv": 29.6,
			"qaoa-b": 29.6,
			"qaoa-r": 37.3,
			"tl-3": 40.7,
			"vqe-2": 14.8,
			"wstate": 18.5, 
		},
		"16" : {
			"hamsim_3": 81,
			"ghz" : 56.5,
			"bv": 62.5,
			"qaoa-b": 68,
			"qaoa-r": 81,
			"tl-3": 81,
			"vqe-2": 50,
			"wstate": 50, 
		},
		"7" : {
			"hamsim_3" : 100,
			"ghz" : 95,
			"bv": 90,
			"qaoa-b": 100,
			"qaoa-r": 100,
			"tl-3": 100,
			"vqe-2": 85,
			"wstate": 95,
		}
	}

	new_data = {
		"127": {
			"hamsim_3" : 34.8,
			"ghz" : 36.5,
			"bv": 32,
			"qaoa-b": 28.5,
			"qaoa-r": 34.3,
			"tl-3": 31.3,
			"vqe-2": 27.8,
			"wstate": 34.5,
		},
		"65" : {
			"hamsim_3": 61.9,
			"ghz" : 44.5,
			"bv": 62.3,
			"qaoa-b": 44.9,
			"qaoa-r": 71.5,
			"tl-3": 82.1,
			"vqe-2": 31.6,
			"wstate": 41.3, 
		},
		"27" : {
			"hamsim_3": 78.3,
			"ghz" : 56.32,
			"bv": 90.2,
			"qaoa-b": 88.5,
			"qaoa-r": 84.3,
			"tl-3": 94.7,
			"vqe-2": 52.3,
			"wstate": 54.5, 
		},
		"16" : {
			"hamsim_3": 100,
			"ghz" : 76.5,
			"bv": 82.5,
			"qaoa-b": 88,
			"qaoa-r": 100,
			"tl-3": 100,
			"vqe-2": 70,
			"wstate": 70, 
		},
		"7" : {
			"hamsim_3" : 100,
			"ghz" : 100,
			"bv": 100,
			"qaoa-b": 100,
			"qaoa-r": 100,
			"tl-3": 100,
			"vqe-2": 100,
			"wstate": 100,
		}
	}


	values = []
	for v in data.values():
		tmp = []	
		for vv in v.values():
			tmp.append(vv)
		values.append(tmp)

	values2 = []
	for v in new_data.values():
		tmp = []	
		for vv in v.values():
			tmp.append(vv)
		values2.append(tmp)
	
	means = [np.mean(v) for v in values]
	means.reverse()
	means2 = [np.mean(v) for v in values2]
	means2.reverse()

	y = np.array([means, means2])	
	yerr = np.zeros(len(means))
	yerr = np.array([yerr, yerr])

	xvalues = ["7", "16", "27", "65", "127"]
	labels = ["Baseline", "w/ Multi-programming"]

	fig, ax = plt.subplots(ncols=1, figsize=COLUMN_FIGSIZE)

	grouped_bar_plot(ax, y.T, yerr.T, labels)

	ax.set_ylabel("Utilization [%]")
	ax.set_title("", fontweight="bold", fontsize=FONTSIZE)
	ax.set_xlabel("QPU Size [Number of Qubits]")
	ax.set_xticklabels(xvalues)
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=14)
	ax.legend()

	output_file = "figures/scale_sim/utilization_results.pdf"
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")


def plot_spatial_variance():
	data = {"fake_cairo" : [0.8962961780981665, 0.8964425013722386, 0.8970863768346037, 0.8973690823724951, 0.8976488681078874],
	"fake_guadalupe": [0.8315360091541378, 0.8324275961694153, 0.8353847090976664, 0.8354746832012874, 0.8403469853392719] ,
	"fake_jakarta": [0.7835302890395199, 0.7865838485981893, 0.7879302885459152, 0.7882868862395279, 0.7920969023015155],
	"fake_kolkata": [0.917298028570531, 0.9178851226288793, 0.9200866664031844, 0.9206942522919213, 0.920974734897659] ,
	"fake_lagos": [0.8885110038999398, 0.8893666058911175, 0.8904675410477862, 0.8908724305281229, 0.8917161632170806],
	"fake_mumbai":[0.790515907352535, 0.7933494762629415, 0.7933611783747272, 0.7958892519803716, 0.8002791754891536],
	"fake_perth": [0.8029098342541645, 0.8056482416190556, 0.8143229130116179, 0.8161225508293457, 0.8243053714433508],
	"fake_prague": [0.9164787103136032, 0.9171999482642089, 0.9176523126331453, 0.9185221065229016, 0.9189276534712143],
	"fake_sherbrooke": [0.9316380092049327, 0.9327537555801326, 0.933361899726868, 0.9344816315813117, 0.9375357602745984],
	"fake_toronto": [0.6574017240255662, 0.6584931477131187, 0.6589934782746221, 0.6622117922599726, 0.6638642011419673],
	}

	values = [np.mean(v) for v in list(data.values())]
	y = np.array([values])
	median = np.median(values)
	yerr = [np.std(v) for v in list(data.values())]
	yerr = np.array([yerr])
	#print(len(values))
	#print(len(sns.color_palette("pastel")))

	labels = [s.split("_")[1] for s in data.keys()]

	fig, ax = plt.subplots(ncols=1, figsize=COLUMN_FIGSIZE)
	ax.set_ylim(0.5, 1.0)
	custom_bar_plot(ax,y, yerr, labels, spacing=4)

	ax.axhline(median, color=sns.color_palette("pastel")[3], linestyle="--")
	#ax.axhline(np.amin(values), color="lightcoral", linestyle="--")
	ax.set_ylabel("Fidelity")
	ax.set_title("", fontweight="bold", fontsize=FONTSIZE)
	ax.set_xlabel("IBMQ Backend")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=14)

	output_file = "figures/scale_sim/spatial_results.pdf"
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")

def plot_temporal_variance():
	y = [0.8180191746, 0.8220953639, 0.8216993235, 0.822515943, 0.805091413, 0.8183074096, 0.8365701316, 0.8246906414, 0.8214604949, 0.8160748275, 0.7595980513, 0.8251312511, 
        0.842101672, 0.840062022, 0.8406734673, 0.8427749555, 0.8385395298, 0.8243551136, 0.8550852817, 0.8407238751, 0.8231059173, 0.8316884878, 0.8251012964, 0.8257370426, 0.822763477,
	    0.8344936881, 0.8313854624, 0.8376180401, 0.8159783738, 0.8376200286, 0.8381760733, 0.8059318756, 0.8185026766, 0.8191160439, 0.8182886122, 0.8235175986, 0.8198328236, 0.8237580375, 
		0.8216143624, 0.8364219555, 0.8281693987, 0.8409689012, 0.8155441324, 0.7516793746, 0.5625267459, 0.6153751898, 0.7811350703, 0.8173997766, 0.8184834656, 0.8127444147, 0.8182328503,
		0.8312488087, 0.8317334916, 0.822833798, 0.8277281716, 0.8324118164, 0.8428896699, 0.8446135212, 0.8279863612, 0.8281418395, 0.8313925868, 0.7851758116, 0.8164898418, 0.8349636139,
		0.7994006936, 0.8085122492, 0.7892250253, 0.7968751405, 0.8100067243, 0.813981553, 0.8115752652, 0.7815751914, 0.7978490471, 0.8129667651, 0.7683013934, 0.7753786759, 0.762515556,
		0.8161338068, 0.8039958121, 0.1176905513, 0.1152107419, 0.05034899452, 0.1528359521, 0.1556395552, 0.779896242, 0.8069562604, 0.7978034473, 0.7977759019, 0.8013895057, 0.7951778163,
		0.7494026836, 0.1945077264, 0.6312009287, 0.6411222179, 0.6383937357, 0.761960609, 0.2772769759, 0.3058998266, 0.3236697611, 0.1940343924, 0.1981356321, 0.4968996507, 0.4552561435,
		0.5010237725, 0.5043802214, 0.5586196412, 0.5486795463, 0.5683461809, 0.4388483328, 0.7674307005, 0.7560229598, 0.6698343536, 0.7887219142, 0.7926850704, 0.7979401075, 0.8006216355, 
		0.8124279659, 0.8121399512, 0.8114361589, 0.8080806982, 0.8176319388, 0.7979914019, 0.7919686232, 0.8021790513, 0.7933033231, 0.8002426514, 0.805603067, 0.8151166686, 0.8022960094, 
		0.7882405741, 0.783514678, 0.7842957509, 0.7887602237, 0.7852950932, 0.810667561, 0.8012095646, 0.7990404887, 0.7842903884, 0.8099307768, 0.8162683819, 0.8059230235, 0.8140190379, 
		0.8015234619, 0.8089212202, 0.8005409752, 0.8016435215, 0.7656329138, 0.8005257059, 0.7721599254, 0.7810868756, 0.7833084248, 0.8001031473, 0.7670729089, 0.7678047903, 0.2150648362, 
		0.7914673006, 0.7900339536, 0.7783710314, 0.6768061441, 0.7660285322, 0.7600808044, 0.8128348264, 0.7375571938, 0.7500567222, 0.7981538628, 0.7892844491, 0.7955666485, 0.7989342092, 
		0.8013476286, 0.7938243378, 0.7980892856, 0.7910447397, 0.7752881127, 0.7736976215, 0.7746056438, 0.01514665768, 0.8077583539, 0.7990511139, 0.7938243378, 0.8001031473
	]
	x = np.arange(1, 181)
	colors = sns.color_palette("pastel")
	median = np.median(y)
	mean = np.mean(y)
	max = np.amax(y)
	min = np.amin(y)

	fig, ax = plt.subplots(ncols=1, figsize=COLUMN_FIGSIZE)
	ax.set_ylim(0.0, 1.0)
	ax.set_xlim(1, 180)
	#ax.set_yticks([0.0, 0.5, 1.0])

	ax.errorbar(
		x,
		y,
		color=colors[0],
		#linestyle=LINE_STYLES[ls],
		linewidth=2,
		capsize=3,
		capthick=1.5,
		ecolor="black",
	)	

	ax.set_ylabel("Fidelity")
	#ax.set_title("", fontweight="bold", fontsize=FONTSIZE)
	ax.set_xlabel("Calibration Day")

	ax.axhline(median, color=colors[3], linestyle="--")
	#ax.axhline(mean, color=colors[2], linestyle="--")
	#ax.axhline(min, color="dimgrey", linestyle="--")
	#ax.axhline(max, color="dimgrey", linestyle="--")
	fig.text(0.5, 1, HIGHERISBETTER, ha="center", va="center", fontweight="bold", color="navy", fontsize=14)

	output_file = "figures/scale_sim/temporal_results.pdf"
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout(pad=1)
	plt.savefig(output_file, bbox_inches="tight")



def plot_dep_min_stats() -> plt.Figure:
    DEP_MIN_DATA_3 = {
        "BV": "bench/results/greedy_dep_min/3/bv.csv",
        "VQE-1": "bench/results/greedy_dep_min/3/vqe_1.csv",
        "HS-2": "bench/results/greedy_dep_min/3/hamsim_2.csv",
        "TL-1": "bench/results/greedy_dep_min/3/twolocal_1.csv",
        "TL-2": "bench/results/greedy_dep_min/3/twolocal_2.csv",
        "TL-3": "bench/results/greedy_dep_min/3/twolocal_3.csv",
        "QAOA-B": "bench/results/greedy_dep_min/3/qaoa_b.csv",
        "QAOA-3": "bench/results/greedy_dep_min/3/qaoa_r3.csv",
        "QAOA-4": "bench/results/greedy_dep_min/3/qaoa_r4.csv",
    }

    dfs = [pd.read_csv(file) for file in DEP_MIN_DATA_3.values()]
    labels = list(DEP_MIN_DATA_3.keys())

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=WIDE_FIGSIZE, sharey=True)
    xvalues = [8, 16, 24]

    ax0.set_ylim(0, 1.2)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "num_deps", "num_deps_base"
    )
    grouped_bar_plot(ax0, y.T, yerr.T, labels, show_average_text=True)
    ax0.set_ylabel("Rel. Qubit Dependencies")
    ax0.set_title("(a) Qubit Dependencies", fontweight="bold", fontsize=FONTSIZE)
    ax0.set_xlabel("Number of Qubits")
    ax0.set_xticklabels(xvalues)
    _relative_plot(ax0)

    y, yerr = data_frames_to_y_yerr(
        dfs, "num_qubits", np.array(xvalues), "depth", "depth_base"
    )
    grouped_bar_plot(ax1, y.T, yerr.T, labels, show_average_text=True)
    ax1.set_ylabel("Rel. Circuit Depth")
    ax1.set_title("(b) Circuit Depth", fontweight="bold", fontsize=FONTSIZE)
    ax1.set_xlabel("Number of Qubits")
    ax1.set_xticklabels(xvalues)
    _relative_plot(ax1)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=10,
        frameon=False,
    )

    fig.text(
        0.51,
        0.9,
        LOWERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )

#from util import calculate_figure_size, plot_lines, grouped_bar_plot, data_frames_to_y_yerr
#from data import SWAP_REDUCE_DATA, DEP_MIN_DATA, NOISE_SCALE_ALGIERS_DATA, SCALE_SIM_TIME, SCALE_SIM_MEMORY


sns.set_theme(style="whitegrid", color_codes=True)
colors = sns.color_palette("deep")

plt.rcParams.update({"font.size": 12})


def plot_swap_reduce() -> None:
    dfs = [pd.read_csv(file) for file in SWAP_REDUCE_DATA.values()]
    titles = list(SWAP_REDUCE_DATA.keys())

    plot_dataframes(
        dataframes=dfs,
        keys=["num_cnots", "num_cnots_base"],
        labels=["Ours", "Baseline"],
        titles=titles,
        ylabel="Number of CNOTs",
        xlabel="Number of Qubits",
        output_file="figures/swap_reduce/cnot.pdf",
    )
    plot_dataframes(
        dataframes=dfs,
        keys=["depth", "depth_base"],
        labels=["Ours", "Baseline"],
        titles=titles,
        ylabel="Circuit Depth",
        xlabel="Number of Qubits",
        output_file="figures/swap_reduce/depth.pdf",
    )

    fig.text(
        0.51,
        0.98,
        HIGHERISBETTER,
        ha="center",
        fontsize=ISBETTER_FONTSIZE,
        fontweight="bold",
        color="midnightblue",
    )

def insert_column(df):
    df['total_runtime'] = df['run_time'] + df['knit_time']

    return df

def dataframe_out_of_columns(dfs, lines, columns):
    merged_df = pd.DataFrame()

    merged_df["num_qubits"] = dfs[0]["num_qubits"].copy()
    merged_df.set_index("num_qubits")

    for i,f in enumerate(dfs):
        merged_df[lines[i]] = f[columns].copy()

    #merged_df.reset_index(drop = True, inplace = True)
    merged_df.set_index("num_qubits", inplace = True)

    return merged_df

def plot_endtoend_runtimes():
	dfs = [pd.read_csv(file) for file in SCALE_SIM_TIME.values()]
	dfs_mem = [pd.read_csv(file) for file in SCALE_SIM_MEMORY.values()]
	dfs_knit_time = [pd.read_csv(file) for file in SCALE_SIM_KNIT_TIME.values()]

	pivot_df = dfs_knit_time[0].pivot_table(index='num_threads', columns='num_vgates', values='time', aggfunc='mean')
	pivot_df.columns = ['1 vgate', '2 vgates', '3 vgates', '4 vgates']
	pivot_df = pivot_df.rename_axis('num_qubits')
	print(pivot_df)
	#print(pivot_df.keys())
 
	lines = [s.split("-")[-1] for s in SCALE_SIM_TIME.keys()]

	titles = ["(a) Εnd-to-end Runtime", "(b) Runtime Breakdown", "(c) Knitting Scaling", "(d) Memory Consumption"]

	dfs = [insert_column(i) for i in dfs]
	big_dfs = dataframe_out_of_columns(dfs, lines, ["total_runtime"])
	
	dfs_mem_new = pd.DataFrame()
	dfs_mem_new["num_qubits"] = dfs_mem[0]["num_qubits"].copy()
	dfs_mem_new["Baseline"] = dfs_mem[0]["h_fid"]
	dfs_mem_new["QVM"] = dfs_mem[0]["h_fid_base"]
	dfs_mem_new["CutQC"] = dfs_mem[0]["tv_fid"]
	dfs_mem_new.set_index("num_qubits", inplace = True)
	#print(dfs_mem_new)
	
	dfs_ratio = pd.DataFrame()
	dfs_ratio["qpu_size"] = [15, 20, 25]
	dfs_ratio.set_index("qpu_size")
	
	dfs_ratio["simulation"] = [d.loc[4].at['run_time'] for d in dfs]
	dfs_ratio["knitting"] = [d.loc[4].at['knit_time'] for d in dfs]
	
	keys = dfs_ratio.keys()
	keys = keys[1:]

	custom_plot_dataframes(
		dataframes=[big_dfs, dfs_ratio, pivot_df, dfs_mem_new],
		keys=[big_dfs.keys(), keys, pivot_df.keys(), dfs_mem_new.keys()],
		labels=[big_dfs.keys(), dfs_ratio["qpu_size"].tolist(), pivot_df.keys(), dfs_mem_new.keys()],
		titles=titles,
		ylabel=["Runtime [s]", "Runtime [s]", "Knitting Time [s]", "Memory [GBs]"],
		xlabel=["Number of Qubits", "QPU Size [Number of Qubits]", "Number of Threads", "Number of Qubits"],
		output_file="figures/scale_sim/hamsim_1.pdf",
		logscale=True,
		nrows=1
	)	

hatches = [
    "/",
	"\\",
	"//",
	"\\\\",
	"x",
	".",
	",",
	"*",
	"o",
	"O",
	"+",
	"X",
	"s",
	"S",
	"d",
	"D",
	"^",
	"v",
	"<",
	">",
	"p",
	"P",
	"$",
	"#",
	"%",
]

def custom_plot_dataframes(
	dataframes: list[pd.DataFrame],
	keys: list[list[str]],
	labels: list[list[str]],
	titles: list[str],
	ylabel: list[str],
	xlabel: list[str],
	output_file: str = "noisy_scale.pdf",
	nrows: int = 2,
	logscale = False,
) -> None:
	ncols = len(dataframes)
	fig = plt.figure(figsize=WIDE_FIGSIZE)
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
	
	axis[0].set_yscale("log")
	axis[1].set_yscale("log")
	axis[2].set_yscale("log")
	axis[2].set_xscale("log")
	axis[3].set_yscale("log")

	#axis[2].set_xlim([10, 30])
	axis[1].set_ylim([1, 50000])
	#axis[2].set_yticks([0, 1, 10, 100, 1000], [0, 1, 10, 100, 1000])
	axis[2].set_ylim([3*10**(-2), 10**3])
	#axis[2].set_yscale("log")

	for i, ax in enumerate(axis):
		ax.set_ylabel(ylabel=ylabel[i])
		ax.set_xlabel(xlabel=xlabel[i])
	
	#print(keys)
	plot_lines(axis[0], keys[0], labels[0], [dataframes[0]])
	axis[0].legend()		
	axis[0].set_title(titles[0], fontsize=12, fontweight="bold")

	print(keys[2])
	plot_lines(axis[2], keys[2], labels[2], [dataframes[2]])
	axis[2].legend()		
	axis[2].set_title(titles[2], fontsize=12, fontweight="bold")
	
	plot_lines(axis[3], keys[3], labels[3], [dataframes[3]])
	axis[3].legend()		
	axis[3].set_title(titles[3], fontsize=12, fontweight="bold")

	num_vgates = dataframes[1]['qpu_size'].tolist()
	simulation = dataframes[1]['simulation'].tolist()
	knitting = dataframes[1]['knitting'].tolist()
	data = {
		"Simulation" : simulation,
		"Knitting" : knitting,
	}

	x = np.array([15, 20, 25])
	#x = np.arange(len(num_vgates))  # the label locations
	#width = 0.25  # the width of the bars
	#multiplier = 0
	y = np.array(
		[
			[9.52130384114571, 120.0079321230296, 801.0942367650568],
			[11.77336971112527, 726.3718322570203, 208.40429024997866],
			[1.7376548638567328, 5857.7779290829785, 305.2052580610034]
		]
	)

	yerr = np.array(
		[
			[1.3718322570203, 6.270605635945685, 41.68920839508064],
			[2.7376548638567328, 33.503638901049, 8.03563788096653],
			[0.2052580610034, 155.2813523421064, 22.93891781999264]
		]
	)
	
	axis[1].set_xticklabels(x)
	axis[1].grid(axis="y", linestyle="-", zorder=-1)	
	grouped_bar_plot(axis[1], y, yerr, ["Compilation", "Simulation", "Knitting"])
	axis[1].set(ylabel=None)
	axis[1].legend(loc="upper left", ncols=2)

	axis[1].set_yticks(np.logspace(1, 5, base=10, num=5, dtype='int'))
	axis[1].set_title(titles[1], fontsize=12, fontweight="bold")
	
	fig.text(0.51, 1.17, LOWERISBETTER, ha="center", fontweight="heavy", color="midnightblue", fontsize=ISBETTER_FONTSIZE)
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout(pad=-2.3)
	plt.savefig(output_file, bbox_inches="tight")

def plot_dep_min() -> None:
	dfs = [pd.read_csv(file) for file in DEP_MIN_DATA.values()]
	titles = list(DEP_MIN_DATA.keys())

	plot_dataframes(
		dataframes=dfs,
		keys=["num_cnots", "num_cnots_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Number of CNOTs",
		xlabel="Number of Qubits",
		output_file="figures/dep_min/cnot.pdf",
	)
	plot_dataframes(
		dataframes=dfs,
		keys=["depth", "depth_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Circuit Depth",
		xlabel="Number of Qubits",
		output_file="figures/dep_min/depth.pdf",
	)

	plot_dataframes(
		dataframes=dfs,
		keys=["h_fid", "h_fid_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Fidelity",
		xlabel="Number of Qubits",
		output_file="figures/dep_min/fid.pdf",
	)


def plot_noisy_scale() -> None:
	dfs = [pd.read_csv(file) for file in NOISE_SCALE_ALGIERS_DATA.values()]
	titles = list(NOISE_SCALE_ALGIERS_DATA.keys())

	plot_dataframes(
		dataframes=dfs,
		keys=["num_cnots", "num_cnots_base"],
		labels=["Dep Min", "Baseline"],
		titles=titles,
		ylabel="Number of CNOTs",
		xlabel="Number of Qubits",
		output_file="figures/noisy_scale/algiers_cnot.pdf",
		nrows=3,
	)
	plot_dataframes(
		dataframes=dfs,
		keys=["depth", "depth_base"],
		labels=["Dep Min", "Baseline"],
		titles=titles,
		ylabel="Circuit Depth",
		xlabel="Number of Qubits",
		output_file="figures/noisy_scale/algiers_depth.pdf",
		nrows=3,
	)
	plot_dataframes(
		dataframes=dfs,
		keys=["h_fid", "h_fid_base"],
		labels=["Ours", "Baseline"],
		titles=titles,
		ylabel="Fidelity",
		xlabel="Number of Qubits",
		output_file="figures/noisy_scale/algiers_fid.pdf",
		nrows=3,
	)

def plot_dataframes(
	dataframes: list[pd.DataFrame],
	keys: list[str],
	labels: list[str],
	titles: list[str],
	ylabel: str,
	xlabel: str,
	output_file: str = "noisy_scale.pdf",
	nrows: int = 2,
	logscale = False,
) -> None:
	ncols = len(dataframes) // nrows + len(dataframes) % nrows
	# plotting the absolute fidelities
	fig = plt.figure(figsize=calculate_figure_size(nrows, ncols))
	gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

	axis = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

	for i, ax in enumerate(axis):
		if logscale:
			ax.set_yscale("log")
		if i % ncols == 0:
			ax.set_ylabel(ylabel=ylabel)
		if i >= len(axis) - ncols:
			ax.set_xlabel(xlabel=xlabel)

	for let, title, ax, df in zip(string.ascii_lowercase, titles, axis, dataframes):
		plot_lines(ax, keys, labels, [df])
		ax.legend()
		ax.set_title(f"({let}) {title}", fontsize=12, fontweight="bold")

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_file, bbox_inches="tight")


def plot_relative(
	ax,
	dataframes: list[pd.DataFrame],
	num_key: str,
	denom_key: str,
	labels: list[str],
	ylabel: str,
	xlabel: str,
	title: str | None = None,
):
	for df in dataframes:
		df["relative"] = df[num_key] / df[denom_key]

	plot_lines(ax, ["relative"], labels, dataframes)
	ax.set_ylabel(ylabel=ylabel)
	ax.set_xlabel(xlabel=xlabel)

	# line at 1
	ax.axhline(y=1, color="black", linestyle="--")
	ax.set_title(title)
	ax.legend()


def plot_relative_swap_reduce() -> None:
	DATAFRAMES = [pd.read_csv(file) for file in SWAP_REDUCE_DATA.values()]
	LABELS = list(SWAP_REDUCE_DATA.keys())

	fig, ax = plt.subplots(1, 1, figsize=calculate_figure_size(1, 1))
	plot_relative(
		ax,
		DATAFRAMES,
		"num_cnots",
		"num_cnots_base",
		labels=LABELS,
		ylabel="Reulative Number of CNOTs",
		xlabel="Number of Qubits",
	)

	output_file = "figures/swap_reduce/cnot_relative.pdf"

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_file, bbox_inches="tight")



def main():
	#print("Plotting swap reduce...")
	#plot_swap_reduce()
	#print("Plotting dep min...")
	#plot_dep_min()
	#print("Plotting noisy scale...")
	#plot_noisy_scale()
	# plot_relative_swap_reduce()
	#plot_fidelities()
	#plot_cutting()
	#plot_utilization()
	#plot_spatial_variance()
	#plot_temporal_variance()
	plot_endtoend_runtimes()


if __name__ == "__main__":
	main()
