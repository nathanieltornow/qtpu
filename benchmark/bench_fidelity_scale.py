from circuits import ghz, qaoa, twolocal
from fidelity import calc_fidelity
from graphs import barbell
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import (FakeMontrealV2, FakeOslo,
                                            FakeSherbrooke)
from qiskit.providers.ibmq import IBMQ, AccountProvider
from qiskit_aer.noise import NoiseModel
from util import append_to_csv_file

from qvm.stack._types import QVMJobMetadata, QVMLayer
from qvm.stack.decomposer import LadderDecomposer, QPUAwareDecomposer
from qvm.stack.qpu_runner import QPURunner
from qvm.stack.qpus.ibmq_fake import IBMQFakeQPU
from qvm.stack.qpus.ibmq_qpu import IBMQQPU
from qvm.stack.qpus.simulator import SimulatorQPU

SHOTS = 20000


def ladder_stack(provider: AccountProvider, num_qubits: int):
    qpu = IBMQQPU(provider, "ibm_oslo")
    # qpu2 = IBMQFakeQPU(provider, "ibm_oslo")
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = LadderDecomposer(qpu_runner, num_qubits)
    return stack


def bisection_stack(provider: AccountProvider, num_qubits: int):
    qpu = IBMQQPU(provider, "ibm_oslo")
    # qpu2 = IBMQFakeQPU(provider, "ibm_oslo")
    qpu_runner = QPURunner(qpus={"sim": qpu})
    stack = QPUAwareDecomposer(qpu_runner, num_qubits)
    return stack


def run_bench(
    qasms: list[str],
    stack: QVMLayer,
    result_file: str,
    provider: AccountProvider,
):
    circuits = [QuantumCircuit.from_qasm_file(qasm) for qasm in qasms]
    for circ in circuits:
        job_id = stack.run(circ, [], metadata=QVMJobMetadata(shots=SHOTS))
        res = stack.get_results(job_id)[0]
        counts = res.to_counts(SHOTS)
        fid = calc_fidelity(circ, counts, provider)
        append_to_csv_file(
            result_file,
            {
                "num_qubits": len(circ.qubits),
                "fidelity": fid,
            },
        )


def bench_ghz(provider: AccountProvider):
    csv_name = f"results/new_runs/ghz.csv"

    # ghz_qasms = [f"qasm/ghz/{n}.qasm" for n in range(14, 17, 2)] * 5
    # stack = ladder_stack(provider, 4)
    # run_bench(ghz_qasms, stack, csv_name, provider)

    ghz_qasms = [f"qasm/ghz/{n}.qasm" for n in range(2, 11, 2)] * 1
    stack = ladder_stack(provider, 3)
    run_bench(ghz_qasms, stack, csv_name, provider)



def bench_two_local_1_rep(provider: AccountProvider):
    csv_name = f"results/rew_runs/twolocal_1-reps.csv"

        
    ghz_qasms = [f"qasm/twolocal/1_{n}.qasm" for n in range(12, 13, 2)]
    stack = ladder_stack(provider, 4)
    run_bench(ghz_qasms, stack, csv_name, provider)


    ghz_qasms = [f"qasm/twolocal/1_{n}.qasm" for n in range(14, 17, 2)]
    stack = ladder_stack(provider, 4)
    run_bench(ghz_qasms, stack, csv_name, provider)


def bench_twolocal_2_reps(provider: AccountProvider):
    csv_name = f"results/new_runs/twolocal_2-reps.csv"
    
    ghz_qasms = [f"qasm/twolocal/2_{n}.qasm" for n in range(2, 9, 2)]
    stack = ladder_stack(provider, 4)
    run_bench(ghz_qasms, stack, csv_name, provider)

    ghz_qasms = [f"qasm/twolocal/2_{n}.qasm" for n in range(10, 11, 2)]
    stack = ladder_stack(provider, 5)
    run_bench(ghz_qasms, stack, csv_name, provider)

    ghz_qasms = [f"qasm/twolocal/2_{n}.qasm" for n in range(14, 15, 2)]
    stack = ladder_stack(provider, 7)
    run_bench(ghz_qasms, stack, csv_name, provider)

    # ghz_qasms = [f"qasm/twolocal/2_{n}.qasm" for n in range(2, 10, 2)]
    # stack = ladder_stack(provider, 3)
    # run_bench(ghz_qasms, stack, csv_name, provider)
    

def bench_qaoa_l(provider: AccountProvider):
    csv_name = f"results/new_runs/qaoa_l.csv"

    qaoa_qasms = [f"qasm/qaoa/l{n}.qasm" for n in range(1, 5)]
    stack = bisection_stack(provider, 4)
    run_bench(qaoa_qasms, stack, csv_name, provider)
    
    qaoa_qasms = [f"qasm/qaoa/l{n}.qasm" for n in range(6, 7)]
    stack = bisection_stack(provider, 5)
    run_bench(qaoa_qasms, stack, csv_name, provider)
    
    
    

def bench_qaoa_b(provider: AccountProvider):
    csv_name = f"results/new_runs/qaoa_b.csv"

    # qaoa_qasms = [f"qasm/qaoa/b{n}.qasm" for n in range(4, 5)]
    # stack = bisection_stack(provider, 4)
    # run_bench(qaoa_qasms, stack, csv_name, provider)
    
    qaoa_qasms = [f"qasm/qaoa/b{n}.qasm" for n in range(5, 6)]
    stack = ladder_stack(provider, 5)
    run_bench(qaoa_qasms, stack, csv_name, provider)
    
    # qaoa_qasms = [f"qasm/qaoa/b{n}.qasm" for n in range(1, 4)]
    # stack = bisection_stack(provider, 4)
    # run_bench(qaoa_qasms, stack, csv_name, provider)
    


if __name__ == "__main__":
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-research-2", group="code-institute-1", project="main")
    
    import sys
    
    bench = sys.argv[1]
    
    if bench == "ghz":
        bench_ghz(provider)
    elif bench == "twolocal1":
        bench_two_local_1_rep(provider)
    elif bench == "twolocal2":
        bench_twolocal_2_reps(provider)
    elif bench == "qaoal":
        bench_qaoa_l(provider)
    elif bench == "qaoab":
        bench_qaoa_b(provider)
        