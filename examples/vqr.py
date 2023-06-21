from qiskit.primitives import BackendSampler
from qiskit.providers.fake_provider import FakeOslo
from qiskit.quantum_info import hellinger_fidelity
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

from qvm.compiler import vqr
from qvm.run import run_vgate_circuit_as_one

from _example_circuit import example_circuit


def main():
    circuit = example_circuit(6, 1, "circular")

    backend = FakeOslo()
    sampler = BackendSampler(backend)

    coupling_map = backend.coupling_map
    initial_layout = [0, 1, 3, 4, 5, 6]

    vqr_circuit = vqr(
        circuit, coupling_map, initial_layout, max_vgates=2, technique="perfect"
    )
    result, times = run_vgate_circuit_as_one(
        vqr_circuit,
        sampler,
        transpile_args={
            "backend": backend,
            "initial_layout": initial_layout,
            "optimization_level": 3,
        },
        run_args={"shots": 10000, "resilience_level": 0},
    )

    t_circuit = transpile(
        circuit, backend=backend, optimization_level=3, initial_layout=initial_layout
    )
    noisy_res = (
        sampler.run(t_circuit, shots=10000, resilience_level=0).result().quasi_dists[0]
    )
    noisy_res = {bin(k)[2:].zfill(6): v for k, v in noisy_res.items()}

    actual_result = AerSimulator().run(circuit, shots=10000).result().get_counts()
    distr = {k: v / 10000 for k, v in actual_result.items()}

    print("Hellinger fidelity:", hellinger_fidelity(distr, result))
    print("Noisy Hellinger fidelity:", hellinger_fidelity(distr, noisy_res))


if __name__ == "__main__":
    main()
