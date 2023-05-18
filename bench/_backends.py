from qiskit.providers import BackendV2
from qiskit.providers.fake_provider import FakeMumbaiV2, FakeSherbrooke, FakeKolkataV2
from qiskit_aer import AerSimulator

BACKENDS = {
    "kolkata": FakeKolkataV2,
    "mumbai": FakeMumbaiV2,
    "simulator": AerSimulator,
    "sherbrooke": FakeSherbrooke,
}


def get_backend(name: str) -> BackendV2:
    return BACKENDS[name]()
