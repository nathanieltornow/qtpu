from qiskit.providers.fake_provider import (
    FakeBackendV2,
    FakeMumbaiV2,
    FakeHanoiV2,
    FakeCairoV2,
    FakeKolkataV2,
    FakeAuckland,
    FakeOslo,
)


BACKEND_TYPES = {
    "oslo": FakeOslo,
    "mumbai": FakeMumbaiV2,
    "hanoi": FakeHanoiV2,
    "cairo": FakeCairoV2,
    "kolkata": FakeKolkataV2,
    "auckland": FakeAuckland,
}


def get_backend(backend_name: str) -> FakeBackendV2:
    return BACKEND_TYPES[backend_name]()
