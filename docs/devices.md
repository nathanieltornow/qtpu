# Devices

VQC uses devices to execute the fragments of dissected circuits. A device can be anything that can execute quantum circuits, and is **not** restricted to qiskit backends.

## Implementing a Device

Every device inherits from `vqc.Device`, and there for has to implement the `run()` method, which executes a list of quantum circuits each with a number of shots, and returns probability distributions for each of the circuits.

```python
class SomeDevice(Device):
    def run(circuits: List[QuantumCircuit], shots: int) -> List[ProbDistribution]:
        # ...
```

The `vqc.prob.ProbDistribution` object is used to model the probability of each measurement state, mapping integers to floats. A `ProbDistribution` can be created from counts using the `from_counts()` method.

## Mapping Fragments to Devices

Given a `qvm.circuit.DistributedCircuit` (the result of using `cut()` on a quantum circuit), we can map each fragment to a device by using the `set_fragment_device()` method. When executing the circuit, the fragments will be executed on the respective devices. E.g., when we want every fragment to be executed on `SomeDevice`, we'd run:

```python
dist_circ = cut(circuit, ...)
for frag in dist_circ.fragments:
    dist_circ.set_fragment_device(frag, SomeDevice())
```


## [`SimDevice`](../vqc/device/sim.py)

The `vqc.device.SimDevice` is a simple device that uses qiskit's `AerSimulator` to classically simulate the quantum circuits. It is the device every fragment is mapped by default.