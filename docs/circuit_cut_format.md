# Conventions for Representing Cuts in Quantum Circuits

In our quantum circuit framework, **cuts** are represented using QASM 3.0 syntax. We use **barriers** to clearly indicate cuts, either for **wires** or **gates**.

### Wire Cuts
To represent a wire cut, place **two adjacent barriers** after the desired operation. Here’s an example:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
bit[2] meas;
qubit[2] q;

h q[0];
// Wire cut on qubit q[0] after the H gate
barrier q[0];
barrier q[0];

cx q[0], q[1];
meas[0] = measure q[0];
meas[1] = measure q[1];
```

### Gate Cuts
To represent a gate cut, **surround** the operation with **barriers** on the involved qubits:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
bit[2] meas;
qubit[2] q;

h q[0];

// Gate cut on the CX gate between q[0] and q[1]
barrier q[0], q[1];
cx q[0], q[1];
barrier q[0], q[1];

meas[0] = measure q[0];
meas[1] = measure q[1];
```