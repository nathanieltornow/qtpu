"""Clifford-gate analogue of the mqt.bench QNN circuit.

Walks the structure of the qnn benchmark and replaces every single-qubit
gate with a fixed random Clifford 1q gate, leaving entangling gates
(cx/cz) untouched. The result is structurally the same as qnn (same
depth, same entanglement pattern, same qubit count) but now efficiently
classically simulable via the stabilizer formalism.

For hardware fidelity experiments we wrap it with its own inverse under
one barrier:

    full = C · barrier · C†

Applied to |0...0⟩ the ideal output is |0...0⟩ → ⟨Z^⊗n⟩_ideal = 1.
This provides a strong non-trivial signal that noise drives toward 0 —
unlike a random QNN where ⟨Z^n⟩ ≈ 0 both noiseless and noisy, so noise
effects are invisible.

The single barrier at the mirror boundary is sufficient: it prevents
the transpiler from collapsing C·C† = I across the wall, while still
allowing normal gate-level optimization within each side (matching how
a real user would compile the circuit).
"""

from __future__ import annotations

import random

from qiskit.circuit import QuantumCircuit

# Single-qubit Clifford gate set. Cardinality ≈ 24 for the full Clifford
# group on 1 qubit; we use a generating subset that's easy to apply.
CLIFFORD_1Q = ["id", "h", "s", "sdg", "x", "y", "z", "hs", "sh", "shs"]


def _apply_1q_clifford(qc: QuantumCircuit, name: str, q) -> None:
    """Apply a named 1q Clifford gate to qubit q in-place."""
    if name == "id":
        return
    if name == "hs":
        qc.h(q)
        qc.s(q)
        return
    if name == "sh":
        qc.s(q)
        qc.h(q)
        return
    if name == "shs":
        qc.s(q)
        qc.h(q)
        qc.s(q)
        return
    getattr(qc, name)(q)


def cliffordize(qc: QuantumCircuit, seed: int = 42) -> QuantumCircuit:
    """Replace every 1-qubit gate in qc with a fixed random Clifford gate,
    and insert a full-width barrier after every gate.

    The barriers are essential: at opt_level=3 the transpiler will
    consolidate chains of single-qubit Cliffords into a single U3 gate
    (and occasionally notice local cancellations like H·H = I), which
    would collapse the intended circuit depth on hardware. Barriers
    after every gate preserve the logical depth end-to-end; the
    transpiler still handles layout + routing normally.

    Non-1q gates (cx, cz, swap, ...) and structural ops (measure, reset)
    are kept as-is. The same seed produces the same replacement pattern.
    """
    rng = random.Random(seed)
    n_qubits = sum(qr.size for qr in qc.qregs)
    new = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instr in qc.data:
        op = instr.operation
        name = op.name
        if name in {"barrier", "measure", "reset"}:
            new.append(op, instr.qubits, instr.clbits)
            continue
        if op.num_qubits == 1:
            _apply_1q_clifford(new, rng.choice(CLIFFORD_1Q), instr.qubits[0])
        else:
            new.append(op, instr.qubits, instr.clbits)
        new.barrier(range(n_qubits))
    return new


def build_clifford_qnn_mirror(n: int, seed: int = 42) -> QuantumCircuit:
    """Clifford-QNN mirror circuit: C · barrier · C†, applied to |0...0⟩.

    Ideal output: |0...0⟩ ⇒ ⟨Z^⊗n⟩_ideal = +1.
    """
    from mqt.bench import get_benchmark_indep

    base = get_benchmark_indep("qnn", circuit_size=n, opt_level=3)
    base = base.remove_final_measurements(inplace=False)

    c = cliffordize(base, seed=seed)

    full = QuantumCircuit(*c.qregs)
    full.compose(c, inplace=True)
    full.barrier(range(n))
    full.compose(c.inverse(), inplace=True)
    return full


def build_clifford_qnn_conjugated(n: int, seed: int = 42) -> QuantumCircuit:
    """Clifford-QNN with *no* inverse layer, observable absorbed as 1q rotations.

    Instead of running C · C† and measuring Z^⊗n (the mirror trick, which
    doubles the 2q-gate count), we run just C and measure the
    Clifford-conjugated Pauli observable P = C · Z^⊗n · C†, which is
    still a Pauli string since C is Clifford. Measuring P reduces to
    per-qubit basis rotations (H for X, S†·H for Y, nothing for Z) before
    the standard computational-basis parity. Ideal ⟨P⟩ on state C|0⟩ is

        ⟨0|C† · (C Z^⊗n C†) · C|0⟩ = ⟨0|Z^⊗n|0⟩ = +1.

    For qubits where P_k = I (no observable contribution) we append a
    `reset` — the codegen / IBM backend treat a terminal reset as "trace
    this qubit out of the observable" (effective +1 contribution).

    Returns a single QuantumCircuit with the per-qubit basis rotations,
    terminal resets for I-qubits, and (if P has a −1 global phase) an X
    flip on one measured qubit — all appended after a barrier following
    the main C block.
    """
    from mqt.bench import get_benchmark_indep
    from qiskit.quantum_info import Clifford, Pauli

    base = get_benchmark_indep("qnn", circuit_size=n, opt_level=3)
    base = base.remove_final_measurements(inplace=False)

    c = cliffordize(base, seed=seed)
    P_out = Pauli("Z" * n).evolve(Clifford(c), frame="s")

    phase = int(P_out.phase) % 4
    if phase % 2 != 0:
        raise RuntimeError(f"Non-Hermitian conjugated Pauli (phase={phase}); bug?")
    sign = +1 if phase == 0 else -1

    full = QuantumCircuit(*c.qregs)
    full.compose(c, inplace=True)
    full.barrier(range(n))

    first_measured = None
    for k in range(n):
        z = bool(P_out.z[k])
        x = bool(P_out.x[k])
        if not z and not x:  # I: trace this qubit out via terminal reset
            full.reset(k)
            continue
        if first_measured is None:
            first_measured = k
        if z and not x:  # Z: no rotation
            pass
        elif not z and x:  # X: H maps X eigenbasis to Z
            full.h(k)
        else:  # Y: S† then H maps Y eigenbasis to Z
            full.sdg(k)
            full.h(k)

    if sign == -1:
        # Flip exactly one measured qubit's bit to flip overall parity.
        if first_measured is None:
            raise RuntimeError("All qubits are I — degenerate observable.")
        full.x(first_measured)

    return full


if __name__ == "__main__":
    # Sanity check: ideal ⟨observable⟩ should be +1 for both constructions.
    from qiskit.quantum_info import Statevector

    print("--- mirror (C · C†, measure Z^⊗n) ---")
    for n in [4, 6, 8]:
        qc = build_clifford_qnn_mirror(n, seed=42)
        probs = Statevector.from_instruction(qc).probabilities()
        z_n = sum(((-1) ** bin(x).count("1")) * p for x, p in enumerate(probs))
        print(f"  n={n}  ⟨Z^⊗n⟩={z_n:+.6f}  depth={qc.depth()}")

    print("\n--- conjugated (C, measure P = C Z^⊗n C†) ---")
    for n in [4, 6, 8, 20]:
        qc = build_clifford_qnn_conjugated(n, seed=42)
        # For statevector sanity check, strip the reset ops (terminal resets mean
        # "I observable" for those qubits — which in the parity sum means the
        # qubit's bit should contribute +1 regardless of its measurement outcome;
        # we emulate that by masking the I-qubit bits out of the parity computation).
        from qiskit.quantum_info import Clifford, Pauli
        c_only = qc.copy()
        # Find I-qubits via reset presence in the circuit data
        i_qubits = {
            qc.qubits.index(instr.qubits[0])
            for instr in qc.data if instr.operation.name.lower() == "reset"
        }
        # Drop resets so Statevector.from_instruction works (it rejects mid-circuit resets)
        c_only.data = [i for i in c_only.data if i.operation.name.lower() != "reset"]
        probs = Statevector.from_instruction(c_only).probabilities()
        n_tot = qc.num_qubits
        mask = sum(1 << k for k in range(n_tot) if k not in i_qubits)
        z_n = sum(((-1) ** bin(x & mask).count("1")) * p for x, p in enumerate(probs))
        print(
            f"  n={n}  ⟨P⟩={z_n:+.6f}  depth={qc.depth()}  "
            f"I-qubits={len(i_qubits)}"
        )
