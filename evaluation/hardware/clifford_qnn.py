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
    and insert a per-qubit barrier (on the gate's own qubits) after every gate.

    The barriers are essential: at opt_level=3 the transpiler will
    consolidate chains of single-qubit Cliffords into a single U3 gate
    (and occasionally notice local cancellations like H·H = I), which
    would collapse the intended circuit depth on hardware. A per-qubit
    barrier after each gate prevents this local consolidation *on that
    qubit* without blocking cross-qubit parallelism, so the transpiler
    can still schedule independent gates in the same layer and the
    circuit depth stays close to the original structure.

    Non-1q gates (cx, cz, swap, ...) and structural ops (measure, reset)
    are kept as-is. The same seed produces the same replacement pattern.
    """
    rng = random.Random(seed)
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
        new.barrier(instr.qubits)
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


def build_clifford_qnn_conjugated(
    n: int, seed: int = 42, observed: int | None = None
) -> QuantumCircuit:
    """Clifford-QNN with *no* inverse layer, observable absorbed as 1q rotations.

    Instead of running C · C† and measuring Z^⊗n (the mirror trick, which
    doubles the 2q-gate count), we run just C and measure the
    Clifford-conjugated Pauli observable P = C · P_base · C†, which is
    still a Pauli string since C is Clifford. Measuring P reduces to
    per-qubit basis rotations (H for X, S†·H for Y, nothing for Z) before
    the standard computational-basis parity. Ideal ⟨P⟩ on state C|0⟩ is

        ⟨0|C† · (C P_base C†) · C|0⟩ = ⟨0|P_base|0⟩ = +1.

    `observed` selects P_base:
      - None (default): weight-n observable P_base = Z^⊗n. Signal decays
        as p^n under depolarizing-like noise — the parity-observable curse.
      - int k in [0, n): weight-1 observable P_base = Z on qubit k, I on
        the rest. Signal only depends on qubit k's reduced state, so decays
        much more slowly and makes cutting's locality advantage visible.

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

    if observed is None:
        P_base = Pauli("Z" * n)
    else:
        if not (0 <= observed < n):
            raise ValueError(f"observed={observed} out of range [0, {n})")
        # Pauli(str) is MSB-first: leftmost char = highest-index qubit.
        chars = ["I"] * n
        chars[n - 1 - observed] = "Z"
        P_base = Pauli("".join(chars))

    P_out = P_base.evolve(Clifford(c), frame="s")

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


def build_spreading_conjugated(
    n: int, seed: int = 42, observed: int = 0
) -> QuantumCircuit:
    """Spreading benchmark for idle-gate noise experiments.

    Base circuit C:
      layer 0:           H on every qubit
      layer k=1..n-1:    CX(k-1, k); explicit `id` on every idle qubit

    We then measure the Clifford-conjugated observable P_out = C · Z_{observed} · C†
    on state C|0...0⟩. Ideal ⟨P_out⟩ = +1 by the same trick as
    `build_clifford_qnn_conjugated`.

    Why this circuit for the fidelity study:
      1. The explicit `id` gates make **idle decoherence** a first-class noise
         channel. A monolithic run has depth ~n and keeps each qubit idle for
         ~n-2 layers → O(n²) idle noise events. A cut fragment of width W has
         O(W²) idle events each → O(n·W) total across K=n/W fragments. With
         a depolarizing `id` error the cut retention beats the mono retention
         at large n.
      2. The circuit is **all Clifford** (H, CX, id) so it runs on AerSimulator's
         `stabilizer` method at n=60 in seconds.
      3. The forward CX chain spreads Z_0 under Heisenberg conjugation across
         every qubit (P_out is a weight-n X string), giving a non-degenerate
         observable. Using weight-1 on the *base* keeps the locality advantage
         of cutting visible.

    `seed` is accepted for signature compatibility with `build_clifford_qnn_*`
    but the circuit is deterministic.
    """
    from qiskit.quantum_info import Clifford, Pauli

    del seed  # deterministic

    if not (0 <= observed < n):
        raise ValueError(f"observed={observed} out of range [0, {n})")

    c = QuantumCircuit(n)
    # H on every qubit, then the forward CX chain. NO explicit id gates here:
    # idle-gate insertion must happen *after* cutting for the cut path (so each
    # fragment only carries its own local idle time) and *before* execution for
    # the mono path (so mono carries the full depth-n idle time). Baking idles
    # into the base circuit would leak mono's O(n²) idle budget into every cut
    # fragment, which collapses the expected retention advantage.
    for q in range(n):
        c.h(q)
    for k in range(n - 1):
        c.cx(k, k + 1)

    chars = ["I"] * n
    chars[n - 1 - observed] = "Z"  # Pauli(str) is MSB-first
    P_base = Pauli("".join(chars))

    P_out = P_base.evolve(Clifford(c), frame="s")

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
        if not z and not x:
            full.reset(k)
            continue
        if first_measured is None:
            first_measured = k
        if z and not x:
            pass
        elif not z and x:
            full.h(k)
        else:
            full.sdg(k)
            full.h(k)

    if sign == -1:
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

    print("\n--- conjugated weight-1, observed=0 (C, measure P = C Z_0 C†) ---")
    for n in [4, 6, 8, 20]:
        qc = build_clifford_qnn_conjugated(n, seed=42, observed=0)
        c_only = qc.copy()
        i_qubits = {
            qc.qubits.index(instr.qubits[0])
            for instr in qc.data if instr.operation.name.lower() == "reset"
        }
        c_only.data = [i for i in c_only.data if i.operation.name.lower() != "reset"]
        probs = Statevector.from_instruction(c_only).probabilities()
        n_tot = qc.num_qubits
        mask = sum(1 << k for k in range(n_tot) if k not in i_qubits)
        z_n = sum(((-1) ** bin(x & mask).count("1")) * p for x, p in enumerate(probs))
        print(
            f"  n={n}  ⟨Z_0⟩={z_n:+.6f}  depth={qc.depth()}  "
            f"support={n - len(i_qubits)}/{n}"
        )

    print("\n--- spreading (H^n + forward CX chain + idle gates, observed=0) ---")
    for n in [4, 8, 12, 16]:
        qc = build_spreading_conjugated(n, observed=0)
        # Strip idles and resets for Statevector sanity check (id is identity).
        c_only = qc.copy()
        i_qubits = {
            qc.qubits.index(instr.qubits[0])
            for instr in qc.data if instr.operation.name.lower() == "reset"
        }
        c_only.data = [
            i for i in c_only.data
            if i.operation.name.lower() not in ("reset",)
        ]
        probs = Statevector.from_instruction(c_only).probabilities()
        n_tot = qc.num_qubits
        mask = sum(1 << k for k in range(n_tot) if k not in i_qubits)
        z_n = sum(((-1) ** bin(x & mask).count("1")) * p for x, p in enumerate(probs))
        n_id = sum(1 for i in qc.data if i.operation.name.lower() == "id")
        n_cx = sum(1 for i in qc.data if i.operation.name.lower() == "cx")
        print(
            f"  n={n}  ⟨P⟩={z_n:+.6f}  depth={qc.depth()}  "
            f"id={n_id}  cx={n_cx}  support={n - len(i_qubits)}/{n}"
        )
