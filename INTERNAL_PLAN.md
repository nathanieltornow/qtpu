# Internal Revision Plan

## What the reviewers submitted figures show (the problem)

| Figure | Panels | What's shown | What's missing |
|--------|--------|-------------|----------------|
| Figs 9-10 (Compiler) | Pareto frontiers, compile time | Compiler quality | Fine as-is |
| Fig 11 (Runtime) | Runtime bars, speedup, vs cuTensorNet | Total runtime (no breakdown) | No task-level outcome |
| Fig 12 (Scale) | "Runtime", circuit count, FLOPs | Total runtime in (a) | No correctness validation |
| Fig 13 (Hybrid ML) | Compile time ×2, code size ×2 | **Only compilation** | Zero execution metrics |
| Fig 14 (Error Mit.) | Compile time, code size | **Only compilation** | Zero execution metrics |

## Experiments to implement

### 1. Combined workload (NEW — highest priority)

**Goal:** Show composability — knitting + PEC + batching in one hEinsum.

**Setup:**
- 20q and 30q QNN circuits
- Cut to 10q subcircuits (circuit knitting via `qtpu.cut()`)
- Add PEC ISwitches to subcircuit gates (error mitigation)
- Evaluate across batch of inputs (hybrid ML)
- Everything in one hEinsum

**Baseline:**
- QAC for cutting → Mitiq for PEC → batch loop for inputs
- 3 frameworks manually composed
- Circuit count: O(6^k × 4^m × batch_size) — explodes

**Files:**
- `evaluation/use_cases/combined/run.py` — benchmark script
- `evaluation/use_cases/combined/plot.py` — figure generation
- Uses benchkit decorators (`@bk.foreach`, `@bk.log`)

**Metrics to log:**
- `num_circuits_qtpu`, `num_circuits_baseline`
- `compile_time`, `qpu_time_estimated`, `classical_time`
- `reconstructed_expval`, `ideal_expval`, `error`
- `lines_of_code_qtpu`, `lines_of_code_baseline`

**Plot:** 
- (a) Circuit count: qTPU vs baseline (log scale, should be orders of magnitude)
- (b) E2E time breakdown: qTPU vs baseline (stacked bars)
- (c) Correctness: reconstructed vs ideal expectation values

**Implementation path:**
1. Build the composed hEinsum (circuit knitting + PEC ISwitches + batch ISwitches)
2. Build the baseline (QAC + Mitiq PEC generation + batch loop)
3. Simulate both on small circuits (Qiskit Aer statevector for ground truth)
4. Log and plot

### 2. Circuit knitting correctness (addition to Fig 12)

**Goal:** Prove the cutting pipeline produces correct results.

**Setup:**
- 20q, 30q QNN circuits, cut to 10q subcircuits
- Simulate subcircuits (CUDA-Q or Qiskit Aer)
- Reconstruct expectation value via classical contraction
- Compare to ideal statevector simulation

**Files:** Modify `evaluation/use_cases/scale/run.py` — add `run_qtpu_with_execution()` that actually runs simulation.

**Metric:** `|reconstructed - ideal|` with error bars across 3 seeds.

**Plot:** Add panel (d) to Fig 12 or a separate small figure.

### 3. Error mitigation correctness (addition to Fig 14)

**Goal:** Show PEC/ZNE actually improves accuracy when executed.

**Setup:**
- 10–15q circuit (simulable)
- Run PEC end-to-end: compile hEinsum → sample circuits → execute → reconstruct mitigated expectation value
- Compare: unmitigated vs. PEC-mitigated vs. ideal

**Files:** Modify `evaluation/use_cases/error_mitigation/run.py` — add `run_qtpu_mitigation_e2e()`.

**Metric:** `|mitigated - ideal|` vs `|unmitigated - ideal|`

**Plot:** Add panel (c) to Fig 14 showing accuracy improvement.

### 4. IBMBackend + hardware experiment

**Goal:** Condition 2 — run on real QPU.

**Files:**
- `src/qtpu/runtime/backends.py` — add `IBMBackend` class
- `evaluation/use_cases/scale/run_hardware.py` — hardware benchmark script

**IBMBackend interface:**
```python
class IBMBackend(QuantumBackend):
    def __init__(self, backend_name, shots=1000)
    def evaluate(self, qtensor, params, dtype, device) -> (result, eval_time, qpu_time)
```

**Hardware experiment:**
- Same circuits as experiment 2 (20q, 30q → 10q subcircuits)
- Submit to IBM QPU
- Report: correctness, estimated vs actual QPU time, Pareto frontier fidelity

**Fallback:** Qiskit Aer with noise model from real device calibration data.

### 5. Hybrid ML reframe (modify Fig 13)

**Goal:** Be honest, add transparency.

**What to change:**
- Keep existing compile time + code size panels (real advantages)
- Add E2E breakdown TABLE in paper text (not a misleading chart)
- Add discussion of memory scalability at large batch sizes
- Remove any claim about "recompiling every iteration"

**Files:** Modify `.paper/sections/evaluation.tex` — rewrite Section 8.6.

### 6. Paper text changes

| Change | Section | Priority |
|--------|---------|----------|
| LoC comparison table | Sec 5 | High |
| hTN as programming model + IR discussion | Sec 5 | High |
| Expanded TN background + worked example | Sec 3 | High |
| Compiler pseudocode | Sec 6 | Medium |
| Soften novelty claims + related work table | Sec 1 + 9 | Medium |
| Network/IO overhead discussion | Sec 7 | Medium |
| Dynamic noise model discussion | Sec 7 | Medium |
| Batch baseline justification (cite PennyLane etc.) | Sec 8 | Medium |
| NP-hard partitioning discussion | Sec 8 | Low |

## Execution order

```
Week 1: Experiments 1-3 (combined workload, correctness validations)
         Experiment 4 (IBMBackend implementation)
Week 2: Experiment 4 (submit hardware jobs, wait for results)
         Experiment 5 (hybrid ML reframe)
         Paper text changes (high priority)
Week 3: Paper text changes (medium priority)
         Integrate all results, regenerate figures
         Final pass, submit revision
```

## File inventory

```
NEW FILES:
  evaluation/use_cases/combined/run.py      — combined workload benchmark
  evaluation/use_cases/combined/plot.py     — combined workload figure
  evaluation/use_cases/scale/run_hardware.py — hardware benchmark
  src/qtpu/runtime/backends.py              — add IBMBackend class

MODIFY:
  evaluation/use_cases/scale/run.py         — add correctness validation
  evaluation/use_cases/scale/plot.py        — add correctness panel
  evaluation/use_cases/error_mitigation/run.py  — add E2E execution
  evaluation/use_cases/error_mitigation/plot.py — add accuracy panel
  .paper/sections/evaluation.tex            — all text changes
  .paper/sections/tensor_networks.tex       — expanded background
  .paper/sections/introduction.tex          — soften novelty claims
  .paper/sections/related_work.tex          — comparison table
  .paper/sections/programming_model.tex     — LoC table, IR discussion
```
