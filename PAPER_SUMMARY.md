# qTPU: Hybrid Tensor Networks for Quantum-Classical Acceleration

**Paper #1414 — OSDI '26**

## Research Question

How can we design a concise and expressive hybrid computing paradigm that unifies quantum and classical computation while enabling efficient execution across heterogeneous accelerators?

## Key Idea: Hybrid Tensor Networks (hTNs)

Tensor Networks (TNs) already unify quantum circuit simulation and classical linear algebra under one formalism (einsum). qTPU extends this into **hybrid Tensor Networks (hTNs)** — a single connected graph containing both **quantum tensors** (qTensors, executed on QPUs) and **classical tensors** (cTensors, executed on GPUs/TPUs). This unified representation enables holistic optimization across the quantum-classical boundary.

## Problem: Three Limitations of Current Hybrid Computing

1. **Fragmented programming models** — Developers write separate quantum kernels and classical kernels, manually coordinating data movement and control flow.
2. **Lack of holistic compiler optimizations** — The fixed quantum-classical boundary prevents cross-paradigm optimization; compilers cannot reassign work between QPUs and GPUs.
3. **Poor scalability and rigid runtime** — Manual orchestration as sequential steps prevents dynamic parallelism and resource-adaptive scheduling.

## System Design

### Programming Model (Section 5)

Two core primitives:

- **`iswitch(index, family)`** — Annotates quantum programs with symbolic indices, each binding to a discrete set of gate operations. This compactly represents an entire *family* of related quantum circuits as a single qTensor (e.g., all batch variants in ML, or all Pauli-basis variants in error mitigation).
- **`hEinsum(expr, tensors...)`** — A declarative multi-tensor contraction operator composing qTensors and cTensors. Specifies *what* to compute (not *how*), letting the compiler and runtime handle scheduling, placement, and code generation.

### Compiler (Section 6)

Three stages:

1. **Frontend** — Parses the hEinsum into a qTPU intermediate representation (IR). Converts each qTensor into an operation graph (DAG of gates on qubits).

2. **Hyper-Optimizer** — Applies three optimization primitives to rewrite qTensors:
   - *Tensorization & decomposition* — Captures linear combinations of related hEinsums as a single hEinsum with iswitch-encoded variations.
   - *Spatial separation (gate virtualization)* — Cuts two-qubit gates into single-qubit operations + classical coefficients, splitting large qTensors across QPUs.
   - *Temporal separation (wire cutting)* — Cuts qubit wires at specific time steps, splitting deep circuits into shallower fragments.

   The optimizer uses a **dual-objective cost model** balancing:
   - **Classical cost** — FLOPs for tensor contraction.
   - **Quantum error** — Estimated probability of at least one error during circuit execution (based on gate error rates).

   It iteratively partitions qTensor graphs via KaHyPar, greedily moving toward the Pareto-optimal front. Multiple randomized hyperparameter trials explore the cost-error tradeoff space.

3. **Backend** — Canonicalizes the operation graph (binary contractions), optimizes contraction order, and lowers each qTensor into a parametric CUDA-Q quantum kernel with `if-else` blocks for iswitch indices.

### Runtime (Section 7)

Two phases:

1. **Adaptation phase:**
   - **Index Slicer** — Partitions the hEinsum along selected indices to expose independent sub-problems for parallel execution.
   - **Device Placer** — Assigns each qTensor instance to a QPU and classical slices to GPUs, using load-balanced round-robin assignment.
   - **Code Generator** — Compiles quantum kernels via CUDA-Q and classical contraction kernels via PyTorch.

2. **Execution phase (map-reduce):**
   - **qTensor Engine** — Evaluates all qTensor instances in parallel on assigned QPUs, materializing results into cTensors.
   - **cTensor Engine** — Performs classical contractions on GPUs, combining partial results across slices.
   - Fault tolerance: QPU calibration handled by reassignment; classical failures trigger re-execution of affected slices only.

## Evaluation Methodology

### Hardware
- 2x AMD EPYC 9654 96-Core (384 cores w/ HT), 1.5 TB RAM
- 1x NVIDIA A40 GPU (48GB HBM3)
- QPU time estimated via Qiskit transpilation to IBM Marrakesh + ASAP scheduling (1000 shots)

### Benchmarks (from MQT Bench)
| Benchmark | Description | Sizes |
|-----------|------------|-------|
| **QNN** | Quantum neural network for ML | 20–140q |
| **W-State** | n-qubit linearly entangled W-state preparation | 20–140q |
| **VQE-SU2** | Variational Quantum Eigensolver with SU(2) rotations | 20–140q |
| **Dist-VQE** | Distributed VQE with clustered connectivity | 100q |

### Baselines
| Baseline | Used For | Description |
|----------|---------|-------------|
| **QAC** (Qiskit Addon Cutting) | Compiler, Scale | State-of-the-art circuit cutting framework |
| **cuTensorNet** | Runtime | NVIDIA GPU-accelerated tensor network simulation |
| **BATCH** (Listing 1) | Hybrid ML | Manual batched quantum-classical execution |
| **Mitiq** | Error mitigation | Leading QEM framework |

### Metrics
1. **Quantum error** — Estimated probability of at least one error (uniform model: 10^-3 single-qubit, 10^-2 two-qubit gates)
2. **Classical cost** — FLOPs for tensor contraction
3. **Runtime** — Wall-clock seconds (end-to-end or per-phase)
4. **Code size** — Lines of generated CUDA-Q kernel code
5. **QPU time** — Estimated quantum hardware execution time

## Claims and Results

### Compiler

**RQ1 — Compiler trade-off (Fig 9):** Partition 100-qubit circuits into 50-qubit subcircuits. QAC produces a single fixed solution; qTPU exposes a Pareto frontier trading classical cost for quantum error. Mid-range solutions yield 1.5–3.4x error reduction; maximum classical investment achieves 2.2–7.2x improvement.

**RQ2 — Compiler scalability (Fig 10, Table 1):** Circuit sizes 20–140 qubits, cutting to 50%. qTPU maintains stable error reduction (2–26x) across scales. QAC error grows linearly with circuit size. Compile time: QAC scales superlinearly (1.2s to 134s); qTPU stays near-constant (~1–3s, up to 53x speedup at 140 qubits).

### Runtime

**RQ3 — Runtime analysis (Fig 11a):** Total runtime scales with circuit size across all benchmarks. For 100-qubit circuits, quantum execution dominates (75–100% of time); classical contraction is <0.01%; compilation takes 2–3s.

**RQ4 — Multi-QPU scalability (Fig 11b):** Near-linear speedup from 1 to 16 QPUs (14.4x for W-State, 90% efficiency). The slicer produces embarrassingly parallel qTensor instances.

**RQ5 — vs. cuTensorNet (Fig 11c):** On 100-qubit Dist-VQE with 4 QPUs, qTPU outperforms classical simulation beyond 18-qubit clusters, achieving 6.7x speedup at 19-qubit clusters. As cluster size grows, qTPU runtime *decreases* (fewer, larger qTensors) while cuTensorNet grows exponentially.

### Case Study: Hybrid ML (Section 8.6)

**RQ6 — Compilation scalability (Fig 13a,b):** qTPU achieves 3.7x average speedup (up to 6x) over batch execution. qTPU scales sublinearly with batch size; BATCH scales linearly. At 100 qubits, batch size 200: qTPU 4.8s vs BATCH 16.9s (4.5x).

**RQ7 — Code generation overhead (Fig 13c,d):** qTPU compiles the quantum kernel once with parametric control flow. Average 33x code reduction (up to 48x). At 100 qubits, batch 200: 132k lines vs 5.1M lines (38.7x reduction).

### Case Study: Scalable Hybrid Computing / Circuit Knitting (Section 8.7)

**RQ8 — End-to-end runtime (Fig 12a):** QNN on 10-qubit QPU, 20–80 qubits. QAC times out beyond 40 qubits (20-min limit). qTPU completes 80-qubit circuits in <30s.

**RQ9 — Compilation overhead (Fig 12b):** QAC circuit generation grows combinatorially (12 to 6,480 circuits, 20q to 50q). qTPU produces 10–42x fewer subcircuit instances by representing families as qTensors.

**RQ10 — Classical overhead (Fig 12c):** qTPU maintains <10^3 FLOPs across all sizes. QAC grows to 10^6 FLOPs exponentially. The hTN formulation performs only structure-aware contractions.

### Case Study: Quantum Error Mitigation (Section 8.8)

**RQ11 — QEM scalability (Fig 14):** 100-qubit QNN with 200 single-qubit gates, PEC + Pauli twirling + ZNE. Mitiq explicitly generates each circuit variant; qTPU represents all 4^200 ~ 10^120 configurations as a single qTensor.

- **Compile time:** qTPU ~10ms constant; Mitiq scales linearly from ~335ms (100 samples) to 35s (10,000 samples). **Up to 3,500x faster.**
- **Code size:** qTPU ~3.7k LoC; Mitiq generates up to 13.5M LoC. **Up to 3,700x reduction.**

## Summary of Headline Numbers

| Metric | Improvement | Context |
|--------|-----------|---------|
| Classical overhead | 3–4 orders of magnitude lower | vs QAC (circuit knitting) |
| Quantum error | Up to 7.2x lower | vs QAC (compiler optimization) |
| Compilation speed | Up to 53x faster | vs QAC (140q VQE-SU2) |
| End-to-end speedup | Over 20x | vs QAC (circuit knitting, 80q) |
| Multi-QPU scaling | 14.4x on 16 QPUs (90% eff.) | W-State benchmark |
| vs classical sim | 6.7x speedup | vs cuTensorNet (100q Dist-VQE) |
| Code reduction | Up to 3,700x | vs Mitiq (error mitigation) |
| Compile time | Up to 3,500x faster | vs Mitiq (error mitigation) |
