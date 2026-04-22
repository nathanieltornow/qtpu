"""Post-hoc patch of `estimated_qpu_time_s` in existing QNN Marrakesh logs.

Earlier runs of `sweep_marrakesh_qnn.py` logged `estimated_qpu_time_s` using
only the circuit gate-schedule term, which is ~70x short of IBM's billed
`quantum_seconds`. The missing piece is a deterministic per-shot platform
overhead (`default_rep_delay` + control-plane ≈ 340 µs/shot).

This script walks an existing log JSONL and adds two fields per row without
overwriting the originals:
  - `estimated_qpu_time_s_gate_only`: the previously-stored value, preserved.
  - `estimated_qpu_time_s_v2`: overhead-corrected value (gate_term + shots*overhead).

Run:
    uv run python -m evaluation.hardware.patch_qpu_time_estimate logs/hardware/qnn_marrakesh.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Must match the constant in sweep_marrakesh_qnn.py
_IBM_CONTROL_OVERHEAD_S = 90e-6
# Marrakesh FakeBackend config: default_rep_delay = 250 µs
_REP_DELAY_S = 250e-6
_PER_SHOT_OVERHEAD_S = _REP_DELAY_S + _IBM_CONTROL_OVERHEAD_S  # 340 µs


def _patch_row(row: dict) -> dict:
    """Add overhead-corrected estimate fields to mono and cut sub-dicts."""
    shots_mono = row.get("shots_mono", 0)
    shots_cut = row.get("shots_cut", 0)

    # Mono: per-circuit estimate, one circuit submission, shots_mono shots.
    mono = row.get("mono")
    if mono and "estimated_qpu_time_s" in mono:
        gate_only = mono["estimated_qpu_time_s"]
        mono["estimated_qpu_time_s_gate_only"] = gate_only
        if gate_only == gate_only:  # not NaN
            mono["estimated_qpu_time_s_v2"] = gate_only + shots_mono * _PER_SHOT_OVERHEAD_S
        else:
            mono["estimated_qpu_time_s_v2"] = float("nan")

    # Cut: sum over fragments of (per-flat-gate-duration * shots * flat_count).
    # We don't have per-flat duration in the stored row, so we reconstruct the
    # shot count from `estimated_flats_scored` (which is the count of flats
    # actually timed) and add `total_shots × overhead` to the gate-only total.
    cut = row.get("cut")
    if cut and "estimated_qpu_time_s" in cut:
        gate_only = cut["estimated_qpu_time_s"]
        flats_scored = cut.get("estimated_flats_scored", cut.get("num_flat_circuits", 0))
        total_shots = flats_scored * shots_cut
        cut["estimated_qpu_time_s_gate_only"] = gate_only
        if gate_only == gate_only:
            cut["estimated_qpu_time_s_v2"] = gate_only + total_shots * _PER_SHOT_OVERHEAD_S
        else:
            cut["estimated_qpu_time_s_v2"] = float("nan")

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=Path, help="Path to qnn_marrakesh.jsonl")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite input (default: write *.patched.jsonl alongside)")
    args = ap.parse_args()

    if not args.log_path.exists():
        print(f"No such file: {args.log_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.log_path if args.in_place else args.log_path.with_suffix(".patched.jsonl")

    patched_rows = []
    for line in args.log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as e:
            print(f"skip unparseable line: {e}", file=sys.stderr)
            continue
        patched_rows.append(_patch_row(row))

    with out_path.open("w") as f:
        for row in patched_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Patched {len(patched_rows)} rows -> {out_path}")
    # Print a quick before/after table
    print(f"\n{'n':>3} | {'shots_mono':>10} {'shots_cut':>9} | "
          f"{'mono_v1':>9} {'mono_v2':>9} {'mono_act':>9} | "
          f"{'cut_v1':>9} {'cut_v2':>9} {'cut_act':>9}")
    print("-" * 95)
    for r in patched_rows:
        m, c = r.get("mono", {}), r.get("cut", {})
        m_v1 = m.get("estimated_qpu_time_s_gate_only", float("nan"))
        m_v2 = m.get("estimated_qpu_time_s_v2", float("nan"))
        m_act = m.get("actual_qpu_time", float("nan"))
        c_v1 = c.get("estimated_qpu_time_s_gate_only", float("nan"))
        c_v2 = c.get("estimated_qpu_time_s_v2", float("nan"))
        c_act = c.get("actual_qpu_time", float("nan"))
        print(f"{r.get('n', '?'):>3} | {r.get('shots_mono', 0):>10} {r.get('shots_cut', 0):>9} | "
              f"{m_v1:>7.2f}s {m_v2:>7.2f}s {m_act:>7.2f}s | "
              f"{c_v1:>7.2f}s {c_v2:>7.2f}s {c_act:>7.2f}s")


if __name__ == "__main__":
    main()
