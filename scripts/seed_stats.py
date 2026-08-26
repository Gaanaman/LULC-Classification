"""Aggregate multi-seed runs: mean, standard deviation and a paired test.

Reads metrics.json for every seed of a run and reports the spread, then runs a
Welch t-test between two configurations across seeds.
"""
import json, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEEDS = {
    "RGB (3 bands)": ["scratch_cnn_ms_rgb", "scratch_cnn_ms_rgb_s43", "scratch_cnn_ms_rgb_s44"],
    "k=2 learned":   ["scratch_cnn_ms_proj2", "scratch_cnn_ms_proj2_s43", "scratch_cnn_ms_proj2_s44"],
}


def acc(run):
    p = Path(f"outputs/reports/{run}/metrics.json")
    return json.load(open(p))["accuracy"] * 100 if p.exists() else None


def main():
    got = {}
    for label, runs in SEEDS.items():
        vals = [a for a in (acc(r) for r in runs) if a is not None]
        got[label] = vals
        if vals:
            m = st.mean(vals)
            s = st.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{label:16s} n={len(vals)}  " +
                  ", ".join(f"{v:.2f}%" for v in vals) +
                  f"   mean {m:.2f}%  sd {s:.2f}%")
        else:
            print(f"{label:16s} no runs found")

    a, b = got.get("k=2 learned", []), got.get("RGB (3 bands)", [])
    if len(a) > 1 and len(b) > 1:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"\nWelch t-test, k=2 against RGB across seeds: "
              f"t={t:.2f}, p={p:.3f}, mean difference {st.mean(a)-st.mean(b):+.2f}%")
        print(f"lowest k=2 seed {min(a):.2f}%  |  highest RGB seed {max(b):.2f}%  "
              f"-> {'no overlap' if min(a) > max(b) else 'ranges overlap'}")
    else:
        print("\nnot enough seeds yet for the paired comparison")


if __name__ == "__main__":
    main()
