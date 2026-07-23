"""
find_min_coverage.py

Systematically finds the minimum number of technique IDs needed for the
model to correctly identify each APT group — instead of relying on one
lucky/unlucky manual test.

For each group and each candidate technique-count, it draws MULTIPLE random
subsets (not just one) and reports the success rate, so you get a reliable
answer like "APT28 needs ~20-25 techniques for >80% reliable top-3 accuracy"
instead of a single anecdote.

Run:
    python find_min_coverage.py
"""

import pathlib
import random
import sys
from collections import defaultdict

import numpy as np
import joblib

ROOT = pathlib.Path(".").resolve()
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))
from inference import predict_top3

X = joblib.load(ARTIFACTS / "X.joblib")
y = np.array(joblib.load(ARTIFACTS / "y.joblib"))
feature_vocab = joblib.load(ARTIFACTS / "feature_vocab.joblib")
all_tids = sorted(feature_vocab, key=feature_vocab.get)
all_labels = sorted(set(y))

GROUPS_TO_TEST = ["APT28", "Lazarus Group", "APT29", "APT41", "MuddyWater", "Sandworm Team"]

TRIALS_PER_COUNT = 15          # random subsets drawn per technique-count
CANDIDATE_COUNTS = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]
RANDOM_SEED = 42


def reconstruct_full_signature(label):
    idx = np.where(y == label)[0]
    row_sums = X[idx].sum(axis=1)
    full_row = X[idx[np.argmax(row_sums)]]
    return [all_tids[i] for i, v in enumerate(full_row) if v > 0]


random.seed(RANDOM_SEED)

print("=" * 90)
print("MINIMUM TECHNIQUE COUNT SWEEP — how many IDs does each group need?")
print(f"({TRIALS_PER_COUNT} random trials per count, seed={RANDOM_SEED})")
print("=" * 90)

summary_rows = []

for group_key in GROUPS_TO_TEST:
    real_label = next((l for l in all_labels if group_key.lower() in l.lower()), None)
    if not real_label:
        print(f"\n⚠ '{group_key}' not found in labels, skipping.")
        continue

    full_sig = reconstruct_full_signature(real_label)
    n_full = len(full_sig)

    print(f"\n{'─'*90}")
    print(f"{real_label}  (real signature: {n_full} techniques)")
    print(f"{'Count':>6} {'Coverage':>9} {'Rank-1 acc':>11} {'Top-3 acc':>10}  {'Verdict'}")

    first_reliable_count = None

    for n_keep in CANDIDATE_COUNTS:
        if n_keep >= n_full:
            break

        rank1_hits = 0
        top3_hits = 0
        for _ in range(TRIALS_PER_COUNT):
            subset = random.sample(full_sig, n_keep)
            result = predict_top3(subset)
            preds = [r["group"] for r in result]
            if preds[0] == real_label:
                rank1_hits += 1
            if real_label in preds:
                top3_hits += 1

        rank1_acc = rank1_hits / TRIALS_PER_COUNT
        top3_acc = top3_hits / TRIALS_PER_COUNT
        coverage = n_keep / n_full * 100

        verdict = ""
        if top3_acc >= 0.8 and first_reliable_count is None:
            first_reliable_count = n_keep
            verdict = "← first reliable point (>=80% top-3 acc)"

        print(f"{n_keep:>6} {coverage:>8.0f}% {rank1_acc:>11.2f} {top3_acc:>10.2f}  {verdict}")

    summary_rows.append({
        "group": real_label,
        "full_signature_size": n_full,
        "min_reliable_count": first_reliable_count,
        "min_reliable_coverage_pct": (first_reliable_count / n_full * 100) if first_reliable_count else None,
    })

print("\n" + "=" * 90)
print("SUMMARY — minimum technique count for >=80% reliable top-3 accuracy")
print("=" * 90)
print(f"{'Group':<20} {'Full size':>10} {'Min reliable N':>16} {'Min coverage %':>16}")
for r in summary_rows:
    n = r["min_reliable_count"]
    pct = r["min_reliable_coverage_pct"]
    n_str = str(n) if n else "not reached in sweep"
    pct_str = f"{pct:.0f}%" if pct else "—"
    print(f"{r['group']:<20} {r['full_signature_size']:>10} {n_str:>16} {pct_str:>16}")

print("""
NOTE: "min reliable count" is where top-3 accuracy first hits >=80% across
15 random trials — a more trustworthy number than a single manual test,
since it accounts for the fact that WHICH techniques you happen to pick
matters (some 25-technique subsets will work, others won't).
""")