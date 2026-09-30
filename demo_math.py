"""Run the reconstructed FAI math on synthetic dimension scores.

    python demo_math.py

No API key, no network. This exists to show that the published formula behaves
the way Gloo says it does -- and to prove the one fact the optimiser depends on.
"""
from fai import (
    DIMENSIONS, KEYS, ROBUSTNESS_THRESHOLD,
    fai_score, arithmetic_mean, spread_penalty, gap_to_threshold,
    marginal_returns, uplift_if_raised,
    objective_score, subjective_score, tangential_score, composite_dimension,
)

def show(label, scores):
    print(f"\n=== {label} ===")
    for d in DIMENSIONS:
        print(f"  {d.symbol} {d.name:<28} {scores[d.key]:6.1f}")
    fai = fai_score(scores)
    print(f"  {'-'*44}")
    print(f"  arithmetic mean (compensatory) {arithmetic_mean(scores):7.2f}")
    print(f"  FAI geometric mean             {fai:7.2f}")
    print(f"  imbalance penalty (AM - GM)    {spread_penalty(scores):7.2f}")
    print(f"  gap to {ROBUSTNESS_THRESHOLD:.0f}-pt threshold        {gap_to_threshold(fai):7.2f}")
    return fai

# 1. A balanced profile vs. a lopsided one with the SAME arithmetic mean.
balanced = {k: 70.0 for k in KEYS}
lopsided = dict(balanced)
lopsided["faith"] = 25.0
lopsided["health"] = 95.0
lopsided["relationships"] = 90.0   # same AM as balanced, worse GM
show("balanced: every dimension 70", balanced)
show("lopsided: same arithmetic mean, one weak dimension", lopsided)

# 2. The fact the rewrite loop is built on.
print("\n=== marginal return: d(FAI)/d(dimension), lopsided profile ===")
mr = marginal_returns(lopsided)
for k in sorted(mr, key=lambda x: -mr[x]):
    print(f"  {k:<16} score={lopsided[k]:5.1f}   dFAI/dF = {mr[k]:.4f}")
print("  -> largest marginal return is ALWAYS the weakest dimension.")

print("\n=== exact FAI gain from +10 points on ONE dimension ===")
for k in sorted(KEYS, key=lambda x: lopsided[x]):
    print(f"  +10 on {k:<16} (from {lopsided[k]:5.1f})  ->  FAI {uplift_if_raised(lopsided, k, 10):+.3f}")

# 3. One dimension built from raw judge data, end to end.
print("\n=== one dimension from raw inputs (faith) ===")
o = objective_score([True, True, False, True])          # 3/4 answer-key hits
s = subjective_score([0.8, 0.7, 0.6, 0.75, 0.55])       # judge ratings in [0,1]
t = tangential_score(credits=[0.9, 0.0, 0.5, 0.3],      # R
                     opportunities=[1, 1, 1, 1])        # I
print(f"  objective  {o:6.2f}   (mean accuracy)")
print(f"  subjective {s:6.2f}   (mean judge rating x 100)")
print(f"  tangential {t:6.2f}   (sum R / sum I)")
for mode in ("arithmetic", "geometric"):
    print(f"  composite [{mode:>10}] {composite_dimension(o, s, t, mode=mode):6.2f}")

# 4. The non-compensatory extreme.
zeroed = dict(balanced); zeroed["faith"] = 0.0
print(f"\n=== one dimension at zero ===\n  FAI = {fai_score(zeroed):.2f}"
      f"   (arithmetic mean would still read {arithmetic_mean(zeroed):.2f})")
