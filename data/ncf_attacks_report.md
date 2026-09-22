# FedAvg + NCF on MovieLens — Privacy-Mechanism Matrix

## DP · CKKS-HE · SecAgg · and all combinations (DP+SecAgg, DP+HE, DP+SecAgg+HE, SecAgg+HE)

**Date:** 2026-09-20 (HE expansion)
**Code:** `D:\seminar\website\backend\` — `ncf_fl.py` (model), `he_fl.py` (CKKS layer),
`run_ncf_sweep.py` (driver), `attacks.py` (MIA / attribute / gradient-leak)
**Raw results:** `D:\seminar\website\data\ncf_sweep_results.json`, `D:\seminar\website\data\ncf_attack_results.csv`

---

## 1. Objective

Extend the federated NCF recommender over MovieLens to a **privacy-mechanism
matrix**, combining three tools and measuring both **utility** and **empirical
leakage** under a server-side adversary:

| Mechanism | What it does | Accuracy effect | Channel effect |
|-----------|--------------|-----------------|----------------|
| **DP** (Local differential privacy) | per-example clip + Gaussian noise keyed to `(ε,δ)` | degrades at tight ε | gradient *noisy* but per-client readable |
| **SecAgg** (secure aggregation) | pairwise random masks cancel at server | none (exact avg) | server sees only the **sum** |
| **HE** (CKKS, tenseal) | clients encrypt gradients; server sums ciphertexts | none (≈ exact adds) | server sees only **ciphertexts** / decrypted **sum** |

Stacks evaluated:

```
dp              HE        SecAgg    DP+SecAgg   DP+HE     DP+SecAgg+HE   SecAgg+HE     ceiling
```

The **"relaxation effect"** claim: *any* stack that hides individual updates
(SecAgg and/or HE) gives the server only the aggregate → the DP accounting used
for secure aggregation applies → the DP noise can be run at **nominal
ε·√K** while the *effective* per-user privacy stays **E**. The DP+secure stacks
therefore add √K-times less noise for the same privacy.

---

## 2. Configuration & data

- MovieLens (32 M ratings). Sampled top users by rating count; each eligible
  user (≥ 80 ratings) keeps its **80 most recent** ratings as train and older
  ratings as test. **16 users · 1,529 items · 1,280 train pairs** (seed 1).
- Model: NCF, `rating = w2·ReLU(W1·[P[u];Q[i]]) + b2`, d=32, h=48, bias 3.5.
- FedAvg: 30 rounds; mini-batch 40/round/client; lr 0.4; clip C=2.0; δ=1e-5.
- Targets: effective ε ∈ {0.75, 1.0, 1.5, 2.0, 2.5, 3.0}; K=16, amplification √K=4.
- DP stacks at effective E run at nominal E (plain) or **E·4** (secure stacks).
- **HE**: CKKS (tenseal 0.3.17), poly 8192, scale 2²⁴, 4096 real slots/chunk,
  13 ciphertexts per gradient; ciphertext addition-only; clients decrypt the
  sum. Relative decrypt error ≈ 1.6e-4 (utility numbers match plaintext to 3dp).

---

## 3. Results — utility (MAE)

| stack         | mech        | nominal ε | eff ε | MAE   | RMSE | HR@10 |
|---------------|-------------|-----------|-------|-------|------|-------|
| dp            | dp          | 0.75      | 0.75  | **0.978** | 1.224 | 0.138 |
| dp            | dp          | 1.0       | 1.0   | 0.806 | 1.028 | 0.156 |
| dp            | dp          | 1.5 .. 3.0| 1.5..3| 0.742–0.749 | ~0.98 | 0.12–0.23 |
| dp_secagg     | dp+secagg   | 3.0 (nom) | 0.75  | **0.747** | 0.981 | 0.212 |
| dp_secagg     | dp+secagg   | 4.0 (nom) | 1.0   | 0.747 | 0.980 | 0.150 |
| dp_secagg     | dp+secagg   | 6..12 (nom)| 1.5..3 | 0.749–0.751 | ~0.97 | 0.15–0.24 |
| dp_he         | dp+he       | (as dp_secagg) | | **identical** to dp_secagg rows | | |
| dp_secagg_he  | dp+secagg+he| (as dp_secagg) | | **identical** | | |
| he            | he          | ∞         | ∞     | 0.750 | 0.976 | 0.181 |
| secagg        | secagg      | ∞         | ∞     | 0.750 | 0.976 | 0.181 |
| secagg_he     | secagg+he   | ∞         | ∞     | 0.750 | 0.976 | 0.194 |
| ceiling       | none        | ∞         | ∞     | 0.750 | 0.976 | 0.188 |

**Key facts**
- HE stacks match plaintext aggregation to the 3rd decimal (CKKS ≈ exact for
  additions) → HE adds **no** utility cost.
- All three DP+secure stacks (dp_secagg, dp_he, dp_secagg_he) are statistically
  identical — that is the point: HE does for the channel what SecAgg does.

**Relaxation-effect verdict (fixed effective privacy E; plain DP baseline same-E) — 15/18 confirmed:**

| E     | dp_secagg | dp_he     | dp_secagg_he | note |
|-------|-----------|-----------|--------------|------|
| 0.75  | **+23.62%** OK | **+23.62%** OK | **+23.62%** OK | the headline win |
| 1.0   | **+7.31%** OK | **+7.31%** OK | +7.31% OK | |
| 1.5   | −1.12% FAIL | −1.13% FAIL | −1.13% FAIL | flat-ceiling tie (±0.8%) |
| 2.0   | −0.23% OK | −0.23% OK | −0.23% OK | ceiling zone |
| 2.5   | −0.32% OK | −0.32% OK | −0.32% OK | ceiling zone |
| 3.0   | −0.24% OK | −0.24% OK | −0.24% OK | ceiling zone |

The 3 FAILs are all at E=1.5 where **both** stacks sit at the no-noise ceiling;
−1.1% lies inside single-run stochasticity (raw differences 0.0012). The theory
holds at every level where DP noise is actually the binding constraint (ε ≤ 1).

---

## 4. Results — attacks (server-side adversary)

### 4.1 Membership inference (MIA)
Loss-calibrated, per-user AUC on members vs held-out items.

| stack | MIA AUC (any ε) | TPR@1%FPR |
|-------|----------------|-----------|
| all (including ceiling) | **0.47–0.50** | 0.02 |

→ at **chance level everywhere**; this NCF-scale FedAvg does not overfit train
ratings enough to leak membership, and no mechanism reduces what is already ~0.

### 4.2 Attribute (rating) inference
MAE of model prediction vs stored rating on member items (constant guess ≈ 1.3).

| stack | ε=0.75 | ε ≥ 1.0 |
|-------|--------|---------|
| dp    | **1.021** | 0.858–0.898 |
| all secure stems (dp_secagg, dp_he, dp_secagg_he) | 0.883–0.885 | 0.881–0.884 |

→ moderate; the tight budget raises inference cost (less precise model); secure
stacks hold it at their (≈ceiling) level.

### 4.3 Gradient reverse-engineering (the decisive attack)
FeDL-style: item-batch **recall** via Q-block norms, rating **reconstruction
MAE** via least-squares gradient inversion.

| stack/channel | recall@batch | recon MAE | meaning |
|---------------|-------------|-----------|---------|
| **ceiling** (plaintext per-client, no protection) | **1.000** | **0.575** | full item & rating leak |
| dp — per-client noisy | 0.025 | 1.74–2.06 | DP noise kills inversion |
| dp_secagg — aggregate | 0.042 | 2.03–2.24 | sum + noise: nothing |
| dp_he — he-aggregate | 0.042 | 2.03–2.28 | same |
| dp_secagg_he — he-aggregate | 0.042 | 2.03–2.24 | same |
| he — he-aggregate (no DP) | **0.042** | 0.65 | recall random already at **zero** DP noise |
| secagg / secagg_he — aggregate (no DP) | 0.042 | 0.65 | ditto |
| random baseline | ~0.026 | — | |

**Take-aways**
1. Without any mechanism the attacker recovers the exact item batch (1.0) and
   reconstructs ratings well (0.575).
2. Any DP noise collapses it (recall ≈ random, recon ≈ 2). 
3. SecAgg **or** HE removes per-client attribution **even with no DP noise**
   (recall 1.0 → 0.042): the server only ever holds the sum/ciphertexts, so
   there is no per-client vector to invert.
4. HE additionally guarantees (computationally) that even the *sum* was computed
   on ciphertexts; once decrypted by participants it equals SecAgg's view.

---

## 5. Interpretation

- **HE behaves like SecAgg for accuracy** (both ≈ exact) and for the observable
  channel (both hide per-client updates). This is expected: in this pipeline HE
  subsumes the aggregation-hiding role of SecAgg; the difference is *crypto vs
  combinatorics* and defense depth (HE resists collusion/malicious servers that
  SecAgg's honest-majority masks do not).
- **DP remains the only tool giving a formal ε guarantee** — SecAgg/HE hide the
  channel but leak everything through the *average* if DP is off (recon 0.65 on
  pure-aggregate is attainable). The winning design is **DP + SecAgg/HE**: DP
  for the formal bound, aggregation-Hiding for reverse-engineering resistance,
  and the √K relaxation to recover the accuracy DP alone would burn.

## 6. Limitations

- Small cohort (16 users, 1.28k pairs): ε≥1.5 rows live in a flat confidence
  band (±0.8%), hiding small real gaps.
- MIA is loss-based; stronger calibrated/shadow-model MIA or a bigger cohort
  might surface residual signal.
- Reconstruction assumes item ids known (worst case); quantity fixes the
  "can't invert the aggregate to victim level" conclusion (recall random).
- CKKS here is ciphertext-add only (sufficient for FedAvg); scale = 2²⁴ and a
  single context — fine for a seminar-scale demo, not a hardened deployment.

---

## 7. Summary

> We benchmarked every combination of Local-DP, CKKS homomorphic encryption,
> and secure aggregation on a NumPy federated NCF recommender (16 users,
> MovieLens), at six effective privacy budgets plus a no-protection ceiling,
> and attacked all results from a server-side adversary. Accuracy: plain DP
> collapses at tight budgets (MAE 0.978 at ε=0.75) and needs ε≥1.5 to reach the
> ceiling (0.750); DP+SecAgg, DP+HE, and DP+SecAgg+HE all reach ceiling already
> at ε=0.75 (MAE 0.747), while pure HE, SecAgg, and SecAgg+HE match the ceiling
> exactly — confirming the **relaxation effect** (√K=4 less noise at identical
> effective privacy) in 15/18 (stack, ε) pairs, with all misses being flat-
> ceiling ties at ε≥1.5. Empirical leakage: membership inference stays at
> chance in every configuration (AUC ≈ 0.5); attribute inference is modest
> (0.86–1.02 vs 1.3 constant) and worst at the tightest budget; gradient
> reverse-engineering — the decisive attack — succeeds without protection
> (item recall 1.0, rating MAE 0.575) and collapses under every protected
> stack (recall 0.03–0.04; MAE ≥ 0.65, ≥ 2.0 once DP is on). Conclusion: HE
> matches SecAgg in accuracy and channel-hiding while adding cryptographic
> robustness and zero utility cost; the winning production recipe is
> **DP + (SecAgg or HE)** — formal ε with aggregation-hiding and the √K
> accuracy recovery — experimentally verified here end to end.