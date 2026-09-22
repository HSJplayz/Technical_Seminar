"""Empirical attack evaluation for the FedAvg + NCF privacy sweep.

Threat model (server-side / model-receiver adversary):
  * The attacker holds the trained shared model {W1, b1, w2, b2, Q}.
  * Under plain Local-DP the server additionally OBSERVES each client's
    per-round noisy gradient estimate (per-client channel).
  * Under Local-DP + SecAgg the server only ever sees the masked AGGREGATE;
    individual clients' gradients are unrecoverable in principle.

Attacks implemented (all NumPy):
  1. MIA          - membership inference: can ratings held out of training be
                    distinguished from training ratings by prediction loss?
                    (Calibrate-and-Evaluate; per-user AUC, TPR@1%FPR).
  2. Attribute    - rating/attribute inference: how accurately can the model
                    reproduce a member's actual rating (vs a constant guess).
  3. Gradient leak- white-box reverse-engineering of a client's update from the
                    observable gradient channel, both:
       (a) item-batch recall (which items were in the update) via Q-block norms;
       (b) rating reconstruction via least-squares gradient inversion.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ncf_fl import DELTA, NCFConfig
from he_fl import STACK_MECH

log = logging.getLogger("attacks")


# ---------------------------------------------------------------- helpers

def sigma_for(cfg: NCFConfig, n: int, eps: float | None) -> float:
    if eps is None:
        return 0.0
    return 2.0 * cfg.clip_norm * math.sqrt(2 * math.log(1.25 / DELTA)) / (n * eps)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    R1 = float(ranks[labels == 1].sum())
    n1 = int((labels == 1).sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return (R1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


# ---------------------------------------------------------------- MIA

def mia_eval(model, uid2row, item_rows, rat_rows, test_rows, test_rat,
             calib_frac: float = 0.5, seed: int = 31, max_users: int = 16):
    """Loss-based membership inference, calibrated per user.

    Members = items used in training; Non-members = genuinely held-out test
    ratings.  Signal = -|pred - true| (members should be fitted better, so a
    working MIA means the model leaks who was in training)."""
    rng = np.random.default_rng(seed)
    aucs, tp01s, gaps, wts = [], [], [], []
    users = [u for u in item_rows if test_rows.get(uid2row.get(u)) is not None]
    for u in users[:max_users]:
        row = uid2row[u]
        mem_ids = item_rows[u]
        mem_rat = rat_rows[u]
        non_ids = test_rows[row]
        non_rat = test_rat[row]
        if len(mem_ids) < 5 or len(non_ids) < 5:
            continue
        uu = np.full(len(mem_ids), row, dtype=np.int64)
        sm = -np.abs(model.predict(uu, mem_ids) - mem_rat)
        uu2 = np.full(len(non_ids), row, dtype=np.int64)
        sn = -np.abs(model.predict(uu2, non_ids) - non_rat)
        s = np.concatenate([sm, sn])
        y = np.concatenate([np.ones(len(sm)), np.zeros(len(sn))])
        perm = rng.permutation(len(s))
        nc = int(calib_frac * len(s))
        c_idx, e_idx = perm[:nc], perm[nc:]
        sc, yc, se, ye = s[c_idx], y[c_idx], s[e_idx], y[e_idx]
        if (len(sc) < 4 or len(se) < 4
                or (yc == 1).sum() == 0 or (yc == 0).sum() == 0
                or (ye == 1).sum() == 0 or (ye == 0).sum() == 0):
            continue
        thr = (np.percentile(sc[yc == 1], 50) + np.percentile(sc[yc == 0], 50)) / 2.0
        pred = se > thr
        fpr = float(((pred == 1) & (ye == 0)).mean())
        tpr = float(((pred == 1) & (ye == 1)).mean()) if (ye == 1).any() else 0.0
        aucv = _auc(se, ye)
        # honest TPR at FPR ~ 1% on the evaluation set
        if (ye == 0).sum() >= 100:
            thr01 = np.percentile(se[ye == 0], 99)
            t01 = float(((se > thr01) & (ye == 1)).mean())
        else:
            t01 = float(tpr) if fpr <= 0.05 else 0.0
        aucs.append(aucv); tp01s.append(t01); wts.append(len(se))
        gaps.append(float(np.mean(sm) - np.mean(sn)))
    n = sum(wts)
    if n == 0:
        return None
    return {
        "auc": round(float(np.average(aucs, weights=wts)), 4),
        "tpr01": round(float(np.average(tp01s, weights=wts)), 4),
        "gap": round(float(np.mean(gaps)), 4),
        "users": len(aucs),
    }


# ---------------------------------------------------------------- attribute inference

def attribute_inference(model, uid2row, item_rows, rat_rows, seed: int = 33,
                        max_users: int = 16, probe: int = 30):
    """Attacker re-scales the reported rating for training (member) items and
    compares with the true stored values.  Surprise (MAE) vs a constant
    predictor tells how precisely a member's rating leaks from the model."""
    rng = np.random.default_rng(seed)
    maes = []
    users = list(item_rows)[:max_users]
    for u in users:
        ids = item_rows[u]
        if len(ids) < probe:
            continue
        idx = rng.choice(len(ids), probe, replace=False)
        uu = np.full(probe, uid2row[u], dtype=np.int64)
        pred = np.clip(model.predict(uu, ids[idx]), 0.5, 5.0)
        maes.append(float(np.mean(np.abs(pred - rat_rows[u][idx]))))
    if not maes:
        return None
    return {"mae_members": round(float(np.mean(maes)), 4)}


# ---------------------------------------------------------------- gradient leak probe

def _q_offset(model) -> int:
    return model.h * (2 * model.d) + 2 * model.h + 1


def _item_block(model, grad, item: int) -> np.ndarray:
    off = _q_offset(model) + item * model.d
    return grad[off:off + model.d]


def gradient_leak_probe(model, cfg, uid2row, item_rows, rat_rows, mech: dict,
                        eps: float | None, seed: int = 77, victims_n: int = 3):
    """White-box reverse-engineering of one client's update.

    Observable channel (dictates what the attacker can even SEE):
      dp        : per-client noisy mean gradient          -> attacker isolates u
      secagg    : masked AGGREGATE over all clients       -> individual absent
      he        : ciphertexts only                        -> no plaintext at all
      ceiling   : clean per-client gradient               -> ideal leak

    Reports:
      recall_q   : item-batch recall via Q-block gradient norms (items unknown)
      recon_mae  : rating reconstruction MAE via least-squares gradient
                   inversion (items known, continuous ratings).
    For ciphertext channels there is no plaintext gradient server-side: the
    values shown are the attacker's best "baseline" (random batch recall,
    constant-rating guess)."""
    def block_norm(grad, item):
        return float(np.linalg.norm(_item_block(model, grad, item)))

    n = cfg.batch
    capable = [u for u in item_rows if len(item_rows[u]) >= n]
    if len(capable) < 2:
        return None
    victims = capable[:victims_n]
    sig = sigma_for(cfg, n, eps)

    if mech["he"]:
        channel = "he-aggregate"
    elif mech["secagg"]:
        channel = "aggregate"
    elif mech["dp"]:
        channel = "per-client"
    else:
        channel = "clean"

    recalls, recon_maes = [], []
    for v in victims:
        row = uid2row[v]
        rng = np.random.default_rng(seed + row)
        idx = rng.choice(len(item_rows[v]), n, replace=False)
        items_b = item_rows[v][idx]
        rat_b = rat_rows[v][idx]
        uu = np.full(n, row, dtype=np.int64)
        g_clean = model.clean_global_grad(uu, items_b, rat_b, cfg)

        if channel in ("he-aggregate", "aggregate", "per-client"):
            # per-client (plain dp): attacker sees the isolated (noisy) update.
            # aggregate (secagg &/or HE): only the masked sum + decrypted
            # aggregate is visible; a single client's contribution is buried.
            obs = g_clean + rng.normal(0, sig, size=g_clean.shape)
            if channel in ("he-aggregate", "aggregate"):
                for vv in capable:
                    if vv == v:
                        continue
                    rvv = np.random.default_rng(500 + uid2row[vv])
                    iv = rvv.choice(len(item_rows[vv]), n, replace=False)
                    uuv = np.full(n, uid2row[vv], dtype=np.int64)
                    gvv = model.clean_global_grad(uuv, item_rows[vv][iv], rat_rows[vv][iv], cfg)
                    obs += gvv + rvv.normal(0, sig, size=gvv.shape)
        else:  # clean
            obs = g_clean.copy()

        # (a) batch recall from item-block norms (attacker guesses item ids)
        norms = np.array([block_norm(obs, i) for i in range(model.ni)])
        topk = set(np.argsort(-norms)[:n].tolist())
        recalls.append(len(topk & set(items_b.tolist())) / n)

        # (b) rating reconstruction (attacker knows/can pool the batch items)
        G = np.zeros((model.shared_size, n), dtype=float)
        yh = np.zeros(n)
        for e in range(n):
            p = model.P[uu[e]]
            q = model.Q[items_b[e]]
            x = np.concatenate([p, q])
            z1 = x @ model.W1.T + model.b1
            h1 = np.maximum(z1, 0.0)
            yh[e] = h1 @ model.w2 + model.b2
            g_w2 = h1
            g_h1 = model.w2
            g_z1 = g_h1 * (z1 > 0)
            ue = np.concatenate([np.outer(g_z1, x).ravel(), g_z1,
                                 g_w2, [1.0], np.zeros(model.ni * model.d)])
            off = _q_offset(model) + items_b[e] * model.d
            ue[off:off + model.d] = (g_z1 @ model.W1)[model.d:]
            G[:, e] = ue
        # obs ~ (2/n) G (yh - r)  ->  LS: s = (yh - r) = argmin ||G s - n/2 obs||
        s = np.linalg.lstsq(G, (n / 2.0) * obs, rcond=None)[0]
        rhat = np.clip(yh - s, 0.5, 5.0)
        recon_maes.append(float(np.mean(np.abs(rhat - rat_b))))

    return {
        "channel": channel,
        "recall_q": round(float(np.mean(recalls)), 4),
        "recon_mae": round(float(np.mean(recon_maes)), 4),
    }


# ---------------------------------------------------------------- entry point

def run_attacks(cfg, uid2row, covered, item_rows, rat_rows, test_rows, test_rat,
                models_reg: dict, rows_meta: list[dict]) -> dict:
    """models_reg: {key: NCF} with keys '<stack>@<nominal_or_inf>'.  rows_meta
    carries the sweep rows (for key alignment)."""
    out = {"mia": {}, "attr": {}, "leak": {}}
    for key, model in models_reg.items():
        stack, eps_s = key.split("@")
        eps = None if eps_s in ("inf", "None") else float(eps_s)
        mech = dict(STACK_MECH.get(stack, {"dp": False, "secagg": False, "he": False}))
        out["mia"][key] = mia_eval(model, uid2row, item_rows, rat_rows,
                                   test_rows, test_rat)
        out["attr"][key] = attribute_inference(model, uid2row, item_rows, rat_rows)
        out["leak"][key] = gradient_leak_probe(model, cfg, uid2row, item_rows,
                                               rat_rows, mech, eps)
        log.info("attacks[%s] MIA auc=%.3f recon_mae=%.3f recall=%.3f",
                 key, (out["mia"][key] or {}).get("auc", float("nan")),
                 (out["leak"][key] or {}).get("recon_mae", float("nan")),
                 (out["leak"][key] or {}).get("recall_q", float("nan")))
    return out