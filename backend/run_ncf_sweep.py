"""Driver: FedAvg + NCF mechanism sweep on MovieLens.

Stacks (privacy mechanisms combined):
    dp            : plain Local-DP                       (nominal eps = E)
    he            : CKKS homomorphic encryption          (no DP; channel hidden)
    secagg        : secure aggregation masks             (no DP; aggregate only)
    dp_secagg     : Local-DP + SecAgg                    (nominal eps = E*sqrt(K))
    dp_he         : Local-DP + CKKS-HE aggregation       (nominal eps = E*sqrt(K))
    dp_secagg_he  : Local-DP + SecAgg + CKKS             (nominal eps = E*sqrt(K))
    secagg_he     : SecAgg + CKKS                        (no DP; channel hidden)
    ceiling       : nothing (reference, eps=inf)

"Relaxation": ANY stack that hides individual updates from the server (SecAgg
and/or HE) removes the per-client channel, so the same DP accounting derived
for secure aggregation applies — DP noise can be run at nominal E*sqrt(K)
while keeping *effective* per-user privacy E.

Run (from D:\\seminar\\website):
    .venv\\Scripts\\python.exe backend\\run_ncf_sweep.py

Outputs: console tables, attack evaluation, data/ncf_sweep_results.json,
         data/ncf_attack_results.csv
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from attacks import run_attacks
from database import cursor
from he_fl import HE, STACK_MECH
from ncf_fl import DELTA, NCF, NCFConfig, NCFRecommender

log = logging.getLogger("run_ncf_sweep")
logging.basicConfig(level=logging.INFO)

OUT = Path(__file__).resolve().parent.parent / "data" / "ncf_sweep_results.json"

STACKS = ["dp", "dp_secagg", "dp_he", "dp_secagg_he", "he", "secagg", "secagg_he"]
RELAX = ["dp_secagg", "dp_he", "dp_secagg_he"]  # DP + secure aggregation


# ---------------------------------------------------------------- data

def fetch_pool(cfg: NCFConfig, need: int) -> list[int]:
    with cursor() as (cur, _):
        rows = cur.execute(
            "SELECT userId FROM ratings GROUP BY userId HAVING COUNT(*) >= ? "
            "ORDER BY COUNT(*) DESC LIMIT 1000",
            (need,),
        ).fetchall()
    return [r["userId"] for r in rows]


def user_ratings(user_id: int, cap: int, offset: int) -> list[tuple[int, float]]:
    with cursor() as (cur, _):
        rows = cur.execute(
            "SELECT movieId, rating FROM ratings WHERE userId = ? "
            "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (user_id, cap, offset),
        ).fetchall()
    return [(int(r["movieId"]), float(r["rating"])) for r in rows]


def assemble_data(cfg: NCFConfig):
    cap_test = max(5, int(round(cfg.per_client_cap * cfg.test_frac / (1 - cfg.test_frac))))
    pool = fetch_pool(cfg, cfg.per_client_cap + cap_test)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(pool)
    clients = pool[: cfg.clients]
    demo = pool[cfg.clients: cfg.clients + 2]

    ids = sorted(set(clients) | set(demo))
    uid2row = {u: i for i, u in enumerate(ids)}

    item_set: set[int] = set()
    train: dict[int, list[tuple[int, float]]] = {}
    test: dict[int, list[tuple[int, float]]] = {}

    for u in ids:
        tr = user_ratings(u, cfg.per_client_cap, 0)
        te = user_ratings(u, cap_test, cfg.per_client_cap)
        if len(tr) < max(15, cfg.min_ratings):
            continue
        train[u] = tr
        test[u] = te
        item_set.update(mid for mid, _ in tr)
        item_set.update(mid for mid, _ in te)

    covered = sorted(item_set)
    coverage = {m: i for i, m in enumerate(covered)}
    item_rows = {u: np.array([coverage[m] for m, _ in train[u]]) for u in train}
    rat_rows = {u: np.array([r for _, r in train[u]], dtype=float) for u in train}
    test_rows = {}
    test_rat = {}
    for u in test:
        if test[u]:
            row = uid2row[u]
            test_rows[row] = np.array([coverage[m] for m, _ in test[u]])
            test_rat[row] = np.array([r for _, r in test[u]], dtype=float)

    return uid2row, covered, coverage, item_rows, rat_rows, test_rows, test_rat, demo


# ---------------------------------------------------------------- metrics

def ranking_metrics(model: NCF, cfg: NCFConfig, test_rows: dict[int, np.ndarray],
                    test_rat: dict[int, np.ndarray], num_items: int,
                    seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    hr, ndcg, mae, rmse, n = [], [], [], [], 0
    for uid, items_u in test_rows.items():
        if items_u is None or len(items_u) == 0:
            continue
        ratings_u = test_rat[uid]
        uu = np.full(len(items_u), uid, dtype=np.int64)
        pred = np.clip(model.predict(uu, items_u), 0.5, 5.0)
        mae.append(np.mean(np.abs(pred - ratings_u)))
        rmse.append(np.mean((pred - ratings_u) ** 2))
        n += len(items_u)
        cand = np.concatenate([items_u, rng.choice(num_items, cfg.eval_candidates, replace=False)])
        uu2 = np.full(len(cand), uid, dtype=np.int64)
        sc = model.predict(uu2, cand)
        order = np.argsort(-sc)[:10]
        hits = cand[order][np.isin(cand[order], items_u)]
        hr.append(len(hits) / 10.0)
        if len(hits):
            pos = []
            for h in hits:
                where = np.where(cand[order] == h)[0]
                if len(where):
                    pos.append(int(where[0]) + 1)
            dcg = sum(1.0 / math.log2(p + 1) for p in pos)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(pos), 10)))
            ndcg.append(dcg / max(idcg, 1e-12))
    return {
        "mae": round(float(np.mean(mae)), 4),
        "rmse": round(float(math.sqrt(np.mean(rmse))), 4),
        "hr@10": round(float(np.mean(hr)), 4),
        "ndcg@10": round(float(np.mean(ndcg)), 4) if ndcg else 0.0,
        "test_pairs": n,
    }


# ---------------------------------------------------------------- federated loop

def run_one(cfg, uid2row, covered, item_rows, rat_rows, mech: dict,
            eps: float | None, rnd_seed: int) -> NCF:
    """FedAvg rounds respecting the mechanism stack.

    mech: {'dp': bool, 'secagg': bool, 'he': bool}
    Sequence per client per round:
        clean clipped-mean gradient -> [Gaussian noise if dp]
                                      -> [pairwise mask if secagg]
                                      -> [CKKS encrypt if he]
    Server: masks cancel in the sum; if he -> homomorphic sum then collective
            decrypt; else plaintext mean.  FedAvg update on the decrypted avg.
    """
    num_items = len(covered)
    model = NCF(len(uid2row), num_items, cfg)
    cfg.eps_now = eps if mech["dp"] else None
    rng = np.random.default_rng(rnd_seed)
    used = [u for u in item_rows if len(item_rows[u]) >= cfg.batch]
    if len(used) < 2:
        raise RuntimeError("too few clients with enough ratings for the batch size")
    K = len(used)
    he = HE(model.shared_size) if mech["he"] else None

    for r in range(cfg.rounds):
        g_shared, g_rows = [], {}
        for u in used:
            nrow, rat = item_rows[u], rat_rows[u]
            if len(nrow) <= cfg.batch:
                idx = np.arange(len(nrow))
            else:
                idx = rng.choice(len(nrow), cfg.batch, replace=False)
            uu = np.full(len(idx), uid2row[u], dtype=np.int64)
            g, gp = model.shared_grad(uu, nrow[idx], rat[idx], cfg, rng)
            g_shared.append(g)
            g_rows[u] = gp
        if mech["secagg"] and K > 1:
            g_shared = _secagg_masks(g_shared, rnd_seed + r)
        if mech["he"]:
            cts = [he.encrypt(g) for g in g_shared]
            agg_ct = he.sum_all(cts)
            final = he.decrypt(agg_ct) / K
        else:
            final = np.mean(np.stack(g_shared), axis=0)
        model.apply_shared(final, cfg)
        for u, gp in g_rows.items():
            model.apply_user(uid2row[u], gp, cfg)
    return model


def _secagg_masks(updates, seed):
    K, dim = len(updates), len(updates[0])
    rng = np.random.default_rng(seed)
    masks = [np.zeros(dim) for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            m = rng.normal(0, 1, size=dim)
            masks[i] += m
            masks[j] -= m
    return [u + masks[i] for i, u in enumerate(updates)]


def sigma_at(cfg, batch_n: int, eps: float | None) -> float:
    if eps is None:
        return 0.0
    return 2.0 * cfg.clip_norm * math.sqrt(2 * math.log(1.25 / DELTA)) / (batch_n * eps)


# ---------------------------------------------------------------- main

def main() -> None:
    cfg = NCFConfig()

    log.info("sampling users + ratings from movielens.db ...")
    t0 = time.time()
    uid2row, covered, _cov, item_rows, rat_rows, test_rows, test_rat, demo = assemble_data(cfg)
    log.info("users=%d items=%d train_pairs=%d (%.1fs)",
             len(uid2row), len(covered), sum(len(v) for v in item_rows.values()),
             time.time() - t0)
    num_items = len(covered)
    demo_batch_n = int(np.mean([min(cfg.batch, len(v)) for v in item_rows.values()]))
    K = len([u for u in item_rows if len(item_rows[u]) >= cfg.batch])
    ampl = math.sqrt(max(1, K))
    log.info("clients in training=%d, SecAgg/HE amplification ~ sqrt(K)=%.2f", K, ampl)

    targets = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    rows: list[dict] = []
    models_reg: dict = {}

    def do_run(stack, nominal_eps, eff_eps, seed_base):
        mech = STACK_MECH[stack]
        model = run_one(cfg, uid2row, covered, item_rows, rat_rows, mech,
                        nominal_eps, seed_base)
        m = ranking_metrics(model, cfg, test_rows, test_rat, num_items, seed=7)
        rows.append({
            "stack": stack,
            "mechanisms": "+".join(k for k, v in mech.items() if v) or "none",
            "nominal_eps": round(nominal_eps, 3) if nominal_eps is not None else None,
            "sigma": round(sigma_at(cfg, demo_batch_n, nominal_eps), 4),
            "effective_epsilon": round(eff_eps, 3) if eff_eps is not None else None,
            **m,
        })
        models_reg[f"{stack}@{nominal_eps if nominal_eps is not None else 'inf'}"] = model
        return m

    for stack in STACKS:
        mech = STACK_MECH[stack]
        t = time.time()
        if mech["dp"]:
            for tgt in targets:
                nominal = tgt * ampl if stack in RELAX else tgt
                m = do_run(stack, nominal, tgt, round(nominal * 1000))
                log.info("%-14s nominal=%5.2f eff=%4.2f mae=%.4f hr=%.3f (%.0fs)",
                         stack, nominal, tgt, m["mae"], m["hr@10"], time.time() - t)
                t = time.time()
        else:
            m = do_run(stack, None, None, 424200 + STACKS.index(stack))
            log.info("%-14s (no DP) mae=%.4f hr=%.3f (%.0fs)",
                     stack, m["mae"], m["hr@10"], time.time() - t)

    # ---------------- ceiling: plaintext FI3 aggregate, no mechanisms
    ceil_model = run_one(cfg, uid2row, covered, item_rows, rat_rows,
                         {"dp": False, "secagg": False, "he": False}, None, 111)
    ceil_metrics = ranking_metrics(ceil_model, cfg, test_rows, test_rat, num_items, seed=7)
    ceiling_row = {"stack": "ceiling", "mechanisms": "none",
                   "nominal_eps": None, "sigma": 0.0, "effective_epsilon": None,
                   **ceil_metrics}
    models_reg["ceiling@inf"] = ceil_model
    rows.append(ceiling_row)

    # ---------------- relaxation verdict (fixed-effective-privacy framing)
    verdict = _check_relaxation(rows)

    # ---------------- attack evaluation on every trained model
    attacks_res = run_attacks(cfg, uid2row, covered, item_rows, rat_rows,
                              test_rows, test_rat, models_reg, rows)

    # ---------------- demo recommendations (ceiling model)
    demo_recs = _demo_recommendations(ceil_model, uid2row, covered, item_rows, demo)

    # ---------------- print
    print("\n" + "=" * 104)
    print("FedAvg + NCF on MovieLens | users=%d items=%d pairs=%d | K=%d amp=%.2f "
          "| delta=%.0e clip=%.1f batch=%d lr=%.2f rounds=%d"
          % (len(uid2row), len(covered), sum(len(v) for v in item_rows.values()),
             K, ampl, DELTA, cfg.clip_norm, cfg.batch, cfg.lr, cfg.rounds))
    print("=" * 104)
    hdr = (f"{'stack':<15}{'mechanisms':<16}{'nominal':>8}{'sigma':>8}{'eff_eps':>8}"
           f"{'MAE':>8}{'RMSE':>8}{'HR@10':>8}{'NDCG@10':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["stack"] == "ceiling":
            continue
        eps = f"{r['nominal_eps']:.3f}" if r["nominal_eps"] is not None else "inf"
        sig = f"{r['sigma']:.4f}"
        eff = f"{r['effective_epsilon']:.3f}" if r["effective_epsilon"] is not None else "inf"
        print(f"{r['stack']:<15}{r['mechanisms']:<16}{eps:>8}{sig:>8}{eff:>8}"
              f"{r['mae']:>8.3f}{r['rmse']:>8.3f}{r['hr@10']:>8.3f}{r['ndcg@10']:>9.3f}")
    print(f"{'ceiling':<15}{'none':<16}{'inf':>8}{'0.0000':>8}{'inf':>8}"
          f"{ceiling_row['mae']:>8.3f}{ceiling_row['rmse']:>8.3f}"
          f"{ceiling_row['hr@10']:>8.3f}{ceiling_row['ndcg@10']:>9.3f}")

    print("\n" + "=" * 104)
    print(_verdict_text(verdict))
    print("=" * 104)

    _print_attacks(attacks_res, rows)

    all_ids = [m for rec in demo_recs.values() for m, _ in rec]
    titles = movie_titles(all_ids)
    print("\nDemo recommendations (federated NCF, ceiling model):")
    for u, recs in demo_recs.items():
        print(f"  user {u}:")
        for mid, sc in recs:
            print(f"    {sc:+.3f}  ({mid}) {titles.get(mid, '?')}")

    out_rows = [{k: v for k, v in r.items()} for r in rows]
    results = {
        "config": {k: v for k, v in vars(cfg).items() if k != "eps_now"},
        "amplification": ampl, "k_clients": K,
        "stacks": STACKS,
        "ceiling": ceiling_row, "rows": out_rows,
        "verdict": verdict, "demo": demo_recs,
        "attacks": attacks_res,
    }
    OUT.write_text(json.dumps(results, default=str, indent=1))
    _write_attack_csv(attacks_res, rows)
    log.info("results saved -> %s", OUT)
    log.info("attack csv saved -> data/ncf_attack_results.csv")


# ---------------------------------------------------------------- verdict & reporting

def _base_rows(rows_meta):
    return {f"{r['stack']}@{r['nominal_eps'] if r['nominal_eps'] is not None else 'inf'}": r
            for r in rows_meta}


def _check_relaxation(rows, tol: float = 0.008) -> list[dict]:
    """Fixed-effective-privacy framing: for every stack carrying DP +
    secure aggregation (SecAgg or HE), the nominal E*sqrtK run should not be
    worse than the plain-DP run at the SAME effective epsilon E."""
    plain = {r["effective_epsilon"]: r for r in rows if r["stack"] == "dp"}
    checks = []
    for s in rows:
        if s["stack"] not in RELAX:
            continue
        eff = s["effective_epsilon"]
        p = plain.get(eff)
        if p is None:
            continue
        checks.append({
            "effective_eps": eff,
            "stack": s["stack"],
            "secagg_nominal": s["nominal_eps"], "plain_nominal": p["nominal_eps"],
            "mae_secagg": s["mae"], "mae_plain": p["mae"],
            "hr10_secagg": s["hr@10"], "hr10_plain": p["hr@10"],
            "acc_gain_pct": round((p["mae"] - s["mae"]) / max(p["mae"], 1e-9) * 100, 2),
            "meets_theory": s["mae"] <= p["mae"] + tol,
        })
    return checks


def _verdict_text(verdict) -> str:
    title = ("FIXED-PRIVACY FRAMING: DP + secure aggregation (SecAgg and/or HE) at "
             "nominal E*sqrt(K) vs plain Local-DP at E.\n"
             "Relaxation effect = accuracy recovered by adding sqrt(K)-times less "
             "noise at identical effective privacy.")
    if not verdict:
        return title + "\n  (no comparable rows)"
    lines = []
    per = {}
    for c in verdict:
        per.setdefault(c["effective_eps"], []).append(c)
    for eff in sorted(per):
        for c in per[eff]:
            lines.append(
                f"  E={eff:<5}{c['stack']:<15} plain_mae={c['mae_plain']:.4f}  "
                f"secagg_mae={c['mae_secagg']:.4f}  gain={c['acc_gain_pct']:+.2f}%  "
                f"{'OK' if c['meets_theory'] else 'FAIL'}")
    ok = sum(1 for c in verdict if c["meets_theory"])
    lines.append(f"confirmed in {ok}/{len(verdict)} (stack, privacy-level) pairs")
    return title + "\n" + "\n".join(lines)


def _print_attacks(attacks_res, rows_meta) -> None:
    base = _base_rows(rows_meta)
    print("\n" + "=" * 104)
    print("ATTACK EVALUATION  | server-side adversary | MIA=AUC(tpr@1%fpr) "
          "| leak channel & reverse-engineering")
    print("=" * 104)
    hdr = (f"{'stack':<15}{'mech':<16}{'nominal':>8}{'MAE':>7}{'miaAUC':>8}"
           f"{'tpr01':>7}{'attrMAE':>8}{'ch':>12}{'recall':>8}{'recon':>8}")
    print(hdr)
    print("-" * len(hdr))
    for key, r in base.items():
        if r["stack"] == "ceiling":
            continue
        mia = attacks_res.get("mia", {}).get(key) or {}
        attr = attacks_res.get("attr", {}).get(key) or {}
        leak = attacks_res.get("leak", {}).get(key) or {}
        eps = f"{r['nominal_eps']:.3f}" if r["nominal_eps"] is not None else "inf"
        print(f"{r['stack']:<15}{r['mechanisms']:<16}{eps:>8}{r['mae']:>7.3f}"
              f"{mia.get('auc', 0.5):>8.3f}{mia.get('tpr01', 0):>7.3f}"
              f"{attr.get('mae_members', 0):>8.3f}{leak.get('channel', '-'):>12}"
              f"{leak.get('recall_q', 0):>8.3f}{leak.get('recon_mae', 0):>8.3f}")
    print()


def _write_attack_csv(attacks_res, rows_meta) -> None:
    base = _base_rows(rows_meta)
    path = Path(__file__).resolve().parent.parent / "data" / "ncf_attack_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stack", "mechanisms", "nominal_eps", "eff_eps", "utility_mae",
                    "hr_at10", "mia_auc", "mia_tpr_at_1pct_fpr", "attr_mae_members",
                    "leak_channel", "leak_batch_recall", "leak_recon_mae"])
        for key, r in base.items():
            mia = attacks_res.get("mia", {}).get(key) or {}
            attr = attacks_res.get("attr", {}).get(key) or {}
            leak = attacks_res.get("leak", {}).get(key) or {}
            w.writerow([
                r["stack"], r["mechanisms"],
                r["nominal_eps"] if r["nominal_eps"] is not None else "inf",
                r["effective_epsilon"] if r["effective_epsilon"] is not None else "inf",
                r["mae"], r["hr@10"],
                mia.get("auc", ""), mia.get("tpr01", ""), attr.get("mae_members", ""),
                leak.get("channel", ""), leak.get("recall_q", ""), leak.get("recon_mae", ""),
            ])
    log.info("attack csv saved -> %s", path)


# ---------------------------------------------------------------- demo

def _demo_recommendations(model, uid2row, covered, item_rows, demo) -> dict[int, list]:
    ni = len(covered)
    ids = np.arange(ni, dtype=np.int64)
    rev = {i: int(covered[i]) for i in range(ni)}
    rated = {}
    for u in uid2row:
        if u in item_rows:
            rated[uid2row[u]] = {int(i) for i in item_rows[u]}
    rec = NCFRecommender(model, ids, rated)
    out = {}
    for u in demo:
        if u not in uid2row or uid2row[u] not in rated:
            continue
        score = rec.recommend(uid2row[u], topk=6)
        out[u] = [[rev[m], round(s, 3)] for m, s in score]
    return out


def movie_titles(movie_ids: list[int]) -> dict[int, str]:
    if not movie_ids:
        return {}
    ph = ",".join("?" for _ in movie_ids)
    with cursor() as (cur, _):
        rows = cur.execute(
            f"SELECT movieId, title FROM movies WHERE movieId IN ({ph})",
            movie_ids,
        ).fetchall()
    return {r["movieId"]: r["title"] for r in rows}


if __name__ == "__main__":
    main()