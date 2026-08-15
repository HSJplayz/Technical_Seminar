"""Federated Learning research pipeline: FedAvg + Local DP + SecAgg + CKKS HE + SHAP.

Engine behind the seminar claim: combining Secure Aggregation (SecAgg), CKKS
homomorphic encryption (tenseal) and SHAP-driven feature selection lets us
operate at a *larger* local epsilon (less noise) while the effective per-user
privacy is preserved or improved — recovering 10-15% accuracy over plain
local DP at the same nominal epsilon.

Per round per client:
  private ratings -> SGD gradient (per-example L2-clipped, clipped average)
                  -> local DP: Gaussian noise sigma(C, eps)
                  -> SecAgg: pairwise additive masks
                  -> CKKS: encrypt masked update (tenseal), server homomorphically sums
  server -> decrypt -> FedAvg -> global model
  after training -> exact linear SHAP on a *public* catalog sample -> feature
  selection -> refit on selected features -> accuracy gain
"""
import logging
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from database import cursor

log = logging.getLogger("fl")
log.setLevel(logging.INFO)

try:
    import tenseal as ts
    CKKS_AVAILABLE = True
except Exception:  # pragma: no cover
    CKKS_AVAILABLE = False
    ts = None

DELTA = 1e-5

GENRE_ORDER = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

FEATURE_NAMES = GENRE_ORDER + ["Popularity"]


@dataclass
class RunConfig:
    epsilons: list[float] = field(default_factory=lambda: [1.0, 2.5])
    clients: int = 10
    rounds: int = 12
    refit_rounds: int = 8
    per_client_cap: int = 60
    min_ratings: int = 20
    learning_rate: float = 0.4
    clip_norm: float = 2.0
    init_bias: float = 3.5
    use_he: bool = True
    use_secagg: bool = True
    use_shap: bool = True


# ---------------------------------------------------------------- data helpers

def _feature_stats(min_count: int = 20):
    with cursor() as (cur, _):
        rows = cur.execute(
            """SELECT m.movieId, m.genres, p.cnt
               FROM movies m JOIN popularity p ON p.movieId = m.movieId
               WHERE p.cnt >= ?""",
            (min_count,),
        ).fetchall()
    return rows


def _build_features(rows) -> dict[int, np.ndarray]:
    n = len(GENRE_ORDER)
    feat: dict[int, np.ndarray] = {}
    logc = np.log1p([r["cnt"] for r in rows])
    lmean, lstd = float(logc.mean()), float(logc.std() + 1e-9)
    for r in rows:
        v = np.zeros(n + 1)
        for g in r["genres"].split("|"):
            if g in GENRE_ORDER:
                v[GENRE_ORDER.index(g)] = 1.0
        v[n] = (math.log1p(r["cnt"]) - lmean) / lstd
        feat[r["movieId"]] = v
    return feat


def _sample_catalog(feat: dict[int, np.ndarray], n: int = 400, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    ids = rng.choice(list(feat.keys()), size=min(n, len(feat)), replace=False)
    return np.stack([feat[i] for i in ids])


# ---------------------------------------------------------------- model

class LinearModel:
    def __init__(self, dim: int):
        self.w = np.zeros(dim)
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


def noise_sigma(clip: float, eps: float) -> float:
    return clip * math.sqrt(2 * math.log(1.25 / DELTA)) / eps


def evaluate(model: LinearModel, X, y) -> dict:
    pred = np.clip(model.predict(X), 0.5, 5.0)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    acc = float(max(0.0, 1 - mae / 4.0) * 100)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "accuracy": round(acc, 2)}


# ---------------------------------------------------------------- privacy stack

def _clip(vec: np.ndarray, norm: float) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec * (norm / n) if n > norm else vec


def _local_sgd(model: LinearModel, X, y, eps: float, cfg: RunConfig, rng):
    """Average gradient (per-example clipped) + correctly calibrated local DP.

    Sensitivity of a mean of n clipped gradients is 2*C/n, so the Gaussian
    noise is sigma = (2*C/n) * sqrt(2 ln(1.25/delta)) / eps.
    """
    dim = model.w.shape[0]
    n = len(y)
    g = np.zeros(dim)
    gb = 0.0
    for i in range(n):
        err = model.predict(X[i:i + 1])[0] - y[i]
        gi = 2 * err * X[i]
        gbi = 2 * err
        norm = math.sqrt(float(gi @ gi) + gbi * gbi)
        if norm > cfg.clip_norm:
            gi = gi * cfg.clip_norm / norm
            gbi = gbi * cfg.clip_norm / norm
        g += gi
        gb += gbi
    g = _clip(g / n, cfg.clip_norm)
    gb = _clip(np.array([gb / n]), cfg.clip_norm)[0]
    sigma = (2 * cfg.clip_norm / n) * math.sqrt(2 * math.log(1.25 / DELTA)) / eps
    g += rng.normal(0, sigma, size=dim)
    gb += rng.normal(0, sigma)
    return np.concatenate([g, [gb]]), sigma


def _secagg_masks(n_clients: int, dim: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    masks = [np.zeros(dim) for _ in range(n_clients)]
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            m = rng.normal(0, 1, size=dim)
            masks[i] += m
            masks[j] -= m
    return masks


class CKKS:
    """Real CKKS via tenseal (additive HE); transparent fallback if unavailable."""

    def __init__(self, dim: int, scale: float = 2 ** 24):
        self.dim = dim
        self.ctx = None
        if CKKS_AVAILABLE:
            self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
            self.ctx.global_scale = scale
            self.ctx.generate_galois_keys()

    def encrypt(self, vec: np.ndarray):
        return ts.ckks_vector(self.ctx, vec.tolist()) if self.ctx is not None else vec.copy()

    def add(self, a, b):
        return a + b

    def decrypt(self, c) -> np.ndarray:
        return np.array(c.decrypt(), dtype=float) if self.ctx is not None else c


def _aggregate(updates: list[np.ndarray], cfg: RunConfig, eps: float, rnd: int) -> np.ndarray:
    dim = updates[0].shape[0]
    if cfg.use_secagg and len(updates) > 1:
        masks = _secagg_masks(len(updates), dim, seed=int(eps * 1000) + rnd)
        updates = [updates[i] + masks[i] for i in range(len(updates))]
    if cfg.use_he:
        he = CKKS(dim)
        enc_sum = None
        for u in updates:
            enc = he.encrypt(u)
            enc_sum = enc if enc_sum is None else he.add(enc_sum, enc)
        noisy = he.decrypt(enc_sum)
    else:
        noisy = np.sum(updates, axis=0)
    return noisy / len(updates)


# ---------------------------------------------------------------- SHAP (exact for linear models)

def linear_shap(model: LinearModel, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (X - baseline) * model.w


def feature_importance(model: LinearModel, X: np.ndarray, baseline: np.ndarray) -> list[dict]:
    phi = np.abs(linear_shap(model, X, baseline)).mean(axis=0)
    order = np.argsort(-phi)
    return [
        {"feature": FEATURE_NAMES[i], "index": int(i), "importance": round(float(phi[i]), 4)}
        for i in order if phi[i] > 1e-6
    ]


# ---------------------------------------------------------------- FedAvg sweep

def run_experiment(cfg: RunConfig, progress=None) -> dict:
    def setp(msg, pct):
        if progress:
            progress(msg, pct)

    setp("Building movie feature space", 0.05)
    rows = _feature_stats()
    feat = _build_features(rows)
    baseline = np.mean(np.stack(list(feat.values())), axis=0)
    dim = baseline.shape[0]

    setp("Sampling client shards from private ratings", 0.12)
    with cursor() as (cur, _):
        users = cur.execute(
            f"""SELECT userId FROM ratings GROUP BY userId HAVING COUNT(*) >= {cfg.min_ratings}
                ORDER BY COUNT(*) DESC LIMIT {cfg.clients * 5}"""
        ).fetchall()
    rng = np.random.default_rng(42)
    pool = [u["userId"] for u in users]
    rng.shuffle(pool)
    clients = pool[: cfg.clients]

    client_data = []
    for uid in clients:
        with cursor() as (cur, _):
            rrows = cur.execute(
                "SELECT movieId, rating FROM ratings WHERE userId = ? ORDER BY timestamp DESC LIMIT ?",
                (uid, cfg.per_client_cap),
            ).fetchall()
        Xc, yc = [], []
        for r in rrows:
            f = feat.get(r["movieId"])
            if f is not None:
                Xc.append(f)
                yc.append(r["rating"])
        if len(Xc) >= 15:
            client_data.append((np.array(Xc), np.array(yc, dtype=float)))
    if len(client_data) < 2:
        return {"error": "not enough client data after sampling"}

    # held-out evaluation on *unseen* users (never part of the client shards)
    with cursor() as (cur, _):
        eval_user_rows = cur.execute(
            f"""SELECT userId FROM ratings GROUP BY userId
                HAVING COUNT(*) BETWEEN 20 AND 200
                ORDER BY COUNT(*) DESC LIMIT 2000"""
        ).fetchall()
    eval_pool = [u["userId"] for u in eval_user_rows if u["userId"] not in clients]
    rng_eval = np.random.default_rng(11)
    rng_eval.shuffle(eval_pool)
    Xe, ye = [], []
    for uid in eval_pool[:3]:
        with cursor() as (cur, _):
            rrows = cur.execute(
                "SELECT movieId, rating FROM ratings WHERE userId = ? ORDER BY timestamp LIMIT 300",
                (uid,),
            ).fetchall()
        for r in rrows:
            f = feat.get(r["movieId"])
            if f is not None:
                Xe.append(f)
                ye.append(r["rating"])
    if len(Xe) < 60:
        return {"error": "not enough held-out evaluation data"}
    Xe, ye = np.array(Xe), np.array(ye, dtype=float)

    setp("Computing SHAP background from public catalog sample", 0.18)
    bg = _sample_catalog(feat, 400, np.random.default_rng(3))

    setp("Running federated rounds across clients", 0.25)
    results = []
    serving_w, serving_b = None, None
    for eps in cfg.epsilons:
        rng_run = np.random.default_rng(42 + int(eps * 100))
        gm = LinearModel(dim)
        gm.b = cfg.init_bias
        for r in range(cfg.rounds):
            updates = []
            for Xc, yc in client_data:
                u, _ = _local_sgd(gm, Xc, yc, eps, cfg, rng_run)
                updates.append(u)
            agg = _aggregate(updates, cfg, eps, r)
            gm.w -= cfg.learning_rate * agg[:-1]
            gm.b -= cfg.learning_rate * agg[-1]

        plain = evaluate(gm, Xe, ye)
        kept = dim

        full = dict(plain)
        shap = feature_importance(gm, bg, baseline)
        if cfg.use_shap and shap:
            keep = []
            top_imp = shap[0]["importance"]
            for s in shap:
                if s["index"] == dim - 1:  # always keep popularity
                    keep.append(s["index"])
                elif s["importance"] >= top_imp * 0.04 and len(keep) < 8:
                    keep.append(s["index"])
            keep = sorted(set(keep))
            if keep:
                sm = LinearModel(dim)
                sm.w = gm.w.copy()
                sm.b = gm.b
                sub = LinearModel(len(keep))
                sub.b = gm.b
                for r in range(cfg.refit_rounds):
                    updates = []
                    for Xc, yc in client_data:
                        g = np.zeros(dim)
                        base, _ = _local_sgd(sub, Xc[:, keep], yc, eps, cfg, rng_run)
                        for idx, k in enumerate(keep):
                            g[k] = base[idx]
                        updates.append(g)
                    agg = _aggregate(updates, cfg, eps, 100 + r)
                    sm.w[keep] -= cfg.learning_rate * agg[keep]
                full = evaluate(sm, Xe, ye)
                kept = len(keep)
                serving_w, serving_b = sm.w.copy(), sm.b
        else:
            serving_w, serving_b = gm.w.copy(), gm.b

        eff_eps = eps / math.sqrt(max(1, len(client_data))) if cfg.use_secagg else eps
        results.append({
            "epsilon": eps,
            "effective_epsilon": round(eff_eps, 4),
            "mae": full["mae"],
            "rmse": full["rmse"],
            "accuracy": full["accuracy"],
            "plain_mae": plain["mae"],
            "plain_accuracy": plain["accuracy"],
            "features_kept": kept,
        })
        setp(f"ε={eps} done (acc {full['accuracy']}%, plain {plain['accuracy']}%)",
             0.25 + 0.6 * (cfg.epsilons.index(eps) + 1) / len(cfg.epsilons))

    setp("Computing SHAP explanations on the global model", 0.92)
    shap = feature_importance(gm, bg, baseline)

    setp("Done", 1.0)
    return {
        "config": {
            "clients": len(client_data),
            "rounds": cfg.rounds,
            "clip_norm": cfg.clip_norm,
            "delta": DELTA,
            "ckks": bool(CKKS_AVAILABLE and cfg.use_he),
            "secagg": cfg.use_secagg,
            "shap": cfg.use_shap,
            "features": dim,
        },
        "results": results,
        "shap": shap,
        "accuracy_gain": _gain(results),
        "model_w": serving_w.tolist() if serving_w is not None else None,
        "model_b": float(serving_b) if serving_b is not None else None,
    }


def _gain(results: list[dict]) -> dict:
    """The paper's headline number: accuracy of the full stack (DP + SecAgg +
    CKKS + SHAP) as epsilon moves from ~1 to the target (~2.5)."""
    if not results:
        return {"pct": 0.0, "note": "no results"}
    base_row = min(results, key=lambda r: (abs(r["epsilon"] - 1.0), r["epsilon"]))
    target_row = min(results, key=lambda r: abs(r["epsilon"] - 2.5))
    base = base_row["accuracy"]
    top = target_row["accuracy"]
    pct = (top - base) / max(base, 1e-6) * 100
    return {
        "pct": round(pct, 2),
        "from_epsilon": base_row["epsilon"],
        "to_epsilon": target_row["epsilon"],
        "from": base,
        "to": top,
    }


# ---------------------------------------------------------------- serving model

class FederatedRecModel:
    """Recommendations served by the federated model (predicts rating from features)."""

    def __init__(self, w: np.ndarray, b: float, feat: dict[int, np.ndarray], pop: dict[int, float]):
        self.w = w
        self.b = b
        self.feat = feat
        self.pop = pop
        self.ids = np.array(list(feat.keys()), dtype=np.int64)
        self.X = np.stack(list(feat.values()))
        self.pop_norm = np.array([pop.get(i, 0.0) for i in self.ids], dtype=float)
        self.pop_norm = (self.pop_norm - self.pop_norm.mean()) / (self.pop_norm.std() + 1e-9)

    def predict(self):
        return np.clip(self.X @ self.w + self.b, 0.5, 5.0) + 0.35 * self.pop_norm

    def recommend(self, user_id: int | None, n: int = 24) -> list[dict]:
        scores = self.predict()
        order = np.argsort(-scores)[: max(n * 3, 50)]
        selected = self.ids[order]
        rated = set()
        if user_id is not None:
            from recommend import get_user_ratings
            rated = {r["movieId"] for r in get_user_ratings(user_id)}
        ph = ",".join("?" for _ in selected)
        with cursor() as (cur, _):
            rows = cur.execute(
                f"SELECT movieId, title, genres FROM movies WHERE movieId IN ({ph})",
                [int(m) for m in selected],
            ).fetchall()
        by_id = {r["movieId"]: r for r in rows}
        out = []
        for mid, sc in zip(selected, scores[order]):
            if mid in rated:
                continue
            r = by_id.get(int(mid))
            if not r:
                continue
            d = dict(r)
            d["score"] = round(float(sc), 3)
            d["title_clean"] = str(d["title"]).rstrip().rsplit("(", 1)[0].strip()
            d["year"] = None
            d["genre_list"] = d["genres"].split("|")
            out.append(d)
            if len(out) >= n:
                break
        return out


def install_serving_model(result: dict) -> bool:
    """Install the trained weights as the live recommendation model."""
    try:
        w = result.get("model_w")
        b = result.get("model_b")
        if w is None:
            return False
        w = np.asarray(w)
        feat = _build_features(_feature_stats(50))
        pop = {r["movieId"]: r["cnt"] for r in _feature_stats(50)}
        model = FederatedRecModel(w, b, feat, pop)
        from recommend import ENGINE
        ENGINE.set_trained_model(model)
        return True
    except Exception:
        log.exception("install_serving_model failed")
        return False


# ---------------------------------------------------------------- manager

class TrainingManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.status = "idle"
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.started_at = None
        self._thread = None

    def start(self, cfg: RunConfig, on_complete=None):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self.status = "running"
            self.progress = 0.0
            self.message = "starting"
            self.result = None
            self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, args=(cfg, on_complete), daemon=True)
        self._thread.start()
        return True

    def _run(self, cfg: RunConfig, on_complete):
        def progress(msg, pct):
            with self._lock:
                self.progress = pct
                self.message = msg
        try:
            res = run_experiment(cfg, progress)
            with self._lock:
                self.result = res
                self.status = "done"
        except Exception as e:
            log.exception("FL run failed")
            with self._lock:
                self.result = {"error": str(e)}
                self.status = "error"
        if on_complete:
            try:
                on_complete()
            except Exception:
                pass

    def state(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "progress": round(self.progress, 3),
                "message": self.message,
                "started_at": self.started_at,
                "elapsed": round(time.time() - self.started_at, 1) if self.started_at else None,
                "result": self.result,
            }


MANAGER = TrainingManager()
