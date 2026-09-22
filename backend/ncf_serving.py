"""Serving pipeline for the federated NCF model + user-local SHAP explanations.

Two new website features:
  1. "Recommend"  — a global NCF (two-layer MLP over item embeddings) trained
     federatedly (FedAvg + local DP) on MovieLens ratings, then personalized
     for a site user using ONLY that user's own ratings.
  2. "Explain with SHAP" — hand-rolled permutation SHAP over the 19 genre
     features of an item. The explanation path consumes only:
        * the requesting user's own ratings (to fit their personal vector P_u),
        * the public movie catalog (genres/titles),
        * the global federated model parameters (Q is derived via a public
          genre->embedding ridge map A = lstsq(G, Q); nothing user-specific).
     No other user's rating data is ever read while explaining.

Privacy invariant: the personalized vector P_u is a function of the site user's
own ratings only, and is recomputed per request; it is never persisted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from config import DATA_DIR
from database import cursor
from ncf_fl import NCF
from recommend import get_movie, get_user_ratings

log = logging.getLogger("ncf_serving")
log.setLevel(logging.INFO)

GENRE_ORDER = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
G = len(GENRE_ORDER)

# silhouette/personalisation knobs -------------------------------------------
MIN_FIT_RATINGS = 3          # below this we cannot SGD-fit P_u
FIT_ITERS = 60
FIT_LR = 0.04
CANDIDATE_POP_MIN = 30       # rank over popular catalog items
CANDIDATE_CAP = 8000
FIT_MLEN = 100               # max ratings used to fit P_u


def _one_hot(genres: str) -> np.ndarray:
    v = np.zeros(G)
    for g in genres.split("|"):
        if g in GENRE_ORDER:
            v[GENRE_ORDER.index(g)] = 1.0
    return v


class NCFFederatedServer:
    """Background trainer + read-only recommender / explainer."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self.status = "idle"
        self.progress = 0.0
        self.message = ""
        self.error = None
        self.started_at = None
        self.meta = None
        self._thread = None

        self.d = 0
        self.W1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.A = None
        self.g_base = None
        self.cat_ids = None            # (n_catalog,) int64 movieIds
        self.cat_emb = None            # (n_catalog, d) float32
        self.cand_emb = None           # (n_cand, d) float32 aligned with self.cand_ids
        self.cand_ids = None           # (n_cand,) int64
        self.pop_mean = None           # (d,) default P for cold-start users

    # ------------------------------------------------------------ public API

    def ensure(self) -> bool:
        if self.model_path.exists():
            try:
                self._load()
                return True
            except Exception:
                log.exception("failed to load NCF serving model; retraining")
        self.start()
        return False

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self.status = "training"
            self.progress = 0.0
            self.message = "starting federated NCF training"
            self.error = None
            self.started_at = time.time()
        self._thread = threading.Thread(target=self._train, daemon=True)
        self._thread.start()
        return True

    def state(self) -> dict:
        with self._lock:
            out = {
                "status": self.status,
                "progress": round(self.progress, 3),
                "message": self.message,
                "error": self.error,
                "elapsed": round(time.time() - self.started_at, 1) if self.started_at else None,
                "model": self.meta,
            }
        return out

    def ready(self) -> bool:
        return self.status == "ready" and self.W1 is not None

    # ------------------------------------------------------------ personalization

    def _user_embedding(self, user_id: int) -> tuple[np.ndarray, str]:
        """Fit P_u from *this user's own ratings* only. Returns (p, method)."""
        rated = get_user_ratings(user_id)
        pairs = []
        for r in rated[:FIT_MLEN]:
            idx = np.searchsorted(self.cat_ids, r["movieId"])
            if idx < len(self.cat_ids) and self.cat_ids[idx] == r["movieId"]:
                pairs.append((self.cat_emb[idx], float(r["rating"])))
        if not pairs:
            return self.pop_mean.copy(), "cold_start"
        emb = np.stack([e for e, _r in pairs])
        vals = np.array([r for _e, r in pairs], dtype=float)
        if len(pairs) >= MIN_FIT_RATINGS:
            p = self._fit_p(emb, vals)
            return p, "sgd_your_ratings"
        return emb.mean(axis=0), "mean_of_your_ratings"

    def _fit_p(self, emb: np.ndarray, vals: np.ndarray) -> np.ndarray:
        """Gradient descent on P_u only, freezing the global network."""
        n = len(vals)
        p = emb.mean(axis=0).copy()
        W1T = self.W1  # (h, 2d) direct
        left = W1T[:, : self.d]
        for _ in range(FIT_ITERS):
            x = np.concatenate([np.broadcast_to(p, (n, self.d)), emb], axis=1)
            z = x @ W1T.T + self.b1
            h = np.maximum(z, 0.0)
            pred = h @ self.w2 + self.b2
            err = 2.0 * (pred - vals) / n
            g_p = (err[:, None] * self.w2[None, :] * (z > 0)) @ left
            p = p - FIT_LR * g_p.mean(axis=0)
        return p

    # ------------------------------------------------------------ scoring

    def _score_batch(self, p: np.ndarray, emb: np.ndarray) -> np.ndarray:
        n = len(emb)
        x = np.concatenate([np.broadcast_to(p.astype(emb.dtype), (n, self.d)), emb], axis=1)
        z = x @ self.W1.T + self.b1
        h = np.maximum(z, 0.0)
        return h @ self.w2 + self.b2

    def _score_pair(self, p: np.ndarray, e: np.ndarray) -> float:
        x = np.concatenate([p, e])
        z = x @ self.W1.T + self.b1
        h = np.maximum(z, 0.0)
        return float(h @ self.w2 + self.b2)

    def recommend(self, user_id: int, k: int = 8) -> dict:
        p, method = self._user_embedding(user_id)
        rated = {r["movieId"] for r in get_user_ratings(user_id)}
        scores = self._score_batch(p, self.cand_emb)
        order = np.argsort(-scores)
        out = []
        for idx in order:
            mid = int(self.cand_ids[idx])
            if mid in rated:
                continue
            m = get_movie(mid)
            if m is None:
                continue
            m["score"] = round(float(scores[idx]), 3)
            out.append(m)
            if len(out) >= k:
                break
        return {"items": out, "personalized": method, "k": len(out)}

    # ------------------------------------------------------------ SHAP (local surrogate)

    def explain(self, user_id: int, movie_ids: list[int]) -> dict:
        """Explain each movie with a local linear surrogate over the 19 genre
        features, fit in the movie's genre neighbourhood using ONLY this user's
        personal vector P_u + the public catalog. Returns genre SHAP values of
        the surrogate (exact for linear models)."""
        p, method = self._user_embedding(user_id)
        gvecs = self._load_genres_full()
        rng = np.random.default_rng(2026)
        exps = []
        for mid in movie_ids:
            m = get_movie(mid)
            if m is None:
                continue
            idx = np.searchsorted(self.cat_ids, mid)
            if not (idx < len(self.cat_ids) and self.cat_ids[idx] == mid):
                continue
            g_i = gvecs[idx]
            phis, base_pred, proxy_pred = self._surrogate_attrs(p, idx, gvecs, rng)
            nonzero = [(GENRE_ORDER[i], v) for i, v in enumerate(phis) if abs(v) > 1e-6]
            nonzero.sort(key=lambda t: -abs(t[1]))
            top = nonzero[:6]
            z = sum(v for _, v in top)
            e = self.cat_emb[idx]
            exps.append({
                "movieId": int(mid),
                "title_clean": m["title_clean"],
                "year": m["year"],
                "poster_url": m["poster_url"],
                "genres": m["genre_list"],
                "score": round(self._score_pair(p, e), 3),
                "proxy_score": round(proxy_pred, 3),
                "base_score": round(base_pred, 3),
                "sum_attributions": round(float(sum(phis)), 3),
                "attributions": [
                    {"feature": feat, "importance": round(v, 3),
                     "share": round(v / z * 100, 1) if z else 0.0}
                    for feat, v in top
                ],
            })
        return {
            "explanations": exps,
            "personalized": method,
            "note": ("Attributions are SHAP values of a local linear surrogate over the 19 genre "
                     "features, fit in each movie's genre neighbourhood from YOUR ratings + the "
                     "public catalog + the global federated NCF model. No other user's data is used."),
        }

    def _surrogate_attrs(self, p, idx: int, gvecs: np.ndarray, rng, n_nb: int = 240) -> tuple:
        """Fit s_j ≈ g_j·w + c over a neighbourhood that mixes other movies sharing
        a genre with the target AND a sample of the popular catalog (so the genre
        signal is not degenerate). Returns (phis, base_pred, proxy_pred) with
        phis = w * (g_i - mean_g) — exact SHAP for the surrogate."""
        g_i = gvecs[idx]
        share = gvecs[:, g_i > 0].any(axis=1)
        same = self.cat_ids[share]
        same = rng.choice(same, size=min(n_nb // 2, len(same)), replace=False) if len(same) else same
        fill = np.setdiff1d(self.cand_ids, same)
        extra = rng.choice(fill, size=min(n_nb - len(same), len(fill)), replace=False)
        nb = np.concatenate([same, extra])
        pos = np.searchsorted(self.cat_ids, nb)
        valid = (pos < len(self.cat_ids)) & (self.cat_ids[pos] == nb)
        pos = pos[valid]
        if len(pos) < 20:
            base = self.cand_ids[: min(len(self.cand_ids), n_nb)]
            pos = np.searchsorted(self.cat_ids, base)
        y = self._score_batch(p, self.cat_emb[pos])
        Gn = gvecs[pos]
        design = np.concatenate([Gn, np.ones((len(Gn), 1))], axis=1)
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        w, c = params[:G], float(params[G])
        gm = Gn.mean(axis=0)
        base_pred = float(c + gm @ w)
        proxy_pred = float(c + g_i @ w)
        return (w * (g_i - gm)), base_pred, proxy_pred

    def _load_genres_full(self) -> np.ndarray:
        if getattr(self, "_gvecs", None) is not None:
            return self._gvecs
        with cursor() as (cur, _):
            rows = cur.execute("SELECT movieId, genres FROM movies ORDER BY movieId").fetchall()
        M = np.zeros((len(rows), G))
        for i, r in enumerate(rows):
            M[i] = _one_hot(r["genres"])
        self._gvecs = M
        return M

    # ------------------------------------------------------------ training

    def _train(self):
        t0 = time.time()
        try:
            self._setp("sampling federated clients from MovieLens", 0.05)
            clients, client_items, client_vals, item_ids = self._sample_clients()
            num_items = len(item_ids)
            if len(clients) < 2 or num_items < 50:
                raise RuntimeError("not enough federated client data to train NCF")
            cfg = _ServingCfg(num_clients=len(clients))
            model = NCF(len(clients), num_items, cfg)
            # plain SGD needs bigger-than-0.04 init to actually move the layers
            model.P *= cfg.init_scale
            model.Q *= cfg.init_scale
            model.W1 *= cfg.init_scale
            model.w2 *= cfg.init_scale
            cfg.eps_now = 1.0
            rng = np.random.default_rng(7)

            for r in range(cfg.rounds):
                shared = []
                user_rows = []
                for c in range(len(clients)):
                    s, ur = model.shared_grad(
                        np.repeat(c, len(client_items[c])), client_items[c], client_vals[c], cfg, rng
                    )
                    shared.append(s)
                    user_rows.append((c, ur))
                model.apply_shared(np.mean(shared, axis=0), cfg)
                for c, ur in user_rows:
                    model.apply_user(c, ur, cfg)
                self._setp(f"federated round {r + 1}/{cfg.rounds}", 0.1 + 0.55 * (r + 1) / cfg.rounds)

            self._setp("fitting genre->embedding bridge + catalog embeddings", 0.72)
            self.W1 = model.W1.astype(np.float32)
            self.b1 = model.b1.astype(np.float32)
            self.w2 = model.w2.astype(np.float32)
            self.b2 = float(model.b2)
            self.d = cfg.embed_dim
            self._bridge_and_catalog(item_ids, model.Q, rng)

            self._setp("persisting serving model", 0.95)
            self._save(cfg.gen_meta(len(clients)))
            with self._lock:
                self.status = "ready"
                self.progress = 1.0
                self.message = f"ready in {time.time() - t0:.1f}s "
                self.meta = cfg.gen_meta(len(clients))
                self.meta["catalog"] = int(len(self.cat_ids))
                self.meta["personalization"] = "user_local_only"
        except Exception as e:
            log.exception("NCF serving training failed")
            with self._lock:
                self.status = "error"
                self.error = str(e)
                self.message = f"training failed: {e}"

    def _setp(self, msg, pct):
        with self._lock:
            self.progress = pct
            self.message = msg

    def _sample_clients(self, cap: int = 500, n_clients: int = 24):
        with cursor() as (cur, _):
            rows = cur.execute(
                """SELECT userId FROM ratings GROUP BY userId
                   HAVING COUNT(*) >= 300 ORDER BY COUNT(*) DESC LIMIT ?""",
                (n_clients * 6,),
            ).fetchall()
        rng = np.random.default_rng(3)
        pool = [r["userId"] for r in rows]
        rng.shuffle(pool)
        clients = pool[:n_clients]
        item_ids: set[int] = set()
        per: list[tuple[np.ndarray, np.ndarray]] = []
        for uid in clients:
            with cursor() as (cur, _):
                rrows = cur.execute(
                    "SELECT movieId, rating FROM ratings WHERE userId = ? ORDER BY timestamp DESC LIMIT ?",
                    (uid, cap),
                ).fetchall()
            if len(rrows) < 15:
                continue
            mids = np.array([r["movieId"] for r in rrows], dtype=np.int64)
            vals = np.array([r["rating"] for r in rrows], dtype=float)
            item_ids.update(mids.tolist())
            per.append((mids, vals))
        item_ids = sorted(item_ids)
        idx_map = {m: i for i, m in enumerate(item_ids)}
        client_items = [np.array([idx_map[m] for m in mids], dtype=np.int64) for mids, _ in per]
        client_vals = [vals for _, vals in per]
        clients = clients[: len(per)]
        return clients, client_items, client_vals, np.array(item_ids, dtype=np.int64)

    def _bridge_and_catalog(self, item_ids: np.ndarray, Q: np.ndarray, rng) -> None:
        """Build A = lstsq(G', Q) and the full catalog embedding matrix."""
        gmat = np.zeros((len(item_ids), G))
        with cursor() as (cur, _):
            rows = cur.execute(
                f"SELECT movieId, genres FROM movies WHERE movieId IN ({','.join('?' for _ in item_ids)})",
                item_ids.tolist(),
            ).fetchall()
            by_id = {r["movieId"]: r["genres"] for r in rows}
        for i, m in enumerate(item_ids):
            gmat[i] = _one_hot(by_id.get(int(m), ""))
        self.A = np.linalg.lstsq(gmat, np.asarray(Q, dtype=float), rcond=None)[0].astype(np.float32)

        with cursor() as (cur, _):
            cats = cur.execute(
                "SELECT movieId, genres FROM movies ORDER BY movieId"
            ).fetchall()
            cands = cur.execute(
                "SELECT movieId FROM popularity WHERE cnt >= ? ORDER BY score DESC LIMIT ?",
                (CANDIDATE_POP_MIN, CANDIDATE_CAP),
            ).fetchall()

        cat_ids = np.array([r["movieId"] for r in cats], dtype=np.int64)
        cat_emb = np.zeros((len(cat_ids), self.d), dtype=np.float32)
        cat_gvecs = np.zeros((len(cat_ids), G))
        for i, r in enumerate(cats):
            g = _one_hot(r["genres"])
            cat_gvecs[i] = g
            cat_emb[i] = g @ self.A
        # replace ridge estimates with the trained Q rows where they exist
        item_pos = np.searchsorted(item_ids, cat_ids)
        in_bounds = item_pos < len(item_ids)
        known = np.zeros(len(cat_ids), dtype=bool)
        known[in_bounds] = item_ids[item_pos[in_bounds]] == cat_ids[in_bounds]
        cat_emb[known] = Q[item_pos[known]].astype(np.float32)

        self.cat_ids = cat_ids
        self.cat_emb = cat_emb
        self.g_base = cat_gvecs.mean(axis=0)

        cand_arr = np.array([r["movieId"] for r in cands], dtype=np.int64)
        cpos = np.searchsorted(cat_ids, cand_arr)
        self.cand_ids = cand_arr
        self.cand_emb = cat_emb[cpos]

        self.pop_mean = self.cand_emb.mean(axis=0)

    # ------------------------------------------------------------ persistence

    def _save(self, meta: dict) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed appends ".npz", so write to a tmp *without* the
        # final .npz suffix, then atomically rename onto the real path.
        tmp = self.model_path.parent / (self.model_path.name + ".tmp")
        np.savez_compressed(
            str(tmp),
            W1=self.W1, b1=self.b1, w2=self.w2, b2=np.float32(self.b2),
            A=self.A, g_base=self.g_base,
            cat_ids=self.cat_ids, cat_emb=self.cat_emb,
            cand_ids=self.cand_ids, cand_emb=self.cand_emb,
            pop_mean=self.pop_mean,
            meta=np.asarray([json.dumps(meta)]),
        )
        os.replace(str(tmp) + ".npz", str(self.model_path))

    def _load(self) -> None:
        z = np.load(self.model_path)
        self.W1 = z["W1"]
        self.b1 = z["b1"]
        self.w2 = z["w2"]
        self.b2 = float(z["b2"])
        self.A = z["A"]
        self.g_base = z["g_base"]
        self.cat_ids = z["cat_ids"]
        self.cat_emb = z["cat_emb"]
        self.cand_ids = z["cand_ids"]
        self.cand_emb = z["cand_emb"]
        self.pop_mean = z["pop_mean"]
        self.d = self.W1.shape[1] // 2
        meta = json_load(z["meta"])
        with self._lock:
            self.status = "ready"
            self.progress = 1.0
            self.message = "loaded from disk"
            self.meta = meta or {"source": "cached"}


def json_load(arr) -> dict | None:
    try:
        return json.loads(str(arr[0]))
    except Exception:
        return None


class _ServingCfg:
    """Lightweight NCFConfig plus metadata — decoupled from the sweep config's lists."""

    def __init__(self, num_clients: int):
        self.embed_dim = 32
        self.hidden = 64
        self.rounds = 40
        self.clients = num_clients
        self.per_client_cap = 500
        self.min_ratings = 300
        self.test_frac = 0.0
        self.eval_users = 0
        self.eval_candidates = 0
        self.batch = 40
        self.lr = 0.5
        self.clip_norm = 1.5
        self.epsilons = [1.0]
        self.seed = 1
        self.eps_now = 1.0
        self.init_scale = 3.0  # scale the NCF 0.04 init up so SGD actually learns

    def gen_meta(self, num_clients: int) -> dict:
        return {
            "source": "federated_ncf",
            "rounds": self.rounds,
            "clients": num_clients,
            "epsilon": 1.0,
            "clip_norm": self.clip_norm,
            "embed_dim": self.embed_dim,
            "hidden": self.hidden,
        }


MODEL_PATH = DATA_DIR / "ncf_model.npz"
SERVER = NCFFederatedServer(MODEL_PATH)