"""Federated NCF (NumPy) + plain Local-DP vs Local-DP+SecAgg noise sweeps.

Research pipeline for the seminar claim ("relaxation effect"):
    SecAgg over K clients amplifies privacy by ~ sqrt(K), so we can run at a
    LARGER nominal epsilon (less noise) and still hold the same effective
    per-user guarantee — trading the headroom for accuracy.

Layers:
    P[user] concat Q[item] (2*d) -> ReLU(W1) -> w2 -> rating
FedAvg per round:
    client:  per-example L2-clip of gradients, clipped average, Gaussian noise
             sigma = (2*C/n) * sqrt(2 ln(1.25/delta)) / eps   (mean sensitivity)
    secagg:  pairwise additive masks over the shared-gradient vector; the
             masks cancel at the server before the FedAvg step.
Serving:   predict(u, i) over the catalog -> top-k recommendations.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from database import cursor

log = logging.getLogger("ncf_fl")
log.setLevel(logging.INFO)

DELTA = 1e-5


@dataclass
class NCFConfig:
    embed_dim: int = 32
    hidden: int = 48
    rounds: int = 30
    clients: int = 14
    per_client_cap: int = 80
    min_ratings: int = 80
    test_frac: float = 0.25
    eval_users: int = 20
    eval_candidates: int = 150
    batch: int = 40
    lr: float = 0.4
    clip_norm: float = 2.0
    epsilons: list = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    seed: int = 1


# ---------------------------------------------------------------- model

class NCF:
    """Two-layer NCF in numpy: rating = relu(concat(P[u],Q[i]) W1^T + b1) . w2 + b2."""

    def __init__(self, num_users: int, num_items: int, cfg: NCFConfig, init_b: float = 3.5):
        self.cfg = cfg
        self.nu, self.ni = num_users, num_items
        rng = np.random.default_rng(cfg.seed)
        s = 0.04
        self.d, self.h = cfg.embed_dim, cfg.hidden
        self.P = rng.normal(0, s, (num_users, self.d))
        self.Q = rng.normal(0, s, (num_items, self.d))
        self.W1 = rng.normal(0, s, (self.h, 2 * self.d))
        self.b1 = np.zeros(self.h)
        self.w2 = rng.normal(0, s, (self.h,))
        self.b2 = init_b

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.P[users], self.Q[items]], axis=1)
        h1 = np.maximum(x @ self.W1.T + self.b1, 0.0)
        return h1 @ self.w2 + self.b2

    def _per_example(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                     cfg: NCFConfig):
        """Per-example L2-clipped gradient accumulation (no noise, raw sums).

        Returns accumulated sums over examples for (W1, b1, w2, b2, P-row, Q)."""
        n = len(users)
        C = cfg.clip_norm
        per = 1.0 / n
        gW1 = np.zeros_like(self.W1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = 0.0
        gP = np.zeros((1, self.d))
        gQfull = np.zeros_like(self.Q)

        for e in range(n):
            p = self.P[users[e]]
            q = self.Q[items[e]]
            x = np.concatenate([p, q])
            z1 = x @ self.W1.T + self.b1
            h1 = np.maximum(z1, 0.0)
            yh = h1 @ self.w2 + self.b2
            dy = 2.0 * (yh - ratings[e]) * per
            g_w2 = dy * h1
            g_b2 = dy
            g_h1 = dy * self.w2
            g_z1 = g_h1 * (z1 > 0)
            g_W1 = np.outer(g_z1, x)
            g_b1 = g_z1.copy()
            g_x = g_z1 @ self.W1
            g_p = g_x[: self.d]
            g_q = g_x[self.d:]
            vec = np.concatenate([g_W1.ravel(), g_b1, g_w2, [g_b2], g_p, g_q])
            nrm = float(np.linalg.norm(vec))
            if nrm > C:
                vec = vec * (C / nrm)
            off = 0

            def take(k):
                nonlocal off
                v = vec[off:off + k]
                off += k
                return v
            aW1 = take(self.h * (2 * self.d)).reshape(self.W1.shape)
            ab1 = take(self.h)
            aw2 = take(self.h)
            ab2 = take(1)[0]
            ap = take(self.d)
            aq = take(self.d)
            gW1 += aW1; gb1 += ab1; gw2 += aw2; gb2 += ab2
            gP[0] += ap
            gQfull[items[e]] += aq
        return gW1, gb1, gw2, gb2, gP, gQfull

    def clean_global_grad(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                          cfg: NCFConfig) -> np.ndarray:
        """Mean per-example clipped gradient over shared params only (no noise).

        Exactly what a client *would* upload before masking/noise is applied —
        the object reverse-engineering attackers try to reconstruct."""
        n = len(users)
        gW1, gb1, gw2, gb2, _gP, gQfull = self._per_example(users, items, ratings, cfg)
        shared = np.concatenate([gW1.ravel(), gb1, gw2, [gb2]]) / n
        return np.concatenate([shared, (gQfull / n).ravel()])

    def shared_grad(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                    cfg: NCFConfig, rng) -> tuple[np.ndarray, np.ndarray]:
        """Per-example L2-clip + clipped mean + Gaussian noise.

        Returns (noisy_shared_grad, user_row_grad):
          shared  = [W1, b1, w2, b2, Q_rows of rated items]  (fed to FedAvg)
          userrow = gradient for this client's P row (kept local).
        """
        n = len(users)
        gW1, gb1, gw2, gb2, gP, gQfull = self._per_example(users, items, ratings, cfg)
        shared = np.concatenate([gW1.ravel(), gb1, gw2, [gb2]]) / n
        shared = np.concatenate([shared, (gQfull / n).ravel()])
        if getattr(cfg, "eps_now", None) is not None:
            C = cfg.clip_norm
            sigma = 2.0 * C * math.sqrt(2 * math.log(1.25 / DELTA)) / (n * cfg.eps_now)
            shared = shared + rng.normal(0, sigma, size=shared.shape)
        return shared, gP[0] / n

    def apply_shared(self, grad: np.ndarray, cfg: NCFConfig) -> None:
        """FedAvg update: -= lr * grad."""
        n1 = self.h * (2 * self.d)
        n2 = n1 + self.h
        n3 = n2 + self.h
        gW1 = grad[:n1].reshape(self.W1.shape)
        gb1 = grad[n1:n2]
        gw2 = grad[n2:n3]
        gb2 = grad[n3]
        gQ = grad[n3 + 1:]
        self.W1 -= cfg.lr * gW1
        self.b1 -= cfg.lr * gb1
        self.w2 -= cfg.lr * gw2
        self.b2 -= cfg.lr * gb2
        self.Q -= cfg.lr * gQ.reshape(self.Q.shape)

    def apply_user(self, uid: int, grad: np.ndarray, cfg: NCFConfig) -> None:
        self.P[uid] -= cfg.lr * grad

    # fingerprint for SecAgg mask length
    @property
    def shared_size(self) -> int:
        return self.h * (2 * self.d) + 2 * self.h + 1 + self.ni * self.d


class NCFRecommender:
    """Serving wrapper: ranks the catalog for a user by predicted rating."""

    def __init__(self, model: NCF, movie_ids: np.ndarray, rated: dict[int, set[int]]):
        self.model = model
        self.movie_ids = movie_ids
        self.rated = rated

    def recommend(self, user_id: int, topk: int = 10) -> np.ndarray:
        u = np.full(len(self.movie_ids), user_id, dtype=np.int64)
        scores = self.model.predict(u, self.movie_ids)
        skip = self.rated.get(user_id, set())
        order = np.argsort(-scores)
        out = []
        for idx in order:
            mid = int(self.movie_ids[idx])
            if mid in skip:
                continue
            out.append((mid, float(scores[idx])))
            if len(out) >= topk:
                break
        return np.array(out) if out else np.empty((0, 2))