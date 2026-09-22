"""CKKS homomorphic-encryption layer for the NCF FL pipeline (tenseal).

Only ciphertext *addition* is needed — FedAvg aggregates gradients — so we use
an additive-CKKS setup: each client encrypts its (noisy, optionally masked)
gradient, the server sums the ciphertexts, clients collectively decrypt the
aggregate.  Batched plaintext is encoded into N/2 real slots; gradients wider
than the slot count are chunked into several ciphertexts per vector.

   client_i : g_i  --encrypt-->  ct_i          (secret key stays client-side)
   server   : ct_sum = SUM ct_i  (homomorphic; never sees plaintexts)
   clients  : g_sum = decrypt(ct_sum) / K  ==  (1/K) SUM_i g_i   (CKKS ≈ exact
            for additions; noise grows only with the number of adds)
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("he_fl")

try:
    import tenseal as ts
    CKKS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when tenseal is missing
    ts = None
    CKKS_AVAILABLE = False

# mechanism flags per stack name (shared by sweep + attacks)
STACK_MECH = {
    "dp":           {"dp": True,  "secagg": False, "he": False},
    "he":           {"dp": False, "secagg": False, "he": True},
    "secagg":       {"dp": False, "secagg": True,  "he": False},
    "dp_secagg":    {"dp": True,  "secagg": True,  "he": False},
    "dp_he":        {"dp": True,  "secagg": False, "he": True},
    "secagg_he":    {"dp": False, "secagg": True,  "he": True},
    "dp_secagg_he": {"dp": True,  "secagg": True,  "he": True},
}

N = 8192
SLOTS = N // 2
SCALE = 2 ** 24
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        if not CKKS_AVAILABLE:
            raise RuntimeError("tenseal not available in this environment")
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, N, coeff_mod_bit_sizes=[60, 40, 40])
        ctx.global_scale = SCALE
        _ctx = ctx
    return _ctx


class HE:
    """Chunked additive CKKS for gradients wider than SLOTS."""

    def __init__(self, dim: int, slot: int = SLOTS):
        self.dim = dim
        self.slot = slot

    def _chunks(self, vec: np.ndarray):
        out = []
        for i in range(0, self.dim, self.slot):
            out.append(vec[i:i + self.slot])
        return out

    def encrypt(self, vec: np.ndarray):
        ctx = _get_ctx()
        return [ts.ckks_vector(ctx, c.tolist()) for c in self._chunks(vec)]

    def add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def sum_all(self, cts):
        if not cts:
            return None
        acc = list(cts[0])
        for c in cts[1:]:
            acc = self.add(acc, c)
        return acc

    def decrypt(self, ctexts) -> np.ndarray:
        return np.concatenate([np.array(c.decrypt(), dtype=float)
                               for c in ctexts])

    # -- demo helpers: server-side ciphertext operations never touch plaintext
    def ciphertext_blob(self, ctexts) -> list[int]:
        """Opaque server-side view used by the attack evaluation (no raw data)."""
        return [len(c) for c in ctexts]