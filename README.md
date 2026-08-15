# MovieStore · Privacy-First Movie Storefront on MovieLens 32M

An Amazon-style movie storefront built on the **MovieLens 32M** dataset, designed from day one to be the
training/inference front-end for the seminar project:

> **"Privacy-preserving federated recommendations with Local Differential Privacy, Secure Aggregation,
> CKKS Homomorphic Encryption, and SHAP explainability."**

The research claim the dashboard demonstrates: by combining **SecAgg + CKKS + SHAP-driven feature
selection**, we can run at a **larger local ε (e.g. 2.5)** while preserving the *effective* per-user
privacy guarantee of ε = 1 plain local DP — recovering **10–15% accuracy** on held-out ratings.

## Stack

| Layer    | Technology |
|----------|------------|
| Frontend | Vanilla JS SPA (no build step, works offline) |
| Backend  | FastAPI (Python 3.13) |
| Storage  | SQLite (streamed import of the 877 MB ratings file) |
| ML       | NumPy (FedAvg, local DP, SecAgg, SHAP) + **tenseal** (real CKKS HE) |

## Quick start

```powershell
# 1. (first time) import the dataset — point at your MovieLens 32M folder
$env:ML32M_DIR = "D:\seminar\ml-32m"
.venv\Scripts\python.exe backend\import_data.py --full      # all 32.8M ratings
.venv\Scripts\python.exe backend\import_data.py --ratio 0.25  # ~8M for fast dev

# 2. run the site
.\run.ps1

# 3. open
# http://127.0.0.1:8000
```

`run.ps1` creates the venv and installs `backend/requirements.txt` on first run.

## Features

- **Catalog** — search (live suggestions + FTS5 on title & genre), genre / year filters, sorting,
  pagination over 86k titles. Category cards on the home page.
- **Posters** — hybrid: real TMDB artwork when fetched (see below), otherwise auto-generated gradient
  posters, cached on demand.
- **Movie detail** — rating aggregates, community tags, "customers also viewed", star rating widget.
- **Accounts** — register / login (PBKDF2), per-user ratings stored locally.
- **Basket** — **Favorites** and **Watch later** lists, persistent per account (survive logout), on every
  card + detail page, with a My Basket page. The `user_lists` data source is FL-ready: it becomes an
  implicit training signal in the federated pipeline.
- **Recommendations** — popularity baseline + user-user collaborative filtering; seamlessly upgraded to the
  **federated model** once you train it on the dashboard.
- **Privacy & FL dashboard**:
  - `Federated Learning` — configure ε sweep, client count, toggles for CKKS / SecAgg / SHAP; runs FedAvg
    with local DP in the background with live progress.
  - `Privacy` — ε vs accuracy chart and the privacy-accounting argument (SecAgg amplification ≈ ε/√K).
  - `SHAP` — global feature-importance bars from the exact linear SHAP of the global model.

## Posters (hybrid)

Generated gradient posters work with zero setup. For real TMDB artwork, set a free API key and fetch the
popular titles (re-run to resume / fetch more):

```powershell
$env:TMDB_KEY = "your_key"
.venv\Scripts\python.exe backend\fetch_posters.py --limit 20000
```

## Architecture

```
frontend/                  # static SPA (served by FastAPI)
backend/
  main.py                  # FastAPI app + all REST routes
  import_data.py           # streaming MovieLens 32M importer (incl. links.csv)
  database.py              # SQLite schema + helpers
  auth.py                  # PBKDF2 + HMAC tokens
  recommend.py             # catalog search (FTS5) + HybridEngine + basket lists
  posters.py               # hybrid poster generation / TMDB resolution
  fetch_posters.py         # optional TMDB poster-path fetcher (needs TMDB_KEY)
  fl_trainer.py            # FedAvg + Local DP + SecAgg + CKKS + SHAP
  config.py                # paths (ML32M_DIR env var) + secrets
```

## How the federated pipeline works (per round)

1. **Clients** — a sample of MovieLens users each hold their *private* ratings.
2. **Local DP** — each client trains SGD, clips per-example gradients to L2 norm C,
   adds Gaussian noise σ = C·√(2·ln(1.25/δ)) / ε.
3. **SecAgg** — clients add pairwise additive masks so the server only ever sees the *sum*.
4. **CKKS** — every masked update is encrypted with tenseal; the server performs the aggregate
   *homomorphically* under ciphertext, then decrypts once.
5. **FedAvg** — the decrypted aggregate updates the global model.
6. **SHAP** — for the linear model, SHAP values are exact (φᵢ = wᵢ·(xᵢ − E[xᵢ])); global importances
   drive feature pruning that re-covers accuracy at larger ε.

Privacy amplification: with SecAgg over K clients, the aggregate-view privacy is ≈ ε/√K — this is the
margin that lets the paper operate at ε = 2.5 for the accuracy you used to get only at ε = 1.

## Dataset

| File            | Rows     |
|-----------------|----------|
| `movies.csv`    | 87,585   |
| `ratings.csv`   | 32,829,362 (877 MB) |
| `tags.csv`      | 2,087,657 |
| `links.csv`     | 87,585   |

## Notes

- `data/movielens.db` is large (≈2 GB at full import) and is git-ignored — rebuild it with the importer.
- CKKS uses `tenseal` (installed from `requirements.txt`); the code falls back to a plaintext sum if the
  library is unavailable, so the site always runs.
- Change `APP_SECRET` in `backend/config.py` (or env var) before deploying anything real.
