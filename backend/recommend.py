"""Recommendation engine.

Baseline model for the site: popularity + user-user collaborative filtering.
The design is deliberately pluggable: `HybridEngine.set_trained_model(...)` is
the seam where the Federated-Learning model (FedAvg + DP + SecAgg + CKKS)
trained in `fl_engine.py` will be injected later, so the website's recs are
served by whatever model the research pipeline produces.
"""
import numpy as np

import posters
from database import cursor


def movie_year_expr() -> str:
    return "CAST(substr(title, instr(title, '(') + 1, 4) AS INTEGER)"


def _decorate(m: dict) -> dict:
    m["year"] = _parse_year(m["title"])
    m["genre_list"] = m["genres"].split("|")
    m["title_clean"] = _strip_year(m["title"])
    m["poster_url"] = posters.poster_url(m["movieId"])
    return m


def get_movie(movie_id: int) -> dict | None:
    with cursor() as (cur, _):
        row = cur.execute("SELECT movieId, title, genres FROM movies WHERE movieId = ?", (movie_id,)).fetchone()
        if row is None:
            return None
        return _decorate(dict(row))


def _parse_year(title: str) -> int | None:
    if "(" in title and title.rstrip().endswith(")"):
        try:
            return int(title[title.rfind("(") + 1:title.rfind(")")])
        except ValueError:
            return None
    return None


def get_genres() -> list[dict]:
    with cursor() as (cur, _):
        rows = cur.execute("SELECT genres FROM movies").fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        for g in r["genres"].split("|"):
            if g and g != "(no genres listed)":
                counts[g] = counts.get(g, 0) + 1
    return [{"name": g, "count": counts[g]} for g in sorted(counts)]


def _fts_init() -> None:
    with cursor() as (cur, conn):
        cur.execute("CREATE TABLE IF NOT EXISTS fts_meta (k TEXT PRIMARY KEY, v TEXT)")
        if cur.execute("SELECT 1 FROM fts_meta WHERE k = 'movies_fts'").fetchone():
            return
        cur.execute("DROP TABLE IF EXISTS movies_fts")
        cur.execute("CREATE VIRTUAL TABLE movies_fts USING fts5(title, genres, content='')")
        cur.execute(
            """INSERT INTO movies_fts (rowid, title, genres)
               SELECT movieId, title, genres FROM movies"""
        )
        cur.execute("INSERT INTO fts_meta (k, v) VALUES ('movies_fts', '1')")
        conn.commit()


def search_suggest(query: str, limit: int = 8) -> list[dict]:
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []
    _fts_init()
    term = " ".join(w + "*" for w in q.split())
    with cursor() as (cur, _):
        try:
            rows = cur.execute(
                f"""SELECT movieId, title, genres FROM movies_fts
                    WHERE movies_fts MATCH ? ORDER BY bm25(movies_fts) LIMIT ?""",
                (term, limit),
            ).fetchall()
        except Exception:
            rows = cur.execute(
                "SELECT movieId, title, genres FROM movies WHERE LOWER(title) LIKE ? LIMIT ?",
                (f"%{q}%", limit),
            ).fetchall()
    return [_decorate(dict(r)) for r in rows]


def search_movies(query: str = "", genres: list[str] | None = None,
                  year_min: int | None = None, year_max: int | None = None,
                  sort: str = "relevance", page: int = 1, per_page: int = 24) -> dict:
    sql = "SELECT movieId, title, genres FROM movies"
    conds, params = [], []
    if query:
        conds.append("(LOWER(title) LIKE ? OR LOWER(genres) LIKE ?)")
        params += [f"%{query.lower()}%", f"%{query.lower()}%"]
    if genres:
        like_conds = " OR ".join(["genres LIKE ?"] * len(genres))
        conds.append(f"({like_conds})")
        params += [f"%{g}%" for g in genres]
    if year_min is not None:
        conds.append(f"{movie_year_expr()} >= ?")
        params.append(year_min)
    if year_max is not None:
        conds.append(f"{movie_year_expr()} <= ?")
        params.append(year_max)

    if conds:
        sql += " WHERE " + " AND ".join(conds)

    total_sql = "SELECT COUNT(*) FROM (" + sql + ")"
    with cursor() as (cur, _):
        total = cur.execute(total_sql, params).fetchone()[0]
        order = {
            "relevance": "title ASC",
            "year": f"{movie_year_expr()} DESC, title ASC",
            "rating": "(SELECT p.score FROM popularity p WHERE p.movieId = movies.movieId) DESC, title ASC",
            "title": "title ASC",
        }.get(sort, "title ASC")
        sql += " ORDER BY " + order + " LIMIT ? OFFSET ?"
        params += [per_page, (page - 1) * per_page]
        rows = cur.execute(sql, params).fetchall()

    movies = [_decorate(dict(r)) for r in rows]
    return {"items": movies, "total": total, "page": page, "per_page": per_page,
            "pages": max(1, -(-total // per_page))}


def _strip_year(title: str) -> str:
    idx = title.rfind("(")
    return title[:idx].strip() if idx > 0 else title


def popular_movies(limit: int = 24) -> list[dict]:
    with cursor() as (cur, _):
        rows = cur.execute(
            """SELECT m.movieId, m.title, m.genres, p.cnt, ROUND(p.avg_rating,2) AS avg_rating
               FROM popularity p JOIN movies m ON m.movieId = p.movieId
               WHERE p.cnt >= 50 ORDER BY p.score DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [_decorate(dict(r)) for r in rows]


def movie_detail(movie_id: int) -> dict | None:
    m = get_movie(movie_id)
    if m is None:
        return None
    with cursor() as (cur, _):
        agg = cur.execute(
            "SELECT COUNT(*) AS cnt, ROUND(AVG(rating), 2) AS avg_rating FROM ratings WHERE movieId = ?",
            (movie_id,),
        ).fetchone()
        tags = cur.execute(
            "SELECT tag, COUNT(*) AS cnt FROM tags WHERE movieId = ? GROUP BY tag ORDER BY cnt DESC LIMIT 12",
            (movie_id,),
        ).fetchall()
    m["rating_count"] = agg["cnt"]
    m["avg_rating"] = agg["avg_rating"]
    m["top_tags"] = [t["tag"] for t in tags]
    return m


def similar_movies(movie_id: int, limit: int = 10) -> list[dict]:
    """Genre overlap + co-rated neighbourhood -> 'Customers also viewed'."""
    m = get_movie(movie_id)
    if m is None:
        return []
    genres = m["genre_list"]
    genre_sql = " OR ".join(["genres LIKE ?"] * len(genres))
    genre_params = [f"%{g}%" for g in genres]
    with cursor() as (cur, _):
        same_genre = cur.execute(
            f"SELECT movieId, title, genres FROM movies WHERE movieId != ? AND ({genre_sql}) ORDER BY movieId LIMIT 200",
            [movie_id] + genre_params,
        ).fetchall()
        # cap co-rated users so the neighbourhood join stays bounded
        users = cur.execute(
            "SELECT userId FROM ratings WHERE movieId = ? LIMIT 400", (movie_id,)
        ).fetchall()
        co = []
        if users:
            user_ids = [u["userId"] for u in users]
            ph = ",".join("?" for _ in user_ids)
            co = cur.execute(
                f"""SELECT m.movieId, m.title, m.genres, COUNT(*) AS shared,
                           ROUND(AVG(r.rating), 2) AS avg_rating
                    FROM ratings r JOIN movies m ON m.movieId = r.movieId
                    WHERE r.movieId != ? AND r.userId IN ({ph})
                    GROUP BY r.movieId ORDER BY shared DESC LIMIT 200""",
                [movie_id] + user_ids,
            ).fetchall()

    ranked: dict[int, dict] = {}
    for r in same_genre:
        d = dict(r)
        d["overlap"] = len(set(d["genres"].split("|")) & set(genres))
        ranked[d["movieId"]] = d
    for r in co:
        d = dict(r)
        d["overlap"] = 0
        if d["movieId"] not in ranked:
            ranked[d["movieId"]] = d
    out = []
    for mid in sorted(ranked, key=lambda k: (ranked[k].get("shared", 0) or 0, ranked[k].get("overlap", 0)), reverse=True)[:limit]:
        out.append(_decorate(ranked[mid]))
    return out


def get_user_ratings(user_id: int) -> list[dict]:
    with cursor() as (cur, _):
        rows = cur.execute(
            "SELECT movieId, rating, created_at FROM user_ratings WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def rate_movie(user_id: int, movie_id: int, rating: float) -> None:
    rating = min(5.0, max(0.5, float(rating)))
    with cursor() as (cur, conn):
        cur.execute(
            """INSERT INTO user_ratings (user_id, movieId, rating) VALUES (?,?,?)
               ON CONFLICT(user_id, movieId) DO UPDATE SET rating = excluded.rating,
               created_at = datetime('now')""",
            (user_id, movie_id, rating),
        )


def remove_rating(user_id: int, movie_id: int) -> None:
    with cursor() as (cur, _):
        cur.execute(
            "DELETE FROM user_ratings WHERE user_id = ? AND movieId = ?",
            (user_id, movie_id),
        )


def _in_chunks(values: list[int], chunk: int = 500) -> list[list[int]]:
    return [values[i:i + chunk] for i in range(0, len(values), chunk)]


# ---------------------------------------------------------------- basket lists

LIST_TYPES = ("favorite", "watch_later")


def add_to_list(user_id: int, movie_id: int, list_type: str) -> None:
    if list_type not in LIST_TYPES:
        raise ValueError(f"unknown list type: {list_type}")
    with cursor() as (cur, _):
        cur.execute(
            "INSERT OR IGNORE INTO user_lists (user_id, movieId, list_type) VALUES (?,?,?)",
            (user_id, movie_id, list_type),
        )


def remove_from_list(user_id: int, movie_id: int, list_type: str) -> None:
    with cursor() as (cur, _):
        cur.execute(
            "DELETE FROM user_lists WHERE user_id = ? AND movieId = ? AND list_type = ?",
            (user_id, movie_id, list_type),
        )


def get_user_list(user_id: int, list_type: str) -> list[dict]:
    if list_type not in LIST_TYPES:
        raise ValueError(f"unknown list type: {list_type}")
    with cursor() as (cur, _):
        rows = cur.execute(
            """SELECT l.movieId, l.created_at, m.title, m.genres
               FROM user_lists l JOIN movies m ON m.movieId = l.movieId
               WHERE l.user_id = ? AND l.list_type = ?
               ORDER BY l.created_at DESC""",
            (user_id, list_type),
        ).fetchall()
    return [_decorate(dict(r)) for r in rows]


def list_status(user_id: int, movie_id: int) -> dict[str, bool]:
    with cursor() as (cur, _):
        rows = cur.execute(
            "SELECT list_type FROM user_lists WHERE user_id = ? AND movieId = ?",
            (user_id, movie_id),
        ).fetchall()
    return {t: False for t in LIST_TYPES} | {r["list_type"]: True for r in rows}


def list_counts(user_id: int) -> dict[str, int]:
    with cursor() as (cur, _):
        rows = cur.execute(
            "SELECT list_type, COUNT(*) AS n FROM user_lists WHERE user_id = ? GROUP BY list_type",
            (user_id,),
        ).fetchall()
    return {t: 0 for t in LIST_TYPES} | {r["list_type"]: r["n"] for r in rows}


def user_cf_recommendations(user_id: int, limit: int = 24) -> list[dict]:
    """User-user collaborative filtering computed on the fly.

    1. find neighbors who co-rated >= 2 of the user's movies,
    2. load the neighbors' *full* rating profiles,
    3. weighted score of their unseen movies.
    """
    mine = get_user_ratings(user_id)
    if len(mine) < 2:
        return []
    mine_map = {r["movieId"]: r["rating"] for r in mine}
    movie_ids = list(mine_map.keys())

    neighbor_ids: list[int] = []
    with cursor() as (cur, _):
        for chunk in _in_chunks(movie_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = cur.execute(
                f"""SELECT userId FROM ratings WHERE movieId IN ({placeholders})
                    GROUP BY userId HAVING COUNT(*) >= 2
                    ORDER BY COUNT(*) DESC LIMIT 1500""",
                chunk,
            ).fetchall()
            neighbor_ids.extend(r["userId"] for r in rows)
    if not neighbor_ids:
        return []

    profiles: dict[int, dict[int, float]] = {}
    for chunk in _in_chunks(neighbor_ids):
        placeholders = ",".join("?" for _ in chunk)
        with cursor() as (cur, _):
            rows = cur.execute(
                f"SELECT userId, movieId, rating FROM ratings WHERE userId IN ({placeholders}) LIMIT 500000",
                chunk,
            ).fetchall()
        for r in rows:
            profiles.setdefault(r["userId"], {})[r["movieId"]] = r["rating"]

    scores: dict[int, float] = {}
    for other, rated in profiles.items():
        shared = [s for s in movie_ids if s in rated]
        if len(shared) < 2:
            continue
        v = np.array([mine_map[s] for s in shared], dtype=float)
        w = np.array([rated[s] for s in shared], dtype=float)
        if v.std() > 1e-9 and w.std() > 1e-9:
            vc, wc = v - v.mean(), w - w.mean()
            denom = np.linalg.norm(vc) * np.linalg.norm(wc)
            sim = float(vc @ wc / denom) if denom > 1e-9 else 0.0
        else:
            denom = np.linalg.norm(v) * np.linalg.norm(w)
            sim = float(v @ w / denom) if denom > 1e-9 else 0.0
        if sim <= 0:
            continue
        mean_other = float(np.mean(list(rated.values())))
        for mid, val in rated.items():
            if mid in mine_map:
                continue
            scores[mid] = scores.get(mid, 0.0) + sim * (val - mean_other)

    if not scores:
        return []
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: limit * 3]
    ids = [mid for mid, _ in top]
    by_id: dict[int, dict] = {}
    with cursor() as (cur, _):
        for chunk in _in_chunks(ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = cur.execute(
                f"SELECT movieId, title, genres FROM movies WHERE movieId IN ({placeholders})",
                chunk,
            ).fetchall()
            for r in rows:
                by_id[r["movieId"]] = r
    out = []
    for mid, score in top[:limit]:
        r = by_id.get(mid)
        if not r:
            continue
        d = dict(r)
        d["score"] = round(float(score), 3)
        out.append(_decorate(d))
    return out


class HybridEngine:
    """Serves recommendations. Swappable trained model (FL) via set_trained_model."""

    def __init__(self):
        self._trained_model = None

    def set_trained_model(self, model) -> None:
        self._trained_model = model

    def recommend(self, user_id: int | None, limit: int = 24) -> dict:
        if self._trained_model is not None:
            try:
                items = self._trained_model.recommend(user_id, limit)
                if items:
                    return {"source": "federated_model", "items": items}
            except Exception:
                pass
        if user_id is not None:
            items = user_cf_recommendations(user_id, limit)
            if items:
                return {"source": "user_collaborative", "items": items}
        return {"source": "popularity", "items": popular_movies(limit)}


ENGINE = HybridEngine()
