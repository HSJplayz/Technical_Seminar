"""Recommendation engine.

Baseline model for the site: popularity + user-user collaborative filtering.
The design is deliberately pluggable: `HybridEngine.set_trained_model(...)` is
the seam where the Federated-Learning model (FedAvg + DP + SecAgg + CKKS)
trained in `fl_engine.py` will be injected later, so the website's recs are
served by whatever model the research pipeline produces.
"""
import numpy as np

from database import cursor


def movie_year_expr() -> str:
    return "CAST(substr(title, instr(title, '(') + 1, 4) AS INTEGER)"


def get_movie(movie_id: int) -> dict | None:
    with cursor() as (cur, _):
        row = cur.execute("SELECT movieId, title, genres FROM movies WHERE movieId = ?", (movie_id,)).fetchone()
        if row is None:
            return None
        m = dict(row)
        m["year"] = _parse_year(m["title"])
        m["genre_list"] = m["genres"].split("|")
        return m


def _parse_year(title: str) -> int | None:
    if "(" in title and title.rstrip().endswith(")"):
        try:
            return int(title[title.rfind("(") + 1:title.rfind(")")])
        except ValueError:
            return None
    return None


def get_genres() -> list[str]:
    with cursor() as (cur, _):
        rows = cur.execute("SELECT DISTINCT genres FROM movies").fetchall()
    genres: set[str] = set()
    for r in rows:
        genres.update(r["genres"].split("|"))
    return sorted(g for g in genres if g and g != "(no genres listed)")


def search_movies(query: str = "", genres: list[str] | None = None,
                  year_min: int | None = None, year_max: int | None = None,
                  sort: str = "relevance", page: int = 1, per_page: int = 24) -> dict:
    sql = "SELECT movieId, title, genres FROM movies"
    conds, params = [], []
    if query:
        conds.append("LOWER(title) LIKE ?")
        params.append(f"%{query.lower()}%")
    if genres:
        placeholders = ",".join("?" for _ in genres)
        conds.append(f"genres IN ({placeholders})")
        params.extend(genres)
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
            "title": "title ASC",
        }.get(sort, "title ASC")
        sql += " ORDER BY " + order + " LIMIT ? OFFSET ?"
        params += [per_page, (page - 1) * per_page]
        rows = cur.execute(sql, params).fetchall()

    movies = []
    for r in rows:
        m = dict(r)
        m["year"] = _parse_year(m["title"])
        m["genre_list"] = m["genres"].split("|")
        m["title_clean"] = _strip_year(m["title"])
        movies.append(m)
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
    out = []
    for r in rows:
        m = dict(r)
        m["year"] = _parse_year(m["title"])
        m["genre_list"] = m["genres"].split("|")
        m["title_clean"] = _strip_year(m["title"])
        out.append(m)
    return out


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
        d = ranked[mid]
        d["title_clean"] = _strip_year(d["title"])
        d["year"] = _parse_year(d["title"])
        d["genre_list"] = d["genres"].split("|")
        out.append(d)
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


def _in_chunks(values: list[int], chunk: int = 500) -> list[list[int]]:
    return [values[i:i + chunk] for i in range(0, len(values), chunk)]


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
        d["title_clean"] = _strip_year(d["title"])
        d["year"] = _parse_year(d["title"])
        d["genre_list"] = d["genres"].split("|")
        out.append(d)
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
