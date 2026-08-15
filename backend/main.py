"""MovieLens 32M e-commerce-style site + Federated Learning research backend."""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import database
import posters
import recommend
from auth import create_token, hash_password, verify_password, verify_token
import fl_trainer
from fl_trainer import MANAGER, RunConfig
from config import DB_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db(DB_PATH)
    yield


app = FastAPI(title="MovieStore · Privacy-First Movie Storefront", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


# ---------------------------------------------------------------- auth utils

def _current_user(authorization: str | None = Header(default=None)) -> int | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return verify_token(token)


def _require_user(user_id) -> int:
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


# ---------------------------------------------------------------- schemas

class RegisterBody(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=200)


class RateBody(BaseModel):
    movieId: int
    rating: float = Field(ge=0.5, le=5.0)


class FLStartBody(BaseModel):
    epsilons: list[float] = Field(default_factory=lambda: [1.0, 2.5])
    clients: int = Field(default=10, ge=2, le=40)
    epochs: int = Field(default=12, ge=1, le=30)
    use_he: bool = True
    use_secagg: bool = True
    use_shap: bool = True


# ---------------------------------------------------------------- auth routes

@app.post("/api/auth/register")
def register(body: RegisterBody):
    with database.cursor() as (cur, conn):
        if cur.execute("SELECT id FROM users WHERE username = ?", (body.username,)).fetchone():
            raise HTTPException(status_code=409, detail="Username taken")
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (body.username, hash_password(body.password)),
        )
        user_id = cur.lastrowid
    return {"token": create_token(user_id), "user": {"id": user_id, "username": body.username}}


@app.post("/api/auth/login")
def login(body: RegisterBody):
    with database.cursor() as (cur, _):
        row = cur.execute("SELECT id, username, password_hash FROM users WHERE username = ?",
                          (body.username,)).fetchone()
    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": create_token(row["id"]), "user": {"id": row["id"], "username": row["username"]}}


@app.get("/api/me")
def me(authorization: str | None = Header(default=None)):
    uid = _current_user(authorization)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    with database.cursor() as (cur, _):
        row = cur.execute("SELECT id, username, created_at FROM users WHERE id = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


# ---------------------------------------------------------------- catalog routes

@app.get("/api/stats")
def stats():
    return {"counts": database.db_stats(DB_PATH), "fl": MANAGER.state()}


@app.get("/api/genres")
def genres():
    return {"genres": recommend.get_genres()}


@app.get("/api/search/suggest")
def search_suggest(q: str = ""):
    return {"items": recommend.search_suggest(q)}


@app.get("/api/poster/{movie_id}.png")
def movie_poster(movie_id: int):
    try:
        posters.ensure_poster_file(movie_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Movie not found")
    return FileResponse(posters.POSTER_DIR / f"{movie_id}.png", media_type="image/png")


@app.get("/api/movies")
def movies(q: str = "", genre: str = "", sort: str = "relevance",
           year_min: int = 0, year_max: int = 0, page: int = 1, per_page: int = 24):
    genres = [g.strip() for g in genre.split(",") if g.strip()]
    return recommend.search_movies(
        query=q, genres=genres or None,
        year_min=year_min or None, year_max=year_max or None,
        sort=sort, page=page, per_page=min(per_page, 60),
    )


@app.get("/api/popular")
def popular(limit: int = 24):
    return {"items": recommend.popular_movies(limit)}


@app.get("/api/movies/{movie_id}")
def movie(movie_id: int, authorization: str | None = Header(default=None)):
    detail = recommend.movie_detail(movie_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    uid = _current_user(authorization)
    detail["similar"] = recommend.similar_movies(movie_id)
    detail["my_rating"] = None
    detail["lists"] = {t: False for t in recommend.LIST_TYPES}
    if uid is not None:
        with database.cursor() as (cur, _):
            row = cur.execute(
                "SELECT rating FROM user_ratings WHERE user_id = ? AND movieId = ?",
                (uid, movie_id),
            ).fetchone()
        detail["my_rating"] = row["rating"] if row else None
        detail["lists"] = recommend.list_status(uid, movie_id)
    return detail


# ---------------------------------------------------------------- user routes

@app.get("/api/recommendations")
def recommendations(authorization: str | None = Header(default=None), limit: int = 24):
    uid = _current_user(authorization)
    return recommend.ENGINE.recommend(uid, min(limit, 48))


@app.get("/api/my-ratings")
def my_ratings(authorization: str | None = Header(default=None)):
    uid = _require_user(_current_user(authorization))
    rated = recommend.get_user_ratings(uid)
    titles: dict[int, dict] = {}
    if rated:
        for chunk in recommend._in_chunks([r["movieId"] for r in rated]):
            with database.cursor() as (cur, _):
                rows = cur.execute(
                    f"SELECT movieId, title, genres FROM movies WHERE movieId IN ({','.join('?' for _ in chunk)})",
                    chunk,
                ).fetchall()
            titles.update({r["movieId"]: dict(r) for r in rows})
    out = []
    for r in rated:
        m = titles.get(r["movieId"])
        if not m:
            continue
        out.append({**r, **recommend._decorate(m)})
    return {"items": out}


@app.post("/api/rate")
def rate(body: RateBody, authorization: str | None = Header(default=None)):
    uid = _require_user(_current_user(authorization))
    if recommend.get_movie(body.movieId) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    recommend.rate_movie(uid, body.movieId, body.rating)
    return {"ok": True}


@app.delete("/api/rate/{movie_id}")
def unrate(movie_id: int, authorization: str | None = Header(default=None)):
    uid = _require_user(_current_user(authorization))
    recommend.remove_rating(uid, movie_id)
    return {"ok": True}


# ---------------------------------------------------------------- basket routes

@app.get("/api/lists/summary")
def lists_summary(authorization: str | None = Header(default=None)):
    uid = _require_user(_current_user(authorization))
    return {
        "counts": recommend.list_counts(uid),
        "favorite": recommend.get_user_list(uid, "favorite"),
        "watch_later": recommend.get_user_list(uid, "watch_later"),
    }


@app.get("/api/list/{list_type}")
def get_list(list_type: str, authorization: str | None = Header(default=None)):
    uid = _require_user(_current_user(authorization))
    try:
        items = recommend.get_user_list(uid, list_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Unknown list type")
    return {"items": items}


@app.put("/api/list/{list_type}/{movie_id}")
def add_to_list(list_type: str, movie_id: int, authorization: str | None = Header(default=None)):
    uid = _require_user(_current_user(authorization))
    if list_type not in recommend.LIST_TYPES:
        raise HTTPException(status_code=400, detail="Unknown list type")
    if recommend.get_movie(movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    recommend.add_to_list(uid, movie_id, list_type)
    return {"ok": True}


@app.delete("/api/list/{list_type}/{movie_id}")
def remove_from_list(list_type: str, movie_id: int, authorization: str | None = Header(default=None)):
    uid = _require_user(_current_user(authorization))
    if list_type not in recommend.LIST_TYPES:
        raise HTTPException(status_code=400, detail="Unknown list type")
    recommend.remove_from_list(uid, movie_id, list_type)
    return {"ok": True}


# ---------------------------------------------------------------- FL research routes

@app.post("/api/fl/start")
def fl_start(body: FLStartBody, authorization: str | None = Header(default=None)):
    _require_user(_current_user(authorization))
    cfg = RunConfig(
        epsilons=body.epsilons, clients=body.clients, rounds=body.epochs,
        use_he=body.use_he, use_secagg=body.use_secagg, use_shap=body.use_shap,
    )
    started = MANAGER.start(cfg, on_complete=lambda: fl_trainer.install_serving_model(MANAGER.state().get("result") or {}))
    if not started:
        raise HTTPException(status_code=409, detail="A training run is already in progress")
    return MANAGER.state()


@app.get("/api/fl/status")
def fl_status():
    return MANAGER.state()


# ---------------------------------------------------------------- SPA shell

@app.get("/{full_path:path}")
def spa(full_path: str):
    target = FRONTEND / full_path
    if full_path and target.is_file():
        return FileResponse(target)
    return FileResponse(FRONTEND / "index.html")
