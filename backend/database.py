import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    movieId INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    rating REAL NOT NULL,
    timestamp INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
    userId INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    tag TEXT NOT NULL,
    timestamp INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_ratings (
    user_id INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    rating REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, movieId)
);

CREATE TABLE IF NOT EXISTS movie_links (
    movieId INTEGER PRIMARY KEY,
    imdbId TEXT,
    tmdbId TEXT,
    poster_path TEXT
);

CREATE TABLE IF NOT EXISTS user_lists (
    user_id INTEGER NOT NULL,
    movieId INTEGER NOT NULL,
    list_type TEXT NOT NULL CHECK (list_type IN ('favorite', 'watch_later')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, movieId, list_type)
);

CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(userId);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movieId);
CREATE INDEX IF NOT EXISTS idx_ratings_um ON ratings(userId, movieId);
CREATE INDEX IF NOT EXISTS idx_ratings_mr ON ratings(movieId, rating);
CREATE INDEX IF NOT EXISTS idx_tags_movie ON tags(movieId);
CREATE INDEX IF NOT EXISTS idx_user_ratings ON user_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_user_lists_user ON user_lists(user_id);
CREATE INDEX IF NOT EXISTS idx_user_lists_um ON user_lists(user_id, list_type);
CREATE INDEX IF NOT EXISTS idx_movies_lower ON movies(LOWER(title));
"""


def get_conn(db_path=DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def cursor(db_path=DB_PATH):
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        yield cur, conn
        conn.commit()
    finally:
        conn.close()


def db_stats(db_path=DB_PATH) -> dict:
    with cursor(db_path) as (cur, _):
        counts = {}
        for table in ("movies", "ratings", "tags", "users"):
            counts[table] = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return counts
