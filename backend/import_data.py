"""Streaming importer for the MovieLens 32M dataset.

Usage:
    python import_data.py --full          # import every rating (~32.8M rows)
    python import_data.py --ratio 0.25    # import ~25% of ratings (fast dev mode)

Movies and tags are always imported in full. Ratings are streamed line by line
so memory stays flat even for the 877 MB ratings file.
"""
import argparse
import csv
import random
import sys
import time

from config import MOVIES_CSV, RATINGS_CSV, TAGS_CSV, LINKS_CSV
from database import init_db, get_conn

CHUNK = 100_000


def _timed(label):
    start = time.time()
    print(f"[{label}] started at {time.strftime('%H:%M:%S')}", flush=True)
    return start


def import_movies(conn):
    start = _timed("movies")
    conn.execute("DELETE FROM movies")
    cur = conn.cursor()
    rows = []
    with open(MOVIES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            try:
                rows.append((int(r[0]), r[1], r[2]))
            except (ValueError, IndexError):
                continue
            if len(rows) >= CHUNK:
                cur.executemany("INSERT INTO movies VALUES (?,?,?)", rows)
                rows.clear()
    if rows:
        cur.executemany("INSERT INTO movies VALUES (?,?,?)", rows)
    conn.commit()
    print(f"[movies] done in {time.time() - start:.1f}s -> {conn.total_changes} rows", flush=True)


def import_tags(conn):
    start = _timed("tags")
    conn.execute("DELETE FROM tags")
    cur = conn.cursor()
    rows = []
    with open(TAGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            try:
                rows.append((int(r[0]), int(r[1]), r[2], int(r[3])))
            except (ValueError, IndexError):
                continue
            if len(rows) >= CHUNK:
                cur.executemany("INSERT INTO tags VALUES (?,?,?,?)", rows)
                rows.clear()
    if rows:
        cur.executemany("INSERT INTO tags VALUES (?,?,?,?)", rows)
    conn.commit()
    print(f"[tags] done in {time.time() - start:.1f}s -> {conn.total_changes} rows", flush=True)


def import_ratings(conn, ratio: float):
    start = _timed("ratings")
    conn.execute("DELETE FROM ratings")
    cur = conn.cursor()
    rows = []
    total = kept = 0
    with open(RATINGS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            total += 1
            if ratio < 1.0 and random.random() >= ratio:
                continue
            try:
                rows.append((int(r[0]), int(r[1]), float(r[2]), int(r[3])))
            except (ValueError, IndexError):
                continue
            kept += 1
            if len(rows) >= CHUNK:
                cur.executemany("INSERT INTO ratings VALUES (?,?,?,?)", rows)
                rows.clear()
                conn.commit()
    if rows:
        cur.executemany("INSERT INTO ratings VALUES (?,?,?,?)", rows)
    conn.commit()
    print(
        f"[ratings] done in {time.time() - start:.1f}s -> kept {kept}/{total} (ratio {ratio})",
        flush=True,
    )


def build_popularity_cache(conn):
    """Materialise a popularity score per movie (count-weighted average)."""
    start = _timed("popularity cache")
    conn.execute("DROP TABLE IF EXISTS popularity")
    conn.execute("""
        CREATE TABLE popularity AS
        SELECT movieId,
               COUNT(*)            AS cnt,
               AVG(rating)         AS avg_rating,
               COUNT(*) * AVG(rating) / (COUNT(*) + 50.0) AS score
        FROM ratings
        GROUP BY movieId
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pop_score ON popularity(score DESC)")
    conn.commit()
    print(f"[popularity] built in {time.time() - start:.1f}s", flush=True)


def import_links(conn):
    start = _timed("links")
    conn.execute("DELETE FROM movie_links")
    cur = conn.cursor()
    rows = []
    with open(LINKS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            try:
                rows.append((int(r[0]), r[1] if len(r) > 1 else None, r[2] if len(r) > 2 else None, None))
            except (ValueError, IndexError):
                continue
            if len(rows) >= CHUNK:
                cur.executemany("INSERT OR REPLACE INTO movie_links VALUES (?,?,?,?)", rows)
                rows.clear()
    if rows:
        cur.executemany("INSERT OR REPLACE INTO movie_links VALUES (?,?,?,?)", rows)
    conn.commit()
    print(f"[links] done in {time.time() - start:.1f}s -> {conn.total_changes} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="fraction of ratings to keep (0..1); 1.0 = full 32M set")
    ap.add_argument("--full", action="store_true", help="shorthand for --ratio 1.0")
    ap.add_argument("--skip-ratings", action="store_true")
    args = ap.parse_args()

    ratio = 1.0 if args.full else args.ratio
    init_db()
    conn = get_conn()
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    try:
        import_movies(conn)
        import_tags(conn)
        import_links(conn)
        if not args.skip_ratings:
            import_ratings(conn, ratio)
        build_popularity_cache(conn)
        print("\nImport complete.", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
