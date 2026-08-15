"""Fetch real TMDB poster paths for every movie and store them in movie_links.

The site always works without this: movies without a fetched poster use the
auto-generated gradient posters. Run this once with a TMDB_KEY to give every
movie its real artwork. It is resumable - re-run to pick up where it stopped
(404 "no record" results are marked and never re-fetched).

Usage:
    $env:TMDB_KEY="..." ; python backend/fetch_posters.py            # all movies
    python backend/fetch_posters.py --limit 5000 --workers 6        # partial
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import TMDB_API_KEY
from database import cursor

BASE = "https://api.themoviedb.org/3/movie/{tmdb_id}"


class RateLimiter:
    """Token bucket throttled to the TMDB free tier (~45 requests / 10 s)."""

    def __init__(self, rate: float = 4.5, burst: int = 40):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.ts) * self.rate)
            self.ts = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            time.sleep((1 - self.tokens) / self.rate)
            self.tokens = 0


def pending_rows(limit: int):
    with cursor() as (cur, _):
        rows = cur.execute(
            """SELECT l.movieId, l.tmdbId FROM movie_links l
               JOIN popularity p ON p.movieId = l.movieId
               WHERE l.tmdbId IS NOT NULL
                 AND (l.poster_path IS NULL OR l.poster_path = '')
               ORDER BY p.score DESC"""
        ).fetchall()
    return rows[:limit] if limit else rows


def fetch_one(r, session, limiter):
    mid = r["movieId"]
    tmdb = str(r["tmdbId"])
    if not tmdb.isdigit():
        _mark(mid, "none")
        return mid, "none"
    try:
        limiter.acquire()
        resp = session.get(BASE.format(tmdb_id=int(tmdb)),
                           params={"api_key": TMDB_API_KEY}, timeout=20)
        if resp.status_code == 404:
            _mark(mid, "none")
            return mid, "none"
        resp.raise_for_status()
        pp = resp.json().get("poster_path")
        if pp:
            _mark(mid, pp)
            return mid, pp
        return mid, "no-path"
    except Exception as e:
        return mid, f"err:{e}"


def _mark(mid, value):
    with cursor() as (cur, _):
        cur.execute("UPDATE movie_links SET poster_path = ? WHERE movieId = ?", (value, mid))


def fetch(workers: int = 4, limit: int = 0):
    if not TMDB_API_KEY:
        print("No TMDB_KEY set - nothing to do (generated posters stay in use).")
        return 0
    rows = pending_rows(limit)
    if not rows:
        print("Nothing to fetch - every movie already has a poster path.")
        return 0
    print(f"Fetching posters for {len(rows)} movies ({workers} workers, ~4.5 req/s)…", flush=True)
    limiter = RateLimiter(4.5)
    stats = {"done": 0, "got": 0, "none": 0, "err": 0}
    t0 = time.time()
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, r, session, limiter) for r in rows]
        for fut in as_completed(futs):
            mid, out = fut.result()
            stats["done"] += 1
            if out == "none":
                stats["none"] += 1
            elif out.startswith("err:"):
                stats["err"] += 1
            else:
                stats["got"] += 1
            if stats["done"] % 250 == 0:
                el = time.time() - t0
                rate = stats["done"] / max(el, 1)
                eta = (len(rows) - stats["done"]) / max(rate, 1e-6)
                print(f"  {stats['done']}/{len(rows)}  got={stats['got']} none={stats['none']} "
                      f"err={stats['err']}  {rate:.1f}/s  eta {eta / 60:.0f} min", flush=True)
    el = time.time() - t0
    print(f"Done in {el / 60:.1f} min: got={stats['got']}, no-record={stats['none']}, "
          f"errors={stats['err']}. Re-run to resume.", flush=True)
    return stats["got"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all movies")
    args = ap.parse_args()
    sys.exit(fetch(args.workers, args.limit))


if __name__ == "__main__":
    main()
