"""Fetch real TMDB poster paths for popular movies and store them in movie_links.

Optional enhancement: without a TMDB_KEY it does nothing and the site uses the
auto-generated gradient posters. With a key it processes movies in popularity
order (top first) and can be re-run safely to resume / fetch more.

Usage:
    $env:TMDB_KEY="..." ; python backend/fetch_posters.py --limit 20000
"""
import argparse
import sys
import time

import requests

from config import TMDB_API_KEY
from database import cursor

BASE = "https://api.themoviedb.org/3/movie/{tmdb_id}"


def fetch(limit: int, sleep: float = 0.3):
    if not TMDB_API_KEY:
        print("No TMDB_KEY set — nothing to do (generated posters stay in use).")
        return 0
    with cursor() as (cur, conn):
        rows = cur.execute(
            """SELECT l.movieId, l.tmdbId FROM movie_links l
               JOIN popularity p ON p.movieId = l.movieId
               WHERE l.tmdbId IS NOT NULL AND (l.poster_path IS NULL OR l.poster_path = '')
               ORDER BY p.score DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    print(f"Fetching posters for {len(rows)} movies…", flush=True)
    done = 0
    for i, r in enumerate(rows):
        if not r["tmdbId"] or not str(r["tmdbId"]).isdigit():
            continue
        url = BASE.format(tmdb_id=int(r["tmdbId"]))
        try:
            resp = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=20)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            pp = resp.json().get("poster_path")
            if pp:
                with cursor() as (cur, conn):
                    cur.execute("UPDATE movie_links SET poster_path = ? WHERE movieId = ?",
                                (pp, r["movieId"]))
                done += 1
        except Exception as e:
            print(f"  err {r['movieId']}: {e}", flush=True)
            time.sleep(1)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done", flush=True)
        time.sleep(sleep)
    print(f"Fetched {done} posters. Re-run to continue from the top.", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()
    sys.exit(fetch(args.limit, args.sleep))


if __name__ == "__main__":
    main()
