"""Seed the MovieStore site with demo accounts.

- real accounts: heaviest-rating MovieLens users copied from the `ratings`
  table into `users` + `user_ratings` (username `ml_u<mlUserId>`,
  password `ml12345`).
- synthetic accounts: scripted genre personalities that rate the popular
  catalog (username `synth_<nn>_<primary_genre>`, password `synth123`).

Idempotent: usernames that already exist are skipped, so re-running is safe.

Usage:
    python -m seed_ml_users --real 50 --synth 50
"""
import argparse
import random

import database
from auth import hash_password

REAL_PASSWORD = "ml12345"
SYNTH_PASSWORD = "synth123"

ARCHETYPES = [
    ("Drama", "Mystery"),
    ("Mystery", "Thriller"),
    ("Comedy", "Romance"),
    ("Action", "Thriller"),
    ("Sci-Fi", "Action"),
    ("Romance", "Drama"),
    ("Horror", "Thriller"),
    ("War", "Drama"),
    ("Documentary", "Drama"),
    ("Animation", "Comedy"),
]


def seed_real_users(cur, n, min_ratings, max_per_user):
    rows = cur.execute(
        """SELECT userId
             FROM ratings
            GROUP BY userId
           HAVING COUNT(*) >= ?
            ORDER BY COUNT(*) DESC
            LIMIT ?""",
        (min_ratings, n),
    ).fetchall()
    imported = 0
    for (ml_uid,) in rows:
        username = f"ml_u{ml_uid}"
        if cur.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
            continue
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(REAL_PASSWORD)),
        )
        user_id = cur.lastrowid
        rated = cur.execute(
            """SELECT movieId, rating
                 FROM ratings
                WHERE userId = ?
                ORDER BY timestamp DESC
                LIMIT ?""",
            (ml_uid, max_per_user),
        ).fetchall()
        cur.executemany(
            "INSERT INTO user_ratings (user_id, movieId, rating) VALUES (?, ?, ?)",
            [(user_id, r["movieId"], r["rating"]) for r in rated],
        )
        imported += 1
        print(f"  real  {username:<14} {len(rated):>5} ratings", flush=True)
    return imported


def seed_synthetic_users(cur, n, rng):
    cand = cur.execute(
        """SELECT p.movieId, m.genres
             FROM popularity p JOIN movies m ON m.movieId = p.movieId
            WHERE p.cnt >= 50
            ORDER BY p.score DESC
            LIMIT 3000"""
    ).fetchall()
    universe = [(r["movieId"], {g.strip() for g in r["genres"].split("|") if g.strip()})
                for r in cand]
    if not universe:
        raise SystemExit("no popular movies found -- is the DB imported?")
    created = 0
    for i in range(n):
        primary, secondary = ARCHETYPES[i % len(ARCHETYPES)]
        username = f"synth_{i + 1:02d}_{primary.lower()}"
        if cur.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
            continue
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(SYNTH_PASSWORD)),
        )
        user_id = cur.lastrowid
        k = rng.randint(40, 120)
        picks = rng.sample(universe, min(k, len(universe)))
        rows = []
        for movie_id, genres in picks:
            if primary in genres:
                base, sd = 4.3, 0.4
            elif secondary in genres:
                base, sd = 3.8, 0.5
            else:
                base, sd = 2.5, 0.9
            rating = max(0.5, min(5.0, round(rng.gauss(base, sd) * 2) / 2))
            rows.append((user_id, movie_id, rating))
        cur.executemany(
            "INSERT INTO user_ratings (user_id, movieId, rating) VALUES (?, ?, ?)",
            rows,
        )
        created += 1
        print(f"  synth {username:<14} {len(rows):>5} ratings", flush=True)
    return created


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=int, default=50)
    ap.add_argument("--synth", type=int, default=50)
    ap.add_argument("--min-ratings", type=int, default=50)
    ap.add_argument("--max-per-user", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total_real = total_synth = 0
    with database.cursor() as (cur, _):
        total_real = seed_real_users(cur, args.real, args.min_ratings, args.max_per_user)
        total_synth = seed_synthetic_users(cur, args.synth, rng)
        n_ur = cur.execute("SELECT COUNT(*) FROM user_ratings").fetchone()[0]
        n_users = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    print(f"\nImported {total_real} real + {total_synth} synthetic users.")
    print(f"users rows: {n_users}, user_ratings rows: {n_ur}")
    print("Logins: real users -> username `ml_u<id>` / ml12345  |  synthetic -> `synth_01_drama` style / synth123")


if __name__ == "__main__":
    main()