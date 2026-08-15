import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(os.environ.get("ML32M_DIR", r"D:\seminar\ml-32m"))

MOVIES_CSV = DATASET_DIR / "movies.csv"
RATINGS_CSV = DATASET_DIR / "ratings.csv"
TAGS_CSV = DATASET_DIR / "tags.csv"
LINKS_CSV = DATASET_DIR / "links.csv"

TMDB_API_KEY = os.environ.get("TMDB_KEY", "")

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "movielens.db"

SECRET_KEY = os.environ.get("APP_SECRET", "seminar-demo-secret-change-me")
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 14
