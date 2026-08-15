"""Hybrid movie posters.

Real TMDB artwork when a poster path has been fetched (see fetch_posters.py);
otherwise a deterministic Pillow-generated gradient poster, cached on first use
under `frontend/static/posters/`.
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import BASE_DIR
from database import cursor

TMDB_IMG = "https://image.tmdb.org/t/p/w342"

POSTER_DIR = BASE_DIR / "frontend" / "static" / "posters"
POSTER_SIZE = (300, 450)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    path = None
    for cand in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arialbd.ttf",
                 r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf"):
        if Path(cand).exists():
            path = cand
            break
    f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def hash_color(movie_id: int):
    h = (movie_id * 2654435761) & 0xFFFFFFFF
    hue = h % 360
    sat = 45 + (h % 30)
    light = 30 + ((h >> 8) % 18)
    return hue, sat, light


def generate_poster(movie_id: int, title: str, genres: list[str], out_path: Path) -> Path:
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    w, hgt = POSTER_SIZE
    hue, sat, light = hash_color(movie_id)
    img = Image.new("RGB", (w, hgt))
    d = ImageDraw.Draw(img)

    top = f"hsl({hue}, {sat}%, {light}%)"
    h2 = (hue + 38) % 360
    bot = f"hsl({h2}, {sat}%, 8%)"
    _gradient(img, d, top, bot)

    for _ in range(28):
        x = (movie_id * 7919 + _ * 104729) % w
        y = (movie_id * 104729 + _ * 7919) % hgt
        a = (hash_color(movie_id + _)[0]) / 360.0
        d.ellipse([x, y, x + 34 + (movie_id + _) % 46, y + 34 + (movie_id + _) % 46],
                  fill=f"hsl({int(a * 360)}, 70%, 55%)", outline=None)

    d.rectangle([0, 0, w, 8], fill="#ff9900")
    d.rectangle([0, hgt - 42, w, hgt], fill=(13, 15, 18))

    d.text((16, hgt - 34), "MovieStore", font=_font(16), fill="#ff9900")
    d.text((16, hgt - 18), "STREAM · FAVORITE · WATCH LATER", font=_font(10), fill=(150, 152, 155))

    title_clean = title.rsplit("(", 1)[0].strip() if title.endswith(")") and "(" in title else title
    d.text((16, 22), _wrap(title_clean, _font(34), w - 32), font=_font(34),
           fill=(255, 255, 255))
    genre_line = ", ".join(genres[:3]) if genres else ""
    if genre_line:
        d.text((16, hgt - 96), genre_line, font=_font(16), fill=(210, 210, 212))

    img.save(out_path, "PNG")
    return out_path


def _wrap(text: str, font, max_w: int, max_lines: int = 4) -> str:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if font.getbbox(trial)[2] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        return " ".join(lines[:max_lines]).rstrip() + "…"
    return "\n".join(lines)


def _gradient(img: Image.Image, d: ImageDraw.ImageDraw, top: str, bot: str):
    w, hgt = img.size
    step = max(1, hgt // 120)
    for y in range(0, hgt, step):
        t = y / max(1, hgt - 1)
        d.line([(0, y), (w, y)], fill=_lerp_hsl(top, bot, t))


def _lerp_hsl(top: str, bot: str, t: float):
    import re
    m1 = re.match(r"hsl\((\d+), (\d+)%, (\d+)%\)", top)
    m2 = re.match(r"hsl\((\d+), (\d+)%, (\d+)%\)", bot)
    h1, s1, l1 = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
    h2, s2, l2 = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
    h = h1 + (h2 - h1) * t
    s = s1 + (s2 - s1) * t
    l = l1 + (l2 - l1) * t
    return f"hsl({round(h)}, {round(s)}%, {round(l)}%)"


def get_poster_path(movie_id: int) -> str | None:
    with cursor() as (cur, _):
        row = cur.execute("SELECT poster_path FROM movie_links WHERE movieId = ?", (movie_id,)).fetchone()
    return row["poster_path"] if row and row["poster_path"] else None


def poster_url(movie_id: int) -> str:
    pp = get_poster_path(movie_id)
    if pp:
        return f"{TMDB_IMG}/{pp}"
    return f"/api/poster/{movie_id}.png"


def ensure_poster_file(movie_id: int) -> Path:
    out = POSTER_DIR / f"{movie_id}.png"
    if out.exists():
        return out
    from recommend import get_movie
    m = get_movie(movie_id)
    if m is None:
        raise FileNotFoundError(movie_id)
    return generate_poster(movie_id, m["title"], m["genre_list"], out)
