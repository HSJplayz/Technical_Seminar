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

    h2 = (hue + 32) % 360
    _gradient(img, d, f"hsl({hue}, {sat}%, {light}%)", f"hsl({h2}, 62%, 9%)")

    # subtle center light band for depth
    for y in range(int(hgt * 0.28), int(hgt * 0.34)):
        a = int(26 * (1 - abs(y - hgt * 0.31) / (hgt * 0.03)))
        d.line([(0, y), (w, y)], fill=(255, 255, 255, max(0, a)))

    d.rectangle([0, 0, w, 10], fill="#ff9900")
    d.rectangle([0, hgt - 46, w, hgt], fill=(13, 15, 18))

    d.text((16, hgt - 38), "MovieStore", font=_font(17), fill="#ff9900")
    d.text((16, hgt - 20), "MOVIELENS 32M", font=_font(11), fill=(150, 152, 155))

    title_clean = _strip_parens(title)
    big = _font(38)
    title_lines = _wrap_lines(title_clean, big, w - 48, max_lines=3)
    lh = int(big.getbbox("Ag")[3] * 1.18)
    block_h = lh * len(title_lines)
    start_y = max(120, int(hgt * 0.30))
    ty = start_y
    for line in title_lines:
        tw = big.getbbox(line)[2]
        tx = (w - tw) // 2
        d.text((tx + 2, ty + 3), line, font=big, fill=(0, 0, 0))
        d.text((tx, ty), line, font=big, fill=(255, 255, 255))
        ty += lh

    genre_line = ", ".join(genres[:3]) if genres else ""
    if genre_line:
        sm = _font(15)
        gw = sm.getbbox(genre_line)[2]
        d.text(((w - gw) // 2, start_y + block_h + 12), genre_line, font=sm, fill=(240, 240, 240))

    year = _extract_year(title)
    if year:
        ym = _font(20)
        yw = ym.getbbox(year)[2]
        d.text(((w - yw) // 2, start_y + block_h + 40), year, font=ym, fill="#ff9900")

    img.save(out_path, "PNG")
    return out_path


def _strip_parens(title: str) -> str:
    return title.rsplit("(", 1)[0].strip() if title.endswith(")") and "(" in title else title


def _extract_year(title: str) -> str | None:
    if "(" in title and title.rstrip().endswith(")"):
        seg = title[title.rfind("(") + 1:title.rfind(")")]
        if seg.isdigit():
            return seg
    return None


def _wrap_lines(text: str, font, max_w: int, max_lines: int = 3) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if font.getbbox(trial)[2] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        return [" ".join(lines[:max_lines]).rstrip() + "…"]
    return lines


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
    return row["poster_path"] if row and row["poster_path"] and row["poster_path"] != "none" else None


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
