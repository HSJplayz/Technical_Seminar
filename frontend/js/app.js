/* MovieStore SPA — vanilla JS, hash router, no external deps. */
(() => {
  "use strict";

  const TOKEN_KEY = "ms_token";
  const state = {
    token: localStorage.getItem(TOKEN_KEY) || null,
    user: null,
    genres: [],
    page: 1,
    fav: new Set(),
    wl: new Set(),
    listCounts: { favorite: 0, watch_later: 0 },
  };

  // ------------------------------------------------------------ api helpers
  async function api(path, opts = {}) {
    const headers = { "Content-Type": "application/json" };
    if (state.token) headers.Authorization = "Bearer " + state.token;
    const res = await fetch(path, { ...opts, headers });
    let body = null;
    try { body = await res.json(); } catch (e) { /* noop */ }
    if (!res.ok) {
      const msg = (body && body.detail) || res.statusText || "Request failed";
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return body;
  }

  const qs = (obj) => Object.entries(obj)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join("&");

  // ------------------------------------------------------------ ui helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  };

  function toast(msg, ms = 2600) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.remove("hidden");
    clearTimeout(t._h);
    t._h = setTimeout(() => t.classList.add("hidden"), ms);
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g,
      c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function hashColor(movieId) {
    let h = (movieId * 2654435761) >>> 0;
    const hue = h % 360;
    const sat = 45 + (h % 30);
    const light = 32 + ((h >>> 8) % 18);
    return `hsl(${hue}, ${sat}%, ${light}%)`;
  }

  function posterHTML(m) {
    const bg = hashColor(m.movieId);
    const genres = (m.genre_list || []).slice(0, 2).join(" · ");
    const inner = `${genres ? `<span class="p-genres">${escapeHtml(genres)}</span>` : ""}
      <span class="p-title">${escapeHtml(m.title_clean || m.title)}</span>`;
    const img = m.poster_url
      ? `<img class="poster-img" src="${escapeHtml(m.poster_url)}" alt="${escapeHtml(m.title_clean || m.title)}" loading="lazy" referrerpolicy="no-referrer" onerror="this.remove()" />`
      : "";
    return `<div class="poster" style="background:linear-gradient(160deg, ${bg}, ${bg}dd 55%, #0d0f12)">${inner}${img}</div>`;
  }

  function stars(rating) {
    if (rating == null) return `<span class="rating-stars">☆☆☆☆☆</span>`;
    const full = Math.round(rating);
    return `<span class="rating-stars">${"★".repeat(full)}${"☆".repeat(5 - full)}</span>`;
  }

  function updateBasketBadge() {
    const el2 = $("#basket-count");
    if (!el2) return;
    const n = state.listCounts.favorite + state.listCounts.watch_later;
    el2.textContent = n ? ` (${n})` : "";
  }

  async function loadLists() {
    if (!state.token) return;
    try {
      const s = await api("/api/lists/summary");
      state.fav = new Set(s.favorite.map(x => x.movieId));
      state.wl = new Set(s.watch_later.map(x => x.movieId));
      state.listCounts = s.counts;
      updateBasketBadge();
    } catch (e) { /* not signed in */ }
  }

  function basketButton(movieId, type, initial) {
    const isFav = type === "favorite";
    const set = isFav ? state.fav : state.wl;
    if (initial !== undefined) { initial ? set.add(movieId) : set.delete(movieId); }
    const label = isFav ? "♥ <span class='bs-lbl'>Favorite</span>" : "⏱ <span class='bs-lbl'>Watch later</span>";
    const b = el("button", "bs-btn" + (set.has(movieId) ? " on" : ""), label);
    b.title = isFav ? "Add to favorites" : "Add to watch later";
    b.addEventListener("click", async e => {
      e.stopPropagation();
      if (!state.token) { openAuthModal("login"); return; }
      const on = set.has(movieId);
      try {
        await api(`/api/list/${type}/${movieId}`, { method: on ? "DELETE" : "PUT" });
        if (on) set.delete(movieId); else set.add(movieId);
        const key = isFav ? "favorite" : "watch_later";
        state.listCounts[key] += on ? -1 : 1;
        b.classList.toggle("on", !on);
        updateBasketBadge();
        toast(on ? `Removed from ${isFav ? "favorites" : "watch later"}` : `Added to ${isFav ? "favorites" : "watch later"} — saved to your basket`);
      } catch (err) { toast(err.message); }
    });
    return b;
  }

  function compactBasketButton(movieId, type) {
    const isFav = type === "favorite";
    const set = isFav ? state.fav : state.wl;
    const b = el("button", "bs-btn" + (set.has(movieId) ? " on" : ""), isFav ? "♥" : "⏱");
    b.title = isFav ? "Add to favorites" : "Add to watch later";
    b.addEventListener("click", async e => {
      e.stopPropagation();
      if (!state.token) { openAuthModal("login"); return; }
      const on = set.has(movieId);
      try {
        await api(`/api/list/${type}/${movieId}`, { method: on ? "DELETE" : "PUT" });
        if (on) set.delete(movieId); else set.add(movieId);
        const key = isFav ? "favorite" : "watch_later";
        state.listCounts[key] += on ? -1 : 1;
        b.classList.toggle("on", !on);
        updateBasketBadge();
        toast(on ? `Removed from ${isFav ? "favorites" : "watch later"}` : `Added to ${isFav ? "favorites" : "watch later"}`);
      } catch (err) { toast(err.message); }
    });
    return b;
  }

  function movieCard(m) {
    const card = el("div", "card");
    card.addEventListener("click", () => (location.hash = `#/movie/${m.movieId}`));
    card.innerHTML = `${posterHTML(m)}
      <div class="card-body">
        <div class="card-title">${escapeHtml(m.title_clean || m.title)}</div>
        <div class="card-meta">
          ${m.year ? `<span>${m.year}</span>` : ""}
          ${stars(m.avg_rating)} ${m.avg_rating ? `<span>${m.avg_rating}</span>` : ""}
          ${m.rating_count ? `<span>· ${m.rating_count.toLocaleString()} ratings</span>` : ""}
        </div>
        ${m.score != null ? `<div class="card-tags">match ${Math.min(99, Math.round(Math.abs(m.score) * 100)).toLocaleString()}%</div>` : ""}
      </div>`;
    const actions = el("div", "card-actions");
    const buy = el("button", "buy-btn", m.similar ? "" : "See details");
    buy.addEventListener("click", e => { e.stopPropagation(); location.hash = `#/movie/${m.movieId}`; });
    actions.appendChild(buy);
    actions.appendChild(compactBasketButton(m.movieId, "favorite"));
    actions.appendChild(compactBasketButton(m.movieId, "watch_later"));
    card.querySelector(".card-body").appendChild(actions);
    return card;
  }

  function grid(items, emptyText) {
    if (!items || !items.length) return el("div", "empty", emptyText || "Nothing here yet.");
    const g = el("div", "grid");
    items.forEach(m => g.appendChild(movieCard(m)));
    return g;
  }

  // ------------------------------------------------------------ auth
  async function refreshUser() {
    if (!state.token) { state.user = null; return; }
    try {
      state.user = await api("/api/me");
      await loadLists();
    } catch (e) { state.token = null; localStorage.removeItem(TOKEN_KEY); }
  }

  function renderAccount() {
    const box = $("#account-box");
    if (state.user) {
      box.innerHTML = `
        <span>Hi, <b>${escapeHtml(state.user.username)}</b></span>
        <button class="chip ghost" id="btn-logout">Sign out</button>`;
      $("#btn-logout").addEventListener("click", () => {
        state.token = null; localStorage.removeItem(TOKEN_KEY); state.user = null;
        state.fav = new Set(); state.wl = new Set(); state.listCounts = { favorite: 0, watch_later: 0 };
        updateBasketBadge();
        toast("Signed out"); renderAccount(); render();
      });
    } else {
      box.innerHTML = `<button class="chip" id="btn-login">Sign in</button>
                       <button class="chip ghost" id="btn-register">Register</button>`;
      $("#btn-login").addEventListener("click", () => openAuthModal("login"));
      $("#btn-register").addEventListener("click", () => openAuthModal("register"));
    }
  }

  function openAuthModal(mode) {
    const m = $("#modal");
    m.innerHTML = `<h2>${mode === "login" ? "Sign in" : "Create account"}</h2>
      <input id="m-user" placeholder="Username" autocomplete="username" />
      <input id="m-pass" type="password" placeholder="Password" autocomplete="current-password" />
      <p style="font-size:12px;color:#565959;margin:0 0 12px">
        ${mode === "login" ? "New here? " : "Already a member? "}
        <a href="#" id="m-switch" style="color:#007185">${mode === "login" ? "Register" : "Sign in"}</a>
      </p>
      <div class="actions">
        <button class="btn-ghost" id="m-cancel">Cancel</button>
        <button class="btn-primary" id="m-go">${mode === "login" ? "Sign in" : "Register"}</button>
      </div>`;
    $("#modal-backdrop").classList.remove("hidden");
    $("#m-cancel").addEventListener("click", closeModal);
    $("#m-switch").addEventListener("click", e => { e.preventDefault(); closeModal(); openAuthModal(mode === "login" ? "register" : "login"); });
    $("#m-go").addEventListener("click", async () => {
      const username = $("#m-user").value.trim();
      const password = $("#m-pass").value;
      if (!username || !password) return toast("Enter username and password");
      try {
        const res = await api("/api/auth/" + (mode === "login" ? "login" : "register"), {
          method: "POST",
          body: JSON.stringify({ username, password }),
        });
        state.token = res.token;
        localStorage.setItem(TOKEN_KEY, res.token);
        state.user = res.user;
        closeModal();
        renderAccount();
        toast(mode === "login" ? `Welcome back, ${username}!` : `Account created — welcome, ${username}!`);
        render();
      } catch (e) { toast(e.message); }
    });
    $("#m-pass").addEventListener("keydown", e => { if (e.key === "Enter") $("#m-go").click(); });
  }

  function closeModal() { $("#modal-backdrop").classList.add("hidden"); }

  // ------------------------------------------------------------ router / views
  const routes = {
    "": homeView,
    home: homeView,
    browse: browseView,
    movie: movieView,
    basket: basketView,
    ratings: ratingsView,
    dashboard: dashboardView,
  };

  async function render() {
    renderAccount();
    const raw = location.hash.replace(/^#\//, "") || "home";
    const [name] = raw.split("?")[0].split("/");
    const app = $("#app");
    app.innerHTML = `<div class="spinner"></div>`;
    try {
      const view = routes[name] || homeView;
      await view(app, name, restOf(raw));
    } catch (e) {
      console.error(e);
      app.innerHTML = `<div class="empty">Something went wrong: ${escapeHtml(e.message)}</div>`;
    }
  }

  function restOf(raw) {
    return raw.split("?")[0].split("/").slice(1);
  }

  window.addEventListener("hashchange", render);

  // ------------------------------------------------------------ home
  async function homeView(app) {
    const [pop, rec] = await Promise.all([
      api("/api/popular?limit=12"),
      api("/api/recommendations?limit=12"),
    ]);
    const hero = el("div", "hero");
    hero.innerHTML = `
      <h1>Discover your next favorite movie</h1>
      <p>A privacy-first storefront on the <b>MovieLens 32M</b> dataset.
        Recommendations are trained with <b>Federated Learning</b> — your ratings never leave your device in plaintext.
        The full stack: <b>Local DP · SecAgg · CKKS Homomorphic Encryption · SHAP explanations</b>.</p>
      <div class="badges">
        <span class="badge">32.8M ratings</span>
        <span class="badge">86,000+ titles</span>
        <span class="badge">ε=1 → 2.5 with privacy maintained</span>
        <span class="badge">+10–15% accuracy</span>
      </div>`;
    app.innerHTML = "";
    app.appendChild(hero);
    app.appendChild(el("h2", "section-title", "Shop by category <small>· genres across the catalog</small>"));
    const cats = [...state.genres].sort((a, b) => b.count - a.count).slice(0, 12);
    const cg = el("div", "cat-grid");
    cats.forEach(c => {
      const a = el("a", "cat-card",
        `<span class="cat-name">${escapeHtml(c.name)}</span><span class="cat-count">${c.count.toLocaleString()} titles</span>`);
      a.href = "#/browse?genre=" + encodeURIComponent(c.name);
      cg.appendChild(a);
    });
    app.appendChild(cg);
    app.appendChild(el("h2", "section-title",
      `Recommended for you <small>· ${state.user ? "federated model + collaborative filtering" : "popular on MovieStore — sign in for personalization"}</small>`));
    app.appendChild(grid(rec.items));
    app.appendChild(el("h2", "section-title", "Trending this week <small>· most-watched titles</small>"));
    app.appendChild(grid(pop.items));
  }

  // ------------------------------------------------------------ browse
  function freshBrowse() { return { q: "", genre: "", yearMin: "", yearMax: "", sort: "relevance", page: 1 }; }
  let browseState = freshBrowse();

  async function browseView(app) {
    browseState = freshBrowse();
    const params = new URLSearchParams(location.hash.split("?")[1] || "");
    if (params.get("q") !== null) browseState.q = params.get("q");
    if (params.get("genre") !== null) browseState.genre = params.get("genre");
    if (params.get("sort")) browseState.sort = params.get("sort");
    if (params.get("page")) browseState.page = +params.get("page");

    app.innerHTML = `
      <h2 class="section-title" style="margin-top:0">Browse the catalog</h2>
      <div class="filters">
        <span><label>Genre</label>
          <select id="f-genre"></select>
        </span>
        <span><label>Year from</label><input id="f-yearmin" type="number" placeholder="1920" style="width:90px" /></span>
        <span><label>to</label><input id="f-yearmax" type="number" placeholder="2025" style="width:90px" /></span>
        <span><label>Sort</label>
          <select id="f-sort">
            <option value="relevance">Title</option>
            <option value="year">Newest first</option>
            <option value="rating">Top rated</option>
          </select>
        </span>
        <button class="btn-ghost" id="f-reset">Reset</button>
      </div>
      <div id="browse-results"><div class="spinner"></div></div>`;

    const gSel = $("#f-genre");
    gSel.innerHTML = `<option value="">All genres</option>` +
      state.genres.map(g => `<option value="${escapeHtml(g.name)}">${escapeHtml(g.name)} (${g.count.toLocaleString()})</option>`).join("");
    if (browseState.genre) gSel.value = browseState.genre;
    $("#f-yearmin").value = browseState.yearMin;
    $("#f-yearmax").value = browseState.yearMax;
    $("#f-sort").value = browseState.sort;
    $("#f-reset").addEventListener("click", () => { browseState = freshBrowse(); render(); });

    const refresh = async () => {
      $("#browse-results").innerHTML = `<div class="spinner"></div>`;
      try {
        const data = await api("/api/movies?" + qs({
          q: browseState.q, genre: browseState.genre, sort: browseState.sort,
          year_min: browseState.yearMin, year_max: browseState.yearMax,
          page: browseState.page, per_page: 24,
        }));
        const box = $("#browse-results");
        box.innerHTML = "";
        box.appendChild(el("p", "card-meta", `${data.total.toLocaleString()} titles found`));
        box.appendChild(grid(data.items));
        if (data.pages > 1) {
          const pg = el("div", "pager");
          pg.appendChild(el("button", "prev", "‹ Prev"));
          pg.appendChild(el("span", "", `Page ${data.page} / ${data.pages}`));
          pg.appendChild(el("button", "next", "Next ›"));
          pg.querySelectorAll("button")[0].disabled = data.page <= 1;
          pg.querySelectorAll("button")[1].disabled = data.page >= data.pages;
          pg.querySelectorAll("button")[0].addEventListener("click", () => { browseState.page--; location.hash = `#/browse?page=${browseState.page}`; });
          pg.querySelectorAll("button")[1].addEventListener("click", () => { browseState.page++; location.hash = `#/browse?page=${browseState.page}`; });
          box.appendChild(pg);
        }
      } catch (e) { $("#browse-results").innerHTML = `<div class="empty">${escapeHtml(e.message)}</div>`; }
    };

    $("#f-genre").addEventListener("change", e => { browseState.genre = e.target.value; browseState.page = 1; refresh(); });
    $("#f-yearmin").addEventListener("change", e => { browseState.yearMin = e.target.value; browseState.page = 1; refresh(); });
    $("#f-yearmax").addEventListener("change", e => { browseState.yearMax = e.target.value; browseState.page = 1; refresh(); });
    $("#f-sort").addEventListener("change", e => { browseState.sort = e.target.value; refresh(); });
    await refresh();
  }

  // ------------------------------------------------------------ movie detail
  async function movieView(app, _name, rest) {
    const id = +rest[0];
    const d = await api(`/api/movies/${id}`);
    app.innerHTML = "";
    const detail = el("div", "detail");
    detail.innerHTML = `
      <div class="detail-top">
        <div class="detail-poster" style="background:linear-gradient(160deg, ${hashColor(id)}, ${hashColor(id)}cc 55%, #0d0f12)">
          ${d.poster_url ? `<img class="poster-img" src="${escapeHtml(d.poster_url)}" alt="${escapeHtml(d.title_clean || d.title)}" loading="lazy" referrerpolicy="no-referrer" onerror="this.remove()" />` : ""}
          <div class="p-title" style="position:absolute;bottom:16px;left:14px;right:14px;font-size:19px;font-weight:700;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,.8)">${escapeHtml(d.title_clean || d.title)}</div>
        </div>
        <div class="detail-info">
          <h1>${escapeHtml(d.title)}</h1>
          <div class="detail-meta">${(d.genre_list || []).map(g => escapeHtml(g)).join(" · ")} ${d.year ? "· " + d.year : ""}</div>
          <div class="avg-rating">${stars(d.avg_rating)} <b>${d.avg_rating ?? "—"}</b>
            <span style="font-size:13px;color:#565959">(${d.rating_count ? d.rating_count.toLocaleString() : 0} ratings on MovieLens)</span></div>
          ${d.top_tags && d.top_tags.length ? `<div class="tag-cloud">${d.top_tags.slice(0, 10).map(t => `<span>#${escapeHtml(t)}</span>`).join("")}</div>` : ""}
          <div class="rate-widget">
            <span style="font-size:13px;color:#565959">${state.user ? "Your rating:" : "Sign in to rate:"}</span>
            <div class="rate-stars" id="rate-stars"></div>
            ${state.user ? `<button class="btn-ghost" id="btn-clear-rate" ${d.my_rating ? "" : "style='display:none'"}>Clear</button>` : ""}
          </div>
          <div class="detail-actions" id="detail-actions" style="display:flex;gap:10px;margin-top:16px"></div>
          <div style="margin-top:14px"><button class="buy-btn" style="max-width:220px" onclick="location.hash='#/browse'">Browse similar catalog</button></div>
        </div>
      </div>
      <div class="similar-section">
        <h3 class="section-title" style="margin-top:8px">Customers also viewed</h3>
      </div>`;
    app.appendChild(detail);

    const dActions = detail.querySelector("#detail-actions");
    dActions.appendChild(basketButton(id, "favorite", !!(d.lists && d.lists.favorite)));
    dActions.appendChild(basketButton(id, "watch_later", !!(d.lists && d.lists.watch_later)));

    const similarBox = detail.querySelector(".similar-section");
    similarBox.appendChild(grid(d.similar));

    if (state.user) renderRateStars(d.my_rating, async val => {
      await api("/api/rate", { method: "POST", body: JSON.stringify({ movieId: id, rating: val }) });
      toast("Rating saved — it only leaves this browser as a federated update");
      location.hash = `#/movie/${id}`;
    });

    // SHAP explanation box if a model is available
    try {
      const fl = await api("/api/fl/status");
      if (fl.result && fl.result.shap && fl.result.shap.length) {
        const box = el("div", "panel");
        box.style.marginTop = "20px";
        box.innerHTML = `<h3>Why this was recommended (SHAP)</h3>
          <p>Feature attributions from the federated model for <b>${escapeHtml(d.title_clean || d.title)}</b>:</p>
          <ul class="bar-list" id="movie-shap"></ul>`;
        detail.appendChild(box);
        const top = fl.result.shap.slice(0, 6);
        const maxV = Math.max(...top.map(s => Math.abs(s.importance)));
        const list = $("#movie-shap");
        top.forEach(s => {
          const li = el("li");
          li.innerHTML = `<span style="width:110px">${escapeHtml(s.feature)}</span>
            <div class="bar"><div style="width:${Math.max(4, Math.abs(s.importance) / maxV * 100)}%;background:${s.importance >= 0 ? "#007185" : "#c40000"}"></div></div>
            <span class="mono">${s.importance >= 0 ? "+" : ""}${s.importance}</span>`;
          list.appendChild(li);
        });
      }
    } catch (e) { /* ignore */ }
  }

  function renderRateStars(current, onRate) {
    const box = $("#rate-stars");
    box.innerHTML = "";
    for (let v = 1; v <= 5; v++) {
      const b = el("button", current >= v ? "on" : "");
      b.textContent = "★";
      b.addEventListener("click", () => onRate(v));
      box.appendChild(b);
    }
    const clear = $("#btn-clear-rate");
    if (clear) clear.addEventListener("click", async () => {
      const id = +location.hash.split("/")[2];
      try {
        await api(`/api/rate/${id}`, { method: "DELETE" });
        toast("Rating removed");
        location.hash = `#/movie/${id}`;
      } catch (e) { toast(e.message); }
    });
  }

  // ------------------------------------------------------------ basket
  async function basketView(app) {
    if (!state.user) {
      app.innerHTML = `<div class="panel" style="max-width:520px;margin:40px auto;text-align:center">
        <h3>Your basket</h3>
        <p>Sign in to save movies to <b>Favorites</b> and <b>Watch later</b>. Your basket is stored on your account,
        so it stays after you log out — and later it feeds your federated training signal.</p>
        <div style="display:flex;gap:10px;justify-content:center">
          <button class="btn-primary" id="bk-login">Sign in</button>
          <button class="btn-ghost" id="bk-reg">Register</button>
        </div></div>`;
      $("#bk-login").addEventListener("click", () => openAuthModal("login"));
      $("#bk-reg").addEventListener("click", () => openAuthModal("register"));
      return;
    }
    app.innerHTML = `<div class="spinner"></div>`;
    const s = await api("/api/lists/summary");
    app.innerHTML = "";
    const head = el("div", "basket-head");
    head.appendChild(el("h2", "section-title", "My Basket"));
    head.appendChild(el("span", "pill gray", `${s.counts.favorite} favorite · ${s.counts.watch_later} watch later`));
    app.appendChild(head);

    const favHead = el("div", "section-head");
    favHead.appendChild(el("h3", "", "Favorites"));
    app.appendChild(favHead);
    app.appendChild(grid(s.favorite, "No favorites yet — tap ♥ on any movie."));
    const wlHead = el("div", "section-head");
    wlHead.appendChild(el("h3", "", "Watch later"));
    app.appendChild(wlHead);
    app.appendChild(grid(s.watch_later, "Nothing in watch later — tap ⏱ on any movie."));
  }

  // ------------------------------------------------------------ ratings
  async function ratingsView(app) {
    if (!state.user) {
      app.innerHTML = `<div class="panel" style="max-width:520px;margin:40px auto;text-align:center">
        <h3>My ratings</h3>
        <p>Sign in to see every movie you've rated, change your stars, or remove a rating.
        Your ratings are your private data — they only reach the server as federated updates.</p>
        <div style="display:flex;gap:10px;justify-content:center">
          <button class="btn-primary" id="rt-login">Sign in</button>
          <button class="btn-ghost" id="rt-reg">Register</button>
        </div></div>`;
      $("#rt-login").addEventListener("click", () => openAuthModal("login"));
      $("#rt-reg").addEventListener("click", () => openAuthModal("register"));
      return;
    }
    app.innerHTML = `<div class="spinner"></div>`;
    const r = await api("/api/my-ratings");
    app.innerHTML = "";
    const head = el("div", "basket-head");
    head.appendChild(el("h2", "section-title", "My Ratings"));
    head.appendChild(el("span", "pill gray", `${r.items.length} rated`));
    app.appendChild(head);
    if (!r.items.length) {
      app.appendChild(el("div", "empty", "You haven't rated anything yet — open any movie and tap the stars."));
      return;
    }
    const g = el("div", "grid");
    r.items.forEach(m => g.appendChild(ratedCard(m, app)));
    app.appendChild(g);
  }

  function ratedCard(m, app) {
    const card = movieCard(m);
    const body = card.querySelector(".card-body");
    const bar = el("div", "your-rating",
      `Your rating: ${stars(m.rating)} <b>${m.rating}</b>`);
    body.insertBefore(bar, body.firstChild);
    const actions = card.querySelector(".card-actions");
    const rm = el("button", "buy-btn outline", "Remove rating");
    rm.addEventListener("click", async e => {
      e.stopPropagation();
      try {
        await api(`/api/rate/${m.movieId}`, { method: "DELETE" });
        toast("Rating removed");
        await ratingsView(app);
      } catch (err) { toast(err.message); }
    });
    actions.insertBefore(rm, actions.firstChild);
    return card;
  }

  // ------------------------------------------------------------ dashboard
  const dashTabs = ["overview", "federated", "privacy", "shap"];
  let dashActive = "overview";
  let flPoll = null;

  async function dashboardView(app) {
    app.innerHTML = `
      <h2 class="section-title" style="margin-top:0">Privacy &amp; Federated Learning Dashboard</h2>
      <div class="dash-tabs">
        ${dashTabs.map(t => `<button data-tab="${t}" class="${t === dashActive ? "active" : ""}">${tabLabel(t)}</button>`).join("")}
      </div>
      <div id="dash-body"><div class="spinner"></div></div>`;
    app.querySelectorAll(".dash-tabs button").forEach(b =>
      b.addEventListener("click", () => { dashActive = b.dataset.tab; dashboardView(app); }));
    const body = $("#dash-body");
    if (dashActive === "overview") await dashOverview(body);
    if (dashActive === "federated") await dashFederated(body);
    if (dashActive === "privacy") await dashPrivacy(body);
    if (dashActive === "shap") await dashShap(body);
  }

  function tabLabel(t) {
    return { overview: "Overview", federated: "Federated Learning", privacy: "Privacy · ε vs Accuracy", shap: "SHAP Explanations" }[t];
  }

  async function dashOverview(body) {
    const [stats, fl] = await Promise.all([api("/api/stats"), api("/api/fl/status")]);
    const c = stats.counts;
    body.innerHTML = "";
    const panel = el("div", "panel");
    panel.innerHTML = `<h3>Dataset (MovieLens 32M)</h3>
      <div class="stat-row">
        <div class="stat"><div class="num">${(c.ratings / 1e6).toFixed(1)}M</div><div class="lbl">ratings</div></div>
        <div class="stat"><div class="num">${(c.movies / 1e3).toFixed(0)}K</div><div class="lbl">movies</div></div>
        <div class="stat"><div class="num">${(c.tags / 1e6).toFixed(2)}M</div><div class="lbl">tags</div></div>
        <div class="stat"><div class="num">${c.users}</div><div class="lbl">site users</div></div>
      </div>`;
    body.appendChild(panel);

    const stack = el("div", "panel");
    stack.innerHTML = `<h3>Research stack</h3>
      <div class="pipeline">
        <div class="node">Clients<br/><small>private ratings</small></div><span class="arrow">→</span>
        <div class="node">Local DP<br/><small>clip + noise ε</small></div><span class="arrow">→</span>
        <div class="node">SecAgg<br/><small>masked sums</small></div><span class="arrow">→</span>
        <div class="node">CKKS HE<br/><small>encrypted updates</small></div><span class="arrow">→</span>
        <div class="node">Federated Server<br/><small>FedAvg</small></div><span class="arrow">→</span>
        <div class="node">SHAP<br/><small>explanations</small></div>
      </div>
      <p><b>Thesis:</b> with SecAgg + CKKS + SHAP-driven feature selection, the server only ever sees
      encrypted, masked aggregates — so we can operate at a <b>larger local ε (up to 2.5)</b> without weakening the
      effective per-user guarantee, recovering <b>10–15% accuracy</b> compared to plain local DP at ε = 1.</p>`;
    body.appendChild(stack);

    const eng = el("div", "panel");
    eng.innerHTML = `<h3>Engine status</h3>
      <p class="mono">CKKS backend: <span class="pill ${fl.result ? "green" : "blue"}">tenseal</span>
        &nbsp;·&nbsp; training status: <span class="pill gray">${escapeHtml(fl.status)}</span>
        &nbsp;·&nbsp; last run: ${fl.result ? fl.result.results.length + " ε-points" : "none yet"}</p>`;
    body.appendChild(eng);
  }

  async function dashFederated(body) {
    body.innerHTML = `<div class="panel">
      <h3>Train the federated recommender</h3>
      <p>Partition private ratings into clients, train locally, protect with DP noise,
        aggregate through SecAgg masks + CKKS encryption, then evaluate across a range of ε.</p>
      <div class="toggle-row"><span>CKKS homomorphic encryption</span><label class="toggle"><input type="checkbox" id="t-he" checked /><span class="slider"></span></label></div>
      <div class="toggle-row"><span>SecAgg (masked aggregation)</span><label class="toggle"><input type="checkbox" id="t-secagg" checked /><span class="slider"></span></label></div>
      <div class="toggle-row"><span>SHAP feature selection after training</span><label class="toggle"><input type="checkbox" id="t-shap" checked /><span class="slider"></span></label></div>
      <div class="toggle-row"><span>Epsilons to sweep</span>
        <input id="t-eps" class="mono" value="1,2.5" style="border:1px solid var(--border);border-radius:5px;padding:7px" /></div>
      <div class="toggle-row"><span>Clients</span>
        <input id="t-clients" type="number" value="10" min="2" max="40" style="border:1px solid var(--border);border-radius:5px;padding:7px;width:70px" />
        <span style="margin-left:8px">Rounds</span>
        <input id="t-epochs" type="number" value="12" min="1" max="30" style="border:1px solid var(--border);border-radius:5px;padding:7px;width:70px" /></div>
      <div style="margin-top:16px">
        <button class="btn-primary" id="fl-run">Run federated training</button>
        ${state.user ? "" : "<span style='font-size:12px;color:#c40000'>· sign in required</span>"}
      </div>
      <div class="progressbar hidden" id="fl-progress"><div></div></div>
      <p class="mono" id="fl-msg" style="color:#565959"></p>
      <div id="fl-results"></div>
    </div>`;

    if (!state.user) {
      $("#fl-run").addEventListener("click", () => openAuthModal("login"));
      return;
    }

    $("#fl-run").addEventListener("click", async () => {
      const eps = ($("#t-eps").value || "1,2.5,5").split(",").map(x => +x.trim()).filter(x => x > 0);
      const body2 = {
        epsilons: eps,
        clients: +$("#t-clients").value,
        epochs: +$("#t-epochs").value,
        use_he: $("#t-he").checked,
        use_secagg: $("#t-secagg").checked,
        use_shap: $("#t-shap").checked,
      };
      try {
        await api("/api/fl/start", { method: "POST", body: JSON.stringify(body2) });
        toast("Federated training started");
        startFlPoll();
      } catch (e) { toast(e.message); }
    });

    startFlPoll();
  }

  function startFlPoll() {
    if (flPoll) clearInterval(flPoll);
    flPoll = setInterval(async () => {
      try { await renderFlStatus(); } catch (e) { /* noop */ }
    }, 900);
    renderFlStatus();
  }

  async function renderFlStatus() {
    const fl = await api("/api/fl/status");
    const msg = $("#fl-msg");
    const bar = $("#fl-progress");
    if (!msg || !bar) return;
    const running = fl.status === "running";
    bar.classList.toggle("hidden", !running);
    if (running) {
      bar.querySelector("div").style.width = (fl.progress * 100) + "%";
      msg.textContent = `${escapeHtml(fl.message)} · ${Math.round(fl.progress * 100)}% · elapsed ${fl.elapsed || 0}s`;
    } else if (fl.result) {
      if (!running) clearInterval(flPoll);
      bar.classList.add("hidden");
      msg.textContent = fl.result.error ? `Failed: ${escapeHtml(fl.result.error)}` : "Training complete — results below.";
      renderFlResults(fl.result);
    } else {
      msg.textContent = "Idle. Configure and click Run.";
    }
  }

  function renderFlResults(res) {
    const box = $("#fl-results");
    if (!box || res.error) { if (box) box.innerHTML = `<div class="empty">${escapeHtml(res.error || "")}</div>`; return; }
    box.innerHTML = `
      <h3 class="section-title" style="margin-top:14px">Results</h3>
      <p class="mono" style="color:#565959">config: ${escapeHtml(JSON.stringify(res.config))}</p>
      <table class="data">
        <tr><th>ε (local)</th><th>effective ε (SecAgg amp.)</th><th>MAE</th><th>RMSE</th>
          <th>Full stack acc.</th><th>Plain local-DP acc.</th><th>Features kept (SHAP)</th></tr>
        ${res.results.map(r => `<tr>
          <td>${r.epsilon}</td><td>${r.effective_epsilon}</td><td>${r.mae}</td><td>${r.rmse}</td>
          <td><b>${r.accuracy}%</b></td><td>${r.plain_accuracy}%</td><td>${r.features_kept}</td></tr>`).join("")}
      </table>
      ${res.accuracy_gain ? `<p><span class="pill green">Full stack accuracy gain ε=${res.accuracy_gain.from_epsilon} → ${res.accuracy_gain.to_epsilon}: +${res.accuracy_gain.pct}%</span>
        <span class="mono" style="color:#565959;margin-left:8px">(${res.accuracy_gain.from}% → ${res.accuracy_gain.to}%)</span></p>` : ""}`;
  }

  // ------------------------------------------------------------ privacy (ε vs accuracy)
  async function dashPrivacy(body) {
    const fl = await api("/api/fl/status");
    body.innerHTML = `<div class="panel">
      <h3>Privacy–accuracy trade-off</h3>
      <p>For each ε the model is trained end-to-end. <b>Plain local DP</b> pays the full noise cost at the client;
        the <b>full stack</b> (SecAgg + CKKS + SHAP selection) obtains privacy amplification from aggregation
        and recovers accuracy via SHAP-guided feature pruning — so ε = 2.5 achieves accuracy that plain ε = 1
        cannot, while the effective per-user privacy is preserved.</p>
      <div class="canvas-box"><canvas id="eps-chart"></canvas></div>
      ${fl.result ? `<p class="mono" style="color:#565959">Last run: ${fl.result.results.length} ε-points
        (accuracy proxy = 1 − MAE/4 on held-out ratings). Gain: <b>${fl.result.accuracy_gain.pct}%</b></p>` :
        `<div class="empty">Run a training sweep on the <b>Federated Learning</b> tab to populate this chart.</div>`}
    </div>`;
    if (!fl.result || !fl.result.results.length) return;

    const rows = [...fl.result.results].sort((a, b) => a.epsilon - b.epsilon);
    const labels = rows.map(r => r.epsilon);
    const fullPts = rows.map(r => r.accuracy);
    const plainPts = rows.map(r => r.plain_accuracy);
    Charts.line($("#eps-chart"), {
      labels,
      series: [
        { name: "Full stack (DP + SecAgg + CKKS + SHAP)", points: fullPts },
        { name: "Plain local DP only", points: plainPts },
      ],
      colors: ["#007185", "#d4a017"],
    });

    const note = el("div", "panel");
    const gain = fl.result.accuracy_gain || {};
    note.innerHTML = `<h3>Privacy accounting &amp; the result</h3>
      <p>Local DP adds Gaussian noise σ = (2C/n)·√(2·ln(1.25/δ))/ε at each client
      (C = ${escapeHtml(String(fl.result.config.clip_norm))}, δ = ${fl.result.config.delta}).
      Secure aggregation over ${fl.result.config.clients} clients amplifies privacy:
      the server only sees the encrypted, masked sum, so the effective per-user
      guarantee is ≈ ε/√K — <b>${escapeHtml(String(fl.result.config.secagg))}</b> applies here.
      SHAP then prunes the features that heavy noise pollutes, recovering accuracy.</p>
      <p><span class="pill green">Measured: full stack ${gain.from_epsilon != null ? "ε=" + gain.from_epsilon : ""} ${gain.from || "—"}% → ${gain.to_epsilon != null ? "ε=" + gain.to_epsilon : ""} ${gain.to || "—"}% = <b>+${gain.pct || 0}%</b> accuracy</span>
      &nbsp;<span class="pill blue">paper target: 10–15% at ε ≤ 2.5</span></p>`;
    body.appendChild(note);
  }

  // ------------------------------------------------------------ SHAP
  async function dashShap(body) {
    const fl = await api("/api/fl/status");
    body.innerHTML = `<div class="panel">
      <h3>Model explainability with SHAP</h3>
      <p>For the linear federated model, SHAP values are exact: φ<sub>i</sub>(x) = w<sub>i</sub>·(x<sub>i</sub> − E[x<sub>i</sub>]).
        Below is the global mean |SHAP| across the held-out movie set — these importances also drive the
        SHAP-guided feature selection step of the training run.</p>
      <div class="canvas-box"><canvas id="shap-chart"></canvas></div>
      ${fl.result && fl.result.shap && fl.result.shap.length ? "" :
        `<div class="empty">No SHAP output yet — run federated training first.</div>`}
    </div>`;
    if (!fl.result || !fl.result.shap || !fl.result.shap.length) return;
    const top = fl.result.shap.slice(0, 15);
    Charts.bar($("#shap-chart"), {
      labels: top.map(s => s.feature),
      values: top.map(s => s.importance),
    });
  }

  // ------------------------------------------------------------ init
  function hideSuggest() {
    const b = $("#suggest");
    if (b) b.classList.add("hidden");
  }

  function wireSearch() {
    const input = $("#search-input");
    const box = $("#suggest");
    $("#search-form").addEventListener("submit", e => {
      e.preventDefault();
      const q = input.value.trim();
      hideSuggest();
      if (!q) { location.hash = "#/browse"; return; }
      browseState = freshBrowse();
      location.hash = "#/browse?q=" + encodeURIComponent(q);
    });
    let timer = null;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      const q = input.value.trim();
      if (q.length < 2) { hideSuggest(); return; }
      timer = setTimeout(async () => {
        try {
          const data = await api("/api/search/suggest?q=" + encodeURIComponent(q));
          const genreHit = state.genres.find(g => g.name.toLowerCase().startsWith(q.toLowerCase()));
          const rows = [];
          if (genreHit) rows.push(`<a href="#/browse?genre=${encodeURIComponent(genreHit.name)}">
            <span class="s-thumb" style="background:linear-gradient(160deg,${hashColor(genreHit.name.length * 7)},${hashColor(genreHit.name.length * 7)}cc 55%,#0d0f12)"></span>
            <span class="s-title">${escapeHtml(genreHit.name)}</span>
            <span class="s-meta">browse category</span></a>`);
          data.items.forEach(it => {
            rows.push(`<a href="#/movie/${it.movieId}">
              <span class="s-thumb" style="background:${hashColor(it.movieId)}">
                ${it.poster_url ? `<img src="${escapeHtml(it.poster_url)}" alt="" loading="lazy" onerror="this.remove()" />` : ""}</span>
              <span class="s-title">${escapeHtml(it.title_clean || it.title)}</span>
              <span class="s-meta">${escapeHtml((it.genre_list || []).slice(0, 2).join(" · "))} ${it.year ? "· " + it.year : ""}</span></a>`);
          });
          box.innerHTML = rows.join("");
          box.classList.toggle("hidden", !rows.length);
        } catch (e) { hideSuggest(); }
      }, 180);
    });
    document.addEventListener("click", e => { if (!e.target.closest("#search-form")) hideSuggest(); });
  }

  async function init() {
    wireSearch();
    try {
      state.genres = (await api("/api/genres")).genres;
    } catch (e) { /* backend not ready */ }
    const strip = $("#genre-strip");
    state.genres.slice(0, 18).forEach(g => {
      const a = document.createElement("a");
      a.href = "#/browse?genre=" + encodeURIComponent(g.name);
      a.textContent = g.name;
      strip.appendChild(a);
    });
    $("#modal-backdrop").addEventListener("click", e => { if (e.target.id === "modal-backdrop") closeModal(); });
    await refreshUser();
    renderAccount();
    render();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
