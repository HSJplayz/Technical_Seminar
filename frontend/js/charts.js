/* Tiny dependency-free charts (works offline for the seminar). */
window.Charts = (() => {
  function prepare(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const cw = rect.width || w;
    canvas.width = cw * dpr;
    canvas.height = h * dpr;
    canvas.style.width = cw + "px";
    canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cw, h);
    return { ctx, cw, h };
  }

  function line(canvas, opts) {
    const { ctx, cw, h } = prepare(canvas, opts.height || 320);
    const series = opts.series || [];
    const xlabels = opts.labels || [];
    const pad = { l: 46, r: 16, t: 26, b: 34 };
    const W = cw - pad.l - pad.r, H = h - pad.t - pad.b;
    let minX = 0, maxX = Math.max(series.length ? Math.max(...series.map(s => s.points.length)) : 1, 1);
    let allY = series.flatMap(s => s.points);
    let minY = Math.min(...allY), maxY = Math.max(...allY);
    if (minY === maxY) maxY = minY + 1;
    const padY = (maxY - minY) * 0.12;
    minY -= padY; maxY += padY;

    ctx.font = "11px Segoe UI";
    ctx.fillStyle = "#565959";
    ctx.strokeStyle = "#d5d9d9";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = pad.t + H - (g / 4) * H;
      const val = minY + (g / 4) * (maxY - minY);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + W, y); ctx.stroke();
      ctx.fillText(val.toFixed(1), 4, y + 4);
    }
    if (xlabels.length > 1) {
      for (let i = 0; i < xlabels.length; i++) {
        const x = pad.l + (i / (xlabels.length - 1)) * W;
        ctx.fillText(String(xlabels[i]), x - 12, h - 12);
      }
    }
    ctx.lineWidth = 2.5;
    const colors = opts.colors || ["#ff9900", "#007185", "#7d3c98"];
    series.forEach((s, si) => {
      ctx.strokeStyle = colors[si % colors.length];
      ctx.beginPath();
      s.points.forEach((p, i) => {
        const x = pad.l + (i / Math.max(1, s.points.length - 1)) * W;
        const y = pad.t + H - ((p - minY) / (maxY - minY)) * H;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      // dots
      ctx.fillStyle = colors[si % colors.length];
      s.points.forEach((p, i) => {
        const x = pad.l + (i / Math.max(1, s.points.length - 1)) * W;
        const y = pad.t + H - ((p - minY) / (maxY - minY)) * H;
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
      });
    });
    // legend
    let lx = pad.l;
    series.forEach((s, si) => {
      ctx.fillStyle = colors[si % colors.length];
      ctx.fillRect(lx, 8, 12, 4);
      ctx.fillStyle = "#0f1111";
      ctx.font = "12px Segoe UI";
      ctx.fillText(s.name, lx + 16, 13);
      lx += 18 + ctx.measureText(s.name).width + 16;
    });
  }

  function bar(canvas, opts) {
    const { ctx, cw, h } = prepare(canvas, opts.height || 300);
    const labels = opts.labels || [], values = opts.values || [];
    const pad = { l: 10, r: 10, t: 26, b: 44 };
    const W = cw - pad.l - pad.r, H = h - pad.t - pad.b;
    const max = Math.max(...values.map(Math.abs), 1);
    ctx.font = "11px Segoe UI";
    ctx.fillStyle = "#565959";
    const n = labels.length;
    const slot = W / n;
    const bw = Math.min(36, slot * 0.6);
    labels.forEach((lab, i) => {
      const v = values[i];
      const x = pad.l + slot * i + (slot - bw) / 2;
      const bh = (Math.abs(v) / max) * H;
      const y = v >= 0 ? pad.t + H - bh : pad.t + H;
      ctx.fillStyle = v >= 0 ? "#007185" : "#c40000";
      ctx.fillRect(x, y, bw, Math.max(1, bh));
      ctx.fillStyle = "#565959";
      const tl = lab.length > 11 ? lab.slice(0, 10) + "…" : lab;
      ctx.save();
      ctx.translate(x + bw / 2, pad.t + H + 8);
      ctx.rotate(-0.55);
      ctx.textAlign = "right";
      ctx.fillText(tl, 0, 0);
      ctx.restore();
    });
  }

  return { line, bar };
})();
