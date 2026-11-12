function noop_orphan_removed(){}
async function getJson(url) {
  const r = await fetch(url, {headers: {"accept": "application/json"}});
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return await r.json();
}

function setDl(root, data) {
  if (!root) return;
  for (const dd of root.querySelectorAll("dd[data-k]")) {
    const key = dd.getAttribute("data-k");
    const v = key.split(".").reduce((o, k) => (o == null ? undefined : o[k]), data);
    if (v === undefined || v === null) dd.textContent = "—";
    else if (typeof v === "boolean") {
      dd.textContent = v ? "true" : "false";
      dd.className = v ? "good" : "bad";
    } else {
      dd.textContent = String(v);
    }
  }
}

function drawChart(counts) {
  const root = document.getElementById("run-chart");
  if (!root) return;
  root.innerHTML = "";
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    root.textContent = "no decisions";
    return;
  }
  const W = 480, H = 180, pad = 30;
  const max = Math.max(...entries.map(e => e[1]));
  const bw = Math.max(24, Math.floor((W - pad * 2) / entries.length) - 10);
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("width", "100%");
  entries.forEach(([reason, count], i) => {
    const h = Math.round((count / max) * (H - pad * 2));
    const x = pad + i * (bw + 10);
    const y = H - pad - h;
    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", bw);
    rect.setAttribute("height", h);
    rect.setAttribute("fill", "#2c6fa6");
    svg.appendChild(rect);
    const tcount = document.createElementNS(svgNS, "text");
    tcount.setAttribute("x", x + bw / 2);
    tcount.setAttribute("y", y - 4);
    tcount.setAttribute("text-anchor", "middle");
    tcount.setAttribute("font-size", "11");
    tcount.setAttribute("fill", "#1b2530");
    tcount.textContent = count;
    svg.appendChild(tcount);
    const treason = document.createElementNS(svgNS, "text");
    treason.setAttribute("x", x + bw / 2);
    treason.setAttribute("y", H - pad + 14);
    treason.setAttribute("text-anchor", "middle");
    treason.setAttribute("font-size", "10");
    treason.setAttribute("fill", "#607d8b");
    treason.textContent = reason;
    svg.appendChild(treason);
  });
  root.appendChild(svg);
}

function drawRunTable(rows) {
  const tb = document.querySelector("#run-decisions tbody");
  if (!tb) return;
  tb.innerHTML = "";
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="6" class="sub">no decisions</td></tr>';
    return;
  }
  for (const r of rows.slice().reverse()) {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + esc(r.seq) + "</td><td>" +
                   (typeof r.t === "number" ? r.t.toFixed(3) : "") + "</td><td>" +
                   esc(r.reason || "") + "</td><td>" + esc(r.stock || "") +
                   "</td><td class='" + (r.approved ? "good" : "bad") + "'>" +
                   esc(r.approved) + "</td><td>" +
                   (r.order_ref == null ? "" : r.order_ref) + "</td>";
    tb.appendChild(tr);
  }
}

async function loadDashboard() {
  const status = await getJson("/status");
  setDl(document.getElementById("status"), status);
  const counts = await getJson("/api/counts");
  const root = document.getElementById("counts");
  root.innerHTML = "";
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { root.textContent = "no decisions"; return; }
  for (const [k, v] of entries) {
    const d = document.createElement("div");
    d.className = "count";
    d.innerHTML = '<div class="num">' + esc(v) + '</div><div class="lbl">' + esc(k) + '</div>';
    root.appendChild(d);
  }
  const latest = await getJson("/api/decisions?limit=10");
  const tb = document.querySelector("#latest tbody");
  tb.innerHTML = "";
  for (const r of latest) {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + r.seq + "</td><td>" + r.t.toFixed(3) + "</td><td>" + esc(r.reason) +
                   "</td><td>" + esc(r.stock || "") + "</td><td>" + esc(r.approved) +
                   "</td><td>" + (r.order_ref == null ? "" : esc(r.order_ref)) + "</td>";
    tb.appendChild(tr);
  }
}

if (document.getElementById("status")) {
  loadDashboard().catch(e => {
    const s = document.getElementById("status");
    if (s) s.textContent = e.message;
  });
}

// ---- Run demo ----
const runForm = document.getElementById("run-form");
if (runForm) {
  runForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = document.getElementById("run-btn");
    const status = document.getElementById("run-status");
    const result = document.getElementById("run-result");
    const n = parseInt(document.getElementById("run-n").value, 10) || 1000;
    const seed = parseInt(document.getElementById("run-seed").value, 10) || 0;
    btn.disabled = true;
    status.textContent = "running " + n + " messages…";
    result.innerHTML = "";
    try {
      const r = await fetch("/api/run", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify({n, seed}),
      });
      const data = await r.json();
      if (!data.ok) throw new Error(data.error || "run failed");
      status.textContent = data.n_messages + " messages, " + data.n_orders +
                           " orders, " + data.elapsed_s + " s, verified " +
                           data.log.verified;
      drawChart(data.counts);
      drawRunTable(await getJson("/api/last-run/decisions?limit=20"));
      setDl(document.getElementById("status"), await getJson("/status"));
      result.innerHTML = "";
      for (const [k, v] of Object.entries(data.counts).sort((a, b) => b[1] - a[1])) {
        const d = document.createElement("div");
        d.className = "count";
        d.innerHTML = '<div class="num">' + esc(v) + '</div><div class="lbl">' + esc(k) + '</div>';
        result.appendChild(d);
      }
    } catch (e) {
      status.textContent = "error: " + e.message;
    } finally {
      btn.disabled = false;
    }
  });
}

// ---- Learners ----
function drawCurve(points, drifts) {
  const root = document.getElementById("learn-chart");
  if (!root) return;
  root.innerHTML = "";
  if (!points || !points.length) { root.textContent = "no data"; return; }
  const W = 720, H = 240, pad = 40;
  const xs = points.map(p => p.i);
  const xMin = xs[0], xMax = xs[xs.length - 1];
  const yMin = 0, yMax = 1;
  const sx = i => pad + (i - xMin) / Math.max(1, xMax - xMin) * (W - pad * 1.4);
  const sy = v => H - pad - (v - yMin) / (yMax - yMin) * (H - pad * 1.6);
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("width", "100%");
  // grid
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    const y = sy(g);
    const l = document.createElementNS(svgNS, "line");
    l.setAttribute("x1", pad); l.setAttribute("x2", W - pad * 0.4);
    l.setAttribute("y1", y); l.setAttribute("y2", y);
    l.setAttribute("stroke", "#eceff1");
    svg.appendChild(l);
    const tx = document.createElementNS(svgNS, "text");
    tx.setAttribute("x", 6); tx.setAttribute("y", y + 4);
    tx.setAttribute("font-size", "10"); tx.setAttribute("fill", "#90a4ae");
    tx.textContent = g.toFixed(2);
    svg.appendChild(tx);
  }
  // drift markers
  if (drifts && drifts.length) {
    for (const d of drifts) {
      const x = sx(d);
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", x); line.setAttribute("x2", x);
      line.setAttribute("y1", pad); line.setAttribute("y2", H - pad);
      line.setAttribute("stroke", "#c62828");
      line.setAttribute("stroke-dasharray", "4 3");
      line.setAttribute("stroke-width", "1");
      svg.appendChild(line);
      const lbl = document.createElementNS(svgNS, "text");
      lbl.setAttribute("x", x + 3); lbl.setAttribute("y", pad + 12);
      lbl.setAttribute("font-size", "10"); lbl.setAttribute("fill", "#c62828");
      lbl.textContent = "drift";
      svg.appendChild(lbl);
    }
  }
  // cum line
  const cum = points.map(p => sx(p.i) + "," + sy(p.cum)).join(" ");
  const pCum = document.createElementNS(svgNS, "polyline");
  pCum.setAttribute("points", cum);
  pCum.setAttribute("fill", "none");
  pCum.setAttribute("stroke", "#2c6fa6");
  pCum.setAttribute("stroke-width", "2");
  svg.appendChild(pCum);
  // win line
  const win = points.map(p => sx(p.i) + "," + sy(p.win)).join(" ");
  const pWin = document.createElementNS(svgNS, "polyline");
  pWin.setAttribute("points", win);
  pWin.setAttribute("fill", "none");
  pWin.setAttribute("stroke", "#e67e22");
  pWin.setAttribute("stroke-width", "1.5");
  svg.appendChild(pWin);
  // legend
  const leg1 = document.createElementNS(svgNS, "text");
  leg1.setAttribute("x", W - pad * 3.5); leg1.setAttribute("y", 16);
  leg1.setAttribute("font-size", "11"); leg1.setAttribute("fill", "#2c6fa6");
  leg1.textContent = "cumulative";
  svg.appendChild(leg1);
  const leg2 = document.createElementNS(svgNS, "text");
  leg2.setAttribute("x", W - pad * 3.5); leg2.setAttribute("y", 30);
  leg2.setAttribute("font-size", "11"); leg2.setAttribute("fill", "#e67e22");
  leg2.textContent = "rolling";
  svg.appendChild(leg2);
  root.appendChild(svg);
}

const learnForm = document.getElementById("learn-form");
if (learnForm) {
  learnForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = document.getElementById("learn-btn");
    const status = document.getElementById("learn-status");
    const metrics = document.getElementById("learn-metrics");
    btn.disabled = true;
    status.textContent = "training…";
    metrics.innerHTML = "";
    document.getElementById("learn-chart").innerHTML = "";
    try {
      const body = {
        learner: document.getElementById("learner").value,
        stream: document.getElementById("stream").value,
        n: parseInt(document.getElementById("n").value, 10) || 1000,
        seed: parseInt(document.getElementById("seed").value, 10) || 0,
        window: parseInt(document.getElementById("window").value, 10) || 100,
      };
      const r = await fetch("/api/learners/run", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!data.ok) throw new Error(data.error || "run failed");
      status.textContent = data.learner + " on " + data.stream + " — " +
                           data.n + " samples in " + data.elapsed_s + " s";
      const show = [
        ["accuracy final", data.accuracy_final],
        ["accuracy last " + data.window, data.accuracy_window],
        ["p50 µs", data.p50_us],
        ["p99 µs", data.p99_us],
      ];
      for (const [k, v] of show) {
        const d = document.createElement("div");
        d.className = "count";
        d.innerHTML = '<div class="num">' + esc(v) + '</div><div class="lbl">' + esc(k) + '</div>';
        metrics.appendChild(d);
      }
      drawCurve(data.points, data.drift_positions);
      if (data.slices && data.slices.length) {
        const slRoot = document.createElement("div");
        slRoot.className = "counts";
        slRoot.style.marginTop = "1em";
        for (const s of data.slices) {
          const d = document.createElement("div");
          d.className = "count";
          d.innerHTML = '<div class="num">' + esc(s.acc) + '</div><div class="lbl">' +
                        esc(s.from) + "–" + esc(s.to) + '</div>';
          slRoot.appendChild(d);
        }
        metrics.appendChild(slRoot);
      }
    } catch (e) {
      status.textContent = "error: " + e.message;
    } finally {
      btn.disabled = false;
    }
  });
}

// ---- Playground ----

// ---- Playground (top bar + chain) ----

// ================= Playground =================
function drawPlaygroundChart(chart) {
  const root = document.getElementById("pg-chart");
  if (!root) return;
  root.innerHTML = "";
  if (!chart || chart.kind !== "line" || !chart.points || !chart.points.length) return;
  const svgNS = "http://www.w3.org/2000/svg";
  const W = 640, H = 260, pad = 30;
  const xs = chart.points.map(p => p.i);
  const xMin = xs[0], xMax = xs[xs.length - 1];
  const sx = i => pad + (i - xMin) / Math.max(1, xMax - xMin) * (W - pad * 1.4);
  const sy = v => H - pad - v * (H - pad * 1.6);
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("width", "100%");
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    const y = sy(g);
    const l = document.createElementNS(svgNS, "line");
    l.setAttribute("x1", pad); l.setAttribute("x2", W - pad * 0.4);
    l.setAttribute("y1", y); l.setAttribute("y2", y);
    l.setAttribute("stroke", "#eceff1"); svg.appendChild(l);
    const tx = document.createElementNS(svgNS, "text");
    tx.setAttribute("x", 4); tx.setAttribute("y", y + 4);
    tx.setAttribute("font-size", "10"); tx.setAttribute("fill", "#90a4ae");
    tx.textContent = g.toFixed(2); svg.appendChild(tx);
  }
  if (chart.drifts) for (const d of chart.drifts) {
    const x = sx(d);
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", x); line.setAttribute("x2", x);
    line.setAttribute("y1", pad); line.setAttribute("y2", H - pad);
    line.setAttribute("stroke", "#c62828");
    line.setAttribute("stroke-dasharray", "4 3");
    svg.appendChild(line);
  }
  const pts = chart.points.map(p => sx(p.i) + "," + sy(p.cum)).join(" ");
  const poly = document.createElementNS(svgNS, "polyline");
  poly.setAttribute("points", pts);
  poly.setAttribute("fill", "none");
  poly.setAttribute("stroke", "#2c6fa6");
  poly.setAttribute("stroke-width", "2");
  svg.appendChild(poly);
  root.appendChild(svg);
}

["p-learner", "p-stream", "chk-anomaly", "chk-mixture", "chk-garch"].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", function(){});
});

const pgBtn = document.getElementById("pg-run");
if (pgBtn) {
  pgBtn.addEventListener("click", async () => {
    const code = document.getElementById("pg-code").value;
    const status = document.getElementById("pg-status");
    const out = document.getElementById("pg-stdout");
    const err = document.getElementById("pg-stderr");
    const params = {
      stream: document.getElementById("p-stream").value,
      learner: document.getElementById("p-learner").value,
      n: parseInt(document.getElementById("p-n").value, 10) || 1000,
      seed: parseInt(document.getElementById("p-seed").value, 10) || 0,
      use_anomaly: document.getElementById("chk-anomaly").checked,
      use_mixture: document.getElementById("chk-mixture").checked,
      use_garch: document.getElementById("chk-garch").checked,
    };
    pgBtn.disabled = true;
    status.textContent = "running " + params.learner + " on " + params.stream + "…";
    out.textContent = ""; err.textContent = "";
    document.getElementById("pg-chart").innerHTML = "";
    try {
      const r = await fetch("/api/playground/run", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify({code, params}),
      });
      const data = await r.json();
      status.textContent = "exit " + data.returncode + " in " + data.elapsed_s + " s";
      out.textContent = data.stdout || "(no stdout)";
      err.textContent = data.stderr || "";
      if (data.chart) drawPlaygroundChart(data.chart);
    } catch (e) {
      status.textContent = "error: " + e.message;
    } finally {
      pgBtn.disabled = false;
    }
  });
}

// init


// The chain page uses its own inline script in chain.html.


async function loadSideStatus() {
  const root = document.getElementById("status-side");
  if (!root) return;
  try {
    const r = await fetch("/status");
    const s = await r.json();
    for (const dd of root.querySelectorAll("dd[data-k]")) {
      const k = dd.getAttribute("data-k");
      const v = k.split(".").reduce((o, kk) => (o == null ? undefined : o[kk]), s);
      if (typeof v === "boolean") {
        dd.textContent = v ? "true" : "false";
        dd.className = v ? "good" : "bad";
      } else {
        dd.textContent = v == null ? "—" : String(v);
        dd.className = "";
      }
    }
  } catch (e) { /* silent */ }
}

if (document.getElementById("status-side")) {
  if (document.readyState !== "loading") loadSideStatus();
  else document.addEventListener("DOMContentLoaded", loadSideStatus);
}


// ============================================================
// Command palette (Cmd+K / Ctrl+K)
// ============================================================
(function initCommandPalette() {
  function setup() {
  const overlay = document.getElementById("cmd-palette");
  const input = document.getElementById("cmd-input");
  const results = document.getElementById("cmd-results");
  if (!overlay || !input || !results) return;

  let allPieces = [];
  let filtered = [];
  let activeIdx = 0;
  let lastQuery = "";

  async function load() {
    if (allPieces.length) return;
    try {
      const r = await fetch("/api/pieces/search");
      allPieces = await r.json();
    } catch (e) {}
  }

  function render() {
    results.innerHTML = "";
    if (!filtered.length) {
      results.innerHTML = '<div class="cmd-empty">no results</div>';
      return;
    }
    filtered.forEach((p, i) => {
      const row = document.createElement("a");
      row.className = "cmd-row" + (i === activeIdx ? " active" : "");
      row.href = `/piece/${p.section}/${p.category}/${p.id}`;
      row.innerHTML =
        `<span class="cmd-section">${esc(p.section)}</span>` +
        `<span class="cmd-cat">${esc(p.category)}</span>` +
        `<span class="cmd-name">${esc(p.name)}</span>`;
      row.addEventListener("mouseenter", () => { activeIdx = i; render(); });
      results.appendChild(row);
    });
    // scroll nell'attivo
    const active = results.querySelector(".cmd-row.active");
    if (active) active.scrollIntoView({block: "nearest"});
  }

  function filter() {
    const q = input.value.toLowerCase().trim();
    lastQuery = q;
    filtered = !q ? allPieces.slice(0, 20) : allPieces.filter(p => {
      const hay = `${p.section} ${p.category} ${p.id} ${p.name}`.toLowerCase();
      return hay.includes(q);
    });
    activeIdx = 0;
    render();
  }

  function open() {
    overlay.style.display = "flex";
    input.value = "";
    load().then(filter);
    setTimeout(() => input.focus(), 20);
  }

  function close() {
    overlay.style.display = "none";
  }

  function go() {
    const active = results.querySelector(".cmd-row.active");
    if (active) {
      close();
      window.location.href = active.href;
    }
  }

  input.addEventListener("input", filter);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { close(); }
    else if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIdx = Math.min(activeIdx + 1, filtered.length - 1);
      render();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIdx = Math.max(activeIdx - 1, 0);
      render();
    } else if (e.key === "Enter") {
      e.preventDefault();
      go();
    }
  });

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      if (overlay.style.display === "flex") close();
      else open();
    }
    if (e.key === "Escape" && overlay.style.display === "flex") {
      close();
    }
  });
  }  // end setup()

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();

(function () {
  function ping() { fetch('/ping', { cache: 'no-store' }).catch(function () {}); }
  ping();
  setInterval(ping, 5000);
})();
