// HerdSight farmer dashboard: farm map, camp page and alerts (Phase 6, D-17).
// Plain JavaScript, no build step. Every sentence shown comes from the survey records, which take
// their wording from dashboard/glossary.yaml. The last answers are kept on the phone, and actions
// taken without signal are queued and sent later.
"use strict";

const view = document.getElementById("view");
const ACTION_TEXT = {
  second_check: "Ask the drone to check again",
  found_them: "I found them",
  records_updated: "My records were wrong, I have updated them",
  dismissed: "Hide this alert",
};
const STORE = "herdsight-cache-v1";
const QUEUE = "herdsight-queue-v1";

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function readStore(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function writeStore(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full or blocked */ }
}

async function get(path) {
  const cache = readStore(STORE, {});
  try {
    const r = await fetch(path, { headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error(String(r.status));
    const data = await r.json();
    cache[path] = data;
    writeStore(STORE, cache);
    document.getElementById("offline").hidden = true;
    return data;
  } catch (e) {
    if (path in cache) { document.getElementById("offline").hidden = false; return cache[path]; }
    throw e;
  }
}

async function send(alertId, action) {
  const body = JSON.stringify({ action });
  try {
    const r = await fetch(`/api/alerts/${encodeURIComponent(alertId)}/action`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body,
    });
    if (!r.ok) throw new Error(String(r.status));
    return true;
  } catch {
    const q = readStore(QUEUE, []);
    q.push({ alertId, action });
    writeStore(QUEUE, q);
    return false;
  }
}

async function flushQueue() {
  const q = readStore(QUEUE, []);
  if (!q.length) return;
  const left = [];
  for (const item of q) if (!(await send(item.alertId, item.action))) left.push(item);
  writeStore(QUEUE, left);
}

function when(iso) {
  return new Date(iso).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
}

async function farmId() {
  const farms = await get("/api/farms");
  if (!farms.length) throw new Error("no farm");
  return farms[0];
}

function mapSvg(camps) {
  const pts = camps.flatMap((c) => c.outline);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = 20, w = x1 - x0 + 2 * pad, h = y1 - y0 + 2 * pad;
  const P = (p) => `${p[0] - x0 + pad},${y1 - p[1] + pad}`; // north up
  const shapes = camps.map((c) => {
    const cx = c.outline.reduce((s, p) => s + p[0], 0) / c.outline.length;
    const cy = c.outline.reduce((s, p) => s + p[1], 0) / c.outline.length;
    return `<a href="#/camp/${encodeURIComponent(c.camp_id)}" aria-label="${esc(c.name)}: ${esc(c.label)}">
      <polygon class="${esc(c.colour)}" points="${c.outline.map(P).join(" ")}"></polygon>
      <text x="${cx - x0 + pad}" y="${y1 - cy + pad}" text-anchor="middle">${esc(c.name)}</text></a>`;
  });
  return `<svg class="map" viewBox="0 0 ${w} ${h}" role="img" aria-label="Farm map">${shapes.join("")}</svg>`;
}

async function showFarm() {
  const farm = await get(`/api/farms/${encodeURIComponent(await farmId())}`);
  document.getElementById("sim").hidden = !farm.simulated;
  const n = farm.needs_attention.length;
  const red = farm.camps.some((c) => c.colour === "red");
  const summary = n === 0
    ? `<div class="summary ok">All camps look fine</div>`
    : `<div class="summary ${red ? "bad" : "warn"}">${n} camp${n > 1 ? "s" : ""} need${n > 1 ? "" : "s"} attention</div>`;
  const order = [...farm.needs_attention, ...farm.camps.filter((c) => c.status === "ok").map((c) => c.camp_id)];
  const byId = Object.fromEntries(farm.camps.map((c) => [c.camp_id, c]));
  const items = order.map((id) => byId[id]).map((c) => `<li><a href="#/camp/${encodeURIComponent(c.camp_id)}">
      <span class="dot ${esc(c.colour)}"></span><span class="name">${esc(c.name)}</span>
      <span class="label">${esc(c.label)}${c.latest ? `<br>${when(c.latest.surveyed_at)}` : ""}</span></a></li>`);
  view.innerHTML = `<h1>${esc(farm.name)}</h1>${summary}${mapSvg(farm.camps)}<ul class="list">${items.join("")}</ul>`;
  badge(farm.open_alerts);
}

function chart(history) {
  const pts = [...history].reverse();
  if (pts.length < 2) return "";
  const lv = (r) => r.intervals.find((i) => i.level === 0.9) ?? r.intervals[r.intervals.length - 1];
  const vals = pts.flatMap((r) => [lv(r).low, lv(r).high, r.expected_count ?? r.count]);
  const lo = Math.min(...vals) - 2, hi = Math.max(...vals) + 2, W = 600, H = 120;
  const X = (i) => 10 + (i * (W - 20)) / (pts.length - 1), Y = (v) => H - 10 - ((v - lo) * (H - 20)) / (hi - lo);
  const band = pts.map((r, i) => `${X(i)},${Y(lv(r).high)}`).concat(pts.map((r, i) => `${X(i)},${Y(lv(r).low)}`).reverse());
  const line = pts.map((r, i) => `${X(i)},${Y(r.count)}`).join(" ");
  const rec = pts[pts.length - 1].expected_count;
  const recLine = rec == null ? "" : `<line class="rec" x1="10" x2="${W - 10}" y1="${Y(rec)}" y2="${Y(rec)}"></line>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"
    aria-label="Counts over time with their likely ranges"><polygon class="band" points="${band.join(" ")}"></polygon>
    ${recLine}<polyline class="line" points="${line}"></polyline></svg>
    <p class="muted">Shaded: likely range of each check. Dashed: the number in your records.</p>`;
}

function actionButtons(alert) {
  const done = (alert.actions || []).map((a) => a.action);
  return `<div class="actions">${Object.entries(ACTION_TEXT).map(([k, t]) =>
    `<button data-alert="${esc(alert.alert_id)}" data-action="${k}" ${k === "second_check" ? 'class="primary"' : ""}
      ${done.includes(k) ? "disabled" : ""}>${esc(t)}${done.includes(k) ? " (done)" : ""}</button>`).join("")}</div>`;
}

async function showCamp(campId) {
  const id = await farmId();
  const [farm, camp] = await Promise.all([get(`/api/farms/${encodeURIComponent(id)}`), get(`/api/farms/${encodeURIComponent(id)}/camps/${encodeURIComponent(campId)}`)]);
  document.getElementById("sim").hidden = !farm.simulated;
  const meta = farm.camps.find((c) => c.camp_id === campId) || { name: campId, label: "", colour: "grey" };
  const r = camp.history[0];
  const range = r.intervals.find((i) => i.level === 0.9) ?? r.intervals[0];
  const alerts = await get(`/api/farms/${encodeURIComponent(id)}/alerts`);
  const open = alerts.filter((a) => a.camp_id === campId);
  view.innerHTML = `<p><a href="#/">&larr; Farm</a></p><h1><span class="dot ${esc(meta.colour)}" style="display:inline-block"></span> ${esc(meta.name)}</h1>
    <div class="card"><div class="muted">${esc(meta.label)} · checked ${when(r.surveyed_at)}</div>
      <div class="big">${r.count}</div><div>likely between ${range.low} and ${range.high}${r.expected_count != null ? ` · your records: ${r.expected_count}` : ""}</div>
      ${r.notes.map((n) => `<p class="note">${esc(n)}</p>`).join("")}</div>
    ${open.map((a) => `<div class="card"><strong>${esc(a.message)}</strong>${actionButtons(a)}</div>`).join("")}
    <h2>Recent checks</h2><div class="card">${chart(camp.history)}
      <ul class="list">${camp.history.slice(0, 8).map((h) => `<li><a><span class="name">${when(h.surveyed_at)}</span><span class="label">${h.count} counted</span></a></li>`).join("")}</ul></div>`;
}

async function showAlerts() {
  const id = await farmId();
  const [alerts, farm] = await Promise.all([get(`/api/farms/${encodeURIComponent(id)}/alerts`), get(`/api/farms/${encodeURIComponent(id)}`)]);
  document.getElementById("sim").hidden = !farm.simulated;
  badge(alerts.length);
  view.innerHTML = `<h1>Alerts</h1>` + (alerts.length
    ? alerts.map((a) => `<div class="card"><div class="muted">${esc(a.camp_name)} · ${when(a.surveyed_at)}</div>
        <p><strong>${esc(a.message)}</strong></p><a href="#/camp/${encodeURIComponent(a.camp_id)}">See this camp</a>${actionButtons(a)}</div>`).join("")
    : `<div class="summary ok">No open alerts</div>`);
}

function badge(n) {
  const b = document.getElementById("alert-count");
  b.hidden = !n;
  b.textContent = n;
}

async function route() {
  const hash = location.hash || "#/";
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on",
    (a.dataset.nav === "alerts") === hash.startsWith("#/alerts")));
  try {
    if (hash.startsWith("#/camp/")) await showCamp(decodeURIComponent(hash.slice(7)));
    else if (hash.startsWith("#/alerts")) await showAlerts();
    else await showFarm();
  } catch {
    view.innerHTML = `<p>Could not load the farm. Check the signal and try again.</p>`;
  }
}

view.addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-action]");
  if (!b) return;
  b.disabled = true;
  const sent = await send(b.dataset.alert, b.dataset.action);
  b.insertAdjacentHTML("afterend", `<p class="done">${sent ? "Done." : "Saved. It will be sent when there is signal."}</p>`);
  if (sent) setTimeout(route, 800);
});
window.addEventListener("hashchange", route);
window.addEventListener("online", () => { flushQueue().then(route); });
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
flushQueue().finally(route);
