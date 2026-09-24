/* Facility Monitor dashboard: WebSocket metrics, controls, canvas timelines, event log. */
(() => {
  const $ = (id) => document.getElementById(id);
  const fmt = (n, d = 1) => (typeof n === "number" ? n.toFixed(d) : "0");
  const SETTING_KEYS = ["min_people", "max_people", "grace_seconds", "lights_required", "lights_on_threshold",
    "lights_off_threshold", "light_gain", "model", "conf", "imgsz", "process_every_n"];
  const NUMERIC = new Set(["min_people", "max_people", "grace_seconds", "lights_on_threshold", "lights_off_threshold", "light_gain", "conf", "imgsz", "process_every_n"]);
  // light_gain is stored as a 0..2 multiplier but shown as a 0..200 % slider
  const toUI = (k, v) => (k === "light_gain" ? Math.round(v * 100) : v);
  const fromUI = (k, v) => (k === "light_gain" ? v / 100 : v);
  const clock = (sec) => { const s = Math.max(0, sec || 0); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`; };

  let history = [];
  let settings = {};
  let running = false;
  let starting = false;
  let ws;

  // ------------------------------------------------------------ theme (night is the default; choice is remembered per browser)
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    for (const b of $("theme-seg").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.theme === t));
    try { localStorage.setItem("fm-theme", t); } catch (_) { /* private mode */ }
    if (typeof drawCharts === "function") drawCharts();
  }
  $("theme-seg").addEventListener("click", (e) => { const b = e.target.closest("button[data-theme]"); if (b) applyTheme(b.dataset.theme); });
  let savedTheme = "dark";
  try { savedTheme = localStorage.getItem("fm-theme") || "dark"; } catch (_) { /* ignore */ }
  document.documentElement.dataset.theme = savedTheme;
  for (const b of $("theme-seg").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.theme === savedTheme));

  // ------------------------------------------------------------ notices
  function notify(text) { $("notice-text").textContent = text; $("notice").hidden = false; }
  $("notice-close").addEventListener("click", () => { $("notice").hidden = true; });

  // ------------------------------------------------------------ sources
  async function loadSources(selectId) {
    const r = await fetch("/api/sources");
    const { sources } = await r.json();
    const sel = $("source");
    sel.innerHTML = "";
    const groups = { sample: "Samples", upload: "Uploads", webcam: "Live" };
    for (const kind of ["sample", "upload", "webcam"]) {
      const items = sources.filter((s) => s.kind === kind);
      if (!items.length) continue;
      const og = document.createElement("optgroup");
      og.label = groups[kind];
      for (const s of items) {
        const o = document.createElement("option");
        o.value = s.id;
        o.textContent = s.size_mb ? `${s.name} (${s.size_mb} MB)` : s.name;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    if (selectId) sel.value = selectId;
    else { const demo = sources.find((s) => s.name.startsWith("demo_")); if (demo) sel.value = demo.id; }
  }

  $("file-input").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    $("start").disabled = true;
    $("start").textContent = "Uploading";
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const { id } = await r.json();
      await loadSources(id);
    } catch (err) { notify("Upload failed: " + err.message); }
    finally { $("start").disabled = running; $("start").textContent = "Start"; e.target.value = ""; }
  });

  // ------------------------------------------------------------ start / stop
  $("start").addEventListener("click", async () => {
    $("start").disabled = true;
    starting = true;
    $("notice").hidden = true;
    $("video-wrap").dataset.state = "loading";
    $("loading-text").textContent = "Opening source";
    try {
      const r = await fetch("/api/start", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: $("source").value, loop: $("loop").checked, max_speed: $("max-speed").checked }) });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      $("events").innerHTML = '<p class="empty">No events yet. Alerts and their resolutions appear here with a snapshot.</p>';
      $("ev-count").textContent = "0";
      history = [];
      $("video").src = "/stream.mjpg?" + Date.now();
    } catch (err) {
      notify("Could not start: " + err.message);
      $("start").disabled = false;
      $("video-wrap").dataset.state = "idle";
    } finally { starting = false; }
  });
  $("stop").addEventListener("click", async () => { $("stop").disabled = true; await fetch("/api/stop", { method: "POST" }); });

  // ------------------------------------------------------------ settings
  function renderSettings(s) {
    settings = s;
    for (const k of SETTING_KEYS) {
      const el = $("s-" + k);
      if (!el || document.activeElement === el) continue;
      if (el.type === "checkbox") el.checked = !!s[k]; else el.value = toUI(k, s[k]);
    }
    $("o-on").value = s.lights_on_threshold; $("o-off").value = s.lights_off_threshold;
    $("mark-on").style.left = s.lights_on_threshold + "%"; $("mark-off").style.left = s.lights_off_threshold + "%";
    renderGain(toUI("light_gain", s.light_gain));
  }
  function renderGain(pct) {
    $("o-gain").value = pct;
    $("s-light_gain").style.setProperty("--p", pct / 2 + "%");
    for (const b of $("gain-presets").querySelectorAll("button")) b.classList.toggle("active", Number(b.dataset.gain) === pct);
    const sim = pct !== 100;
    $("k-sim").hidden = !sim;
    $("light-sim-label").hidden = !sim;
    $("light-sim-label").textContent = `simulated light ${pct}%`;
  }
  let saveTimer;
  const status = $("settings-status");
  function pushSettings(patch) {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      const r = await fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch) });
      if (r.ok) { renderSettings(await r.json()); status.textContent = "Saved. Applies immediately."; status.className = "save-state saved"; }
      else { status.textContent = "Value out of range, not saved."; status.className = "save-state error"; }
    }, 150);
  }
  for (const k of SETTING_KEYS) {
    const el = $("s-" + k);
    el.addEventListener("input", () => {
      const v = el.type === "checkbox" ? el.checked : (NUMERIC.has(k) ? Number(el.value) : el.value);
      if (k === "lights_on_threshold") $("o-on").value = v;
      if (k === "lights_off_threshold") $("o-off").value = v;
      if (k === "light_gain") renderGain(v);
      pushSettings({ [k]: fromUI(k, v) });
    });
  }
  $("gain-presets").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-gain]");
    if (!b) return;
    const pct = Number(b.dataset.gain);
    $("s-light_gain").value = pct;
    renderGain(pct);
    pushSettings({ light_gain: pct / 100 });
  });

  // ------------------------------------------------------------ transport (seek)
  let seekDrag = false;      // user is dragging the slider
  let seekPending = null;    // {time, until}: keep showing the target until the pipeline catches up
  let duration = 0;
  const seek = $("seek");
  function setSlider(t) {
    const p = duration ? Math.min(Math.max(t / duration, 0), 1) : 0;
    seek.value = Math.round(p * 1000);
    seek.style.setProperty("--p", (p * 100).toFixed(2) + "%");
    $("t-now").textContent = clock(t);
  }
  async function seekTo(body) {
    try {
      const r = await fetch("/api/seek", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const { time } = await r.json();
      seekPending = { time, until: Date.now() + 2000 };
      setSlider(time);
      history = history.filter((p) => p.t <= time);  // the chart continues from the new position
    } catch (err) { notify("Could not seek: " + err.message); }
  }
  seek.addEventListener("input", () => { seekDrag = true; $("t-now").textContent = clock((seek.value / 1000) * duration); seek.style.setProperty("--p", seek.value / 10 + "%"); });
  seek.addEventListener("change", () => { seekDrag = false; seekTo({ time: (seek.value / 1000) * duration }); });
  $("seek-back").addEventListener("click", () => seekTo({ delta: -10 }));
  $("seek-fwd").addEventListener("click", () => seekTo({ delta: 10 }));
  document.addEventListener("keydown", (e) => {
    // typing fields and sliders own their arrow keys; a focused button (e.g. a preset just clicked) does not
    if (e.target.matches("input, select, textarea") || $("transport").dataset.enabled !== "true") return;
    if (e.key === "ArrowLeft") { e.preventDefault(); seekTo({ delta: e.shiftKey ? -30 : -10 }); }
    if (e.key === "ArrowRight") { e.preventDefault(); seekTo({ delta: e.shiftKey ? 30 : 10 }); }
  });

  // ------------------------------------------------------------ metrics
  function renderMetrics(m) {
    running = m.running;
    $("start").disabled = running || starting;
    $("stop").disabled = !running;
    const wrap = $("video-wrap");
    if (!running) wrap.dataset.state = starting ? "loading" : "idle";
    else if (m.status === "STARTING" || m.note) { wrap.dataset.state = "loading"; $("loading-text").textContent = m.note || "Opening source"; }
    else wrap.dataset.state = "live";

    $("src-name").textContent = m.source || "No source";
    const modelEl = $("model-label"); modelEl.hidden = !(running && m.model); modelEl.textContent = m.model || "";
    const fpsEl = $("fps-label"); fpsEl.hidden = !running; fpsEl.textContent = `${fmt(m.fps)} fps`;
    $("dropped-label").textContent = running && m.skip_ratio ? `skipping ${Math.round(m.skip_ratio * 100)}% of frames to stay real time` : "";

    duration = m.duration || 0;
    const seekable = running && !m.webcam && m.status !== "STARTING";
    $("transport").dataset.enabled = seekable ? "true" : "false";
    $("seek-back").disabled = $("seek-fwd").disabled = !seekable;
    seek.disabled = !(seekable && duration > 0);  // absolute scrubbing needs a known length
    $("seek-hint").hidden = !seekable;
    $("t-total").textContent = duration ? clock(duration) : (running && m.webcam ? "live" : "0:00.0");
    if (seekPending && (Date.now() > seekPending.until || Math.abs(m.video_time - seekPending.time) < 1.5)) seekPending = null;
    if (!seekDrag && !seekPending) setSlider(running ? m.video_time : 0);

    const dim = running ? "" : " dim";
    const people = $("k-people"); people.textContent = running ? m.people_now : "0"; people.className = "kpi-value mono" + dim;
    const ids = m.active_ids || [];
    $("k-ids").textContent = ids.length ? "IDs " + ids.slice(0, 10).map((i) => "#" + i).join(" ") + (ids.length > 10 ? ` +${ids.length - 10}` : "") : "No active tracks";
    const uniq = $("k-unique"); uniq.textContent = running ? m.unique_total : "0"; uniq.className = "kpi-value mono" + dim;
    const fps = $("k-fps"); fps.textContent = running ? fmt(m.fps, 0) : "0"; fps.className = "kpi-value mono" + dim;
    $("k-bright").textContent = running ? fmt(m.brightness, 0) : "0";
    $("bright-fill").style.width = (running ? m.brightness : 0) + "%";
    const lights = $("k-lights");
    lights.textContent = running ? (m.lights_on ? "On" : "Off") : "Unknown";
    lights.className = "kpi-value " + (running ? (m.lights_on ? "on" : "off") : "dim");

    const st = $("status");
    const alerts = m.active_alerts || [];
    if (!running) { st.dataset.status = "idle"; $("status-label").textContent = "Idle"; $("status-sub").textContent = "Waiting for a source"; }
    else if (m.status === "ALERT") { st.dataset.status = "alert"; $("status-label").textContent = "Alert"; $("status-sub").textContent = alerts.map((a) => a.label).join(", "); }
    else if (m.status === "STARTING") { st.dataset.status = "idle"; $("status-label").textContent = "Starting"; $("status-sub").textContent = m.source || ""; }
    else { st.dataset.status = "ok"; $("status-label").textContent = "All clear"; $("status-sub").textContent = "All rules satisfied"; }
    $("active-alerts").innerHTML = running ? alerts.map((a) =>
      `<li><span class="label">${a.label}</span><span class="detail">${a.detail}</span><span class="dur mono">${fmt(m.video_time - a.since, 0)} s</span></li>`).join("") : "";

    if (running && typeof m.video_time === "number") {
      const last = history[history.length - 1];
      if (!last || m.video_time - last.t >= 0.2 || m.video_time < last.t) {
        if (last && m.video_time < last.t) history = [];
        history.push({ t: m.video_time, people: m.people_now, brightness: m.brightness, alert: m.status === "ALERT" });
        if (history.length > 1500) history.shift();
      }
    }
    drawCharts();
  }

  // ------------------------------------------------------------ events
  function addEvent(ev) {
    const list = $("events");
    const empty = list.querySelector(".empty");
    if (empty) empty.remove();
    const el = document.createElement("article");
    el.className = "event";
    const badge = ev.event === "ALERT_START" ? '<span class="badge start">Alert</span>' :
      ev.event === "ALERT_END" ? '<span class="badge end">Resolved</span>' : '<span class="badge error">Error</span>';
    const wall = new Date(ev.wall_time * 1000).toLocaleTimeString();
    el.innerHTML = `${ev.snapshot ? `<img src="${ev.snapshot}" alt="Snapshot at ${fmt(ev.video_time)} s">` : '<div class="noimg"></div>'}
      <div><div class="event-row">${badge}<span class="label">${ev.label}</span><span class="when mono">${fmt(ev.video_time)} s at ${wall}</span></div>
      <p class="detail">${ev.detail}</p></div>`;
    list.prepend(el);
    $("ev-count").textContent = list.querySelectorAll(".event").length;
  }

  // ------------------------------------------------------------ charts (canvas, single series each, one accent)
  const charts = [
    { el: $("chart-people"), key: "people", max: (h) => { const m = Math.max(4, ...h.map((p) => p.people)) + 1; const st = Math.max(1, Math.ceil(m / 5)); return Math.ceil(m / st) * st; }, fmt: (v) => v + " people", step: 1 },
    { el: $("chart-bright"), key: "brightness", max: () => 100, fmt: (v) => fmt(v, 0) + "%", thresholds: true },
  ];
  const PAD = { l: 40, r: 12, t: 22, b: 20 };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function drawChart(c, hoverX) {
    const canvas = c.el.querySelector("canvas");
    const dpr = window.devicePixelRatio || 1;
    const W = c.el.clientWidth, H = c.el.clientHeight;
    if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
    const h = history;
    const tMax = h.length ? h[h.length - 1].t : 60;
    const tMin = Math.max(0, tMax - 300);
    const span = Math.max(tMax - tMin, 30);
    const yMax = c.max(h);
    const X = (t) => PAD.l + ((t - tMin) / span) * pw;
    const Y = (v) => PAD.t + ph - (v / yMax) * ph;

    ctx.fillStyle = css("--critical-soft");
    let bandStart = null;
    for (let i = 0; i < h.length; i++) {
      if (h[i].alert && bandStart === null) bandStart = h[i].t;
      if ((!h[i].alert || i === h.length - 1) && bandStart !== null) {
        ctx.fillRect(X(bandStart), PAD.t, Math.max(X(h[i].t) - X(bandStart), 2), ph); bandStart = null;
      }
    }
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1; ctx.fillStyle = css("--muted");
    ctx.font = "11px " + css("--mono"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const stepY = c.step ? Math.max(1, Math.ceil(yMax / 5)) : yMax / 4;
    for (let v = 0; v <= yMax + 1e-9; v += stepY) {
      const y = Y(v);
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
      ctx.fillText(Math.round(v), PAD.l - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let t = Math.ceil(tMin / 30) * 30; t <= tMin + span; t += 30) ctx.fillText(t + "s", X(t), PAD.t + ph + 4);
    if (c.thresholds && settings.lights_on_threshold != null) {
      ctx.setLineDash([4, 4]);
      for (const [v, col] of [[settings.lights_on_threshold, css("--good")], [settings.lights_off_threshold, css("--critical")]]) {
        ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(PAD.l, Y(v)); ctx.lineTo(W - PAD.r, Y(v)); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    if (h.length > 1) {
      ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
      h.forEach((p, i) => { const x = X(p.t), y = Y(p[c.key]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    const tip = c.el.querySelector(".tooltip");
    if (hoverX != null && h.length) {
      const t = tMin + ((hoverX - PAD.l) / pw) * span;
      let best = h[0];
      for (const p of h) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
      const x = X(best.t), y = Y(best[c.key]);
      ctx.strokeStyle = css("--muted"); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + ph); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = css("--accent"); ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2; ctx.stroke();
      tip.style.display = "block"; tip.textContent = `${fmt(best.t)} s  ${c.fmt(best[c.key])}${best.alert ? "  alert" : ""}`;
      tip.style.left = Math.min(x + 10, W - tip.offsetWidth - 4) + "px"; tip.style.top = Math.max(y - 30, 0) + "px";
    } else tip.style.display = "none";
  }
  const hover = new Map();
  function drawCharts() { for (const c of charts) drawChart(c, hover.get(c)); }
  for (const c of charts) {
    c.el.addEventListener("mousemove", (e) => { hover.set(c, e.offsetX); drawChart(c, e.offsetX); });
    c.el.addEventListener("mouseleave", () => { hover.delete(c); drawChart(c, null); });
  }
  window.addEventListener("resize", drawCharts);

  // ------------------------------------------------------------ websocket
  function connect() {
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.onopen = () => $("conn-dot").classList.add("on");
    ws.onclose = () => { $("conn-dot").classList.remove("on"); setTimeout(connect, 1500); };
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.kind === "init") {
        renderSettings(msg.settings);
        history = msg.history || [];
        for (let i = history.length - 1; i > 0; i--) if (history[i].t < history[i - 1].t) { history = history.slice(i); break; }
        $("events").innerHTML = '<p class="empty">No events yet. Alerts and their resolutions appear here with a snapshot.</p>';
        for (const ev of (msg.events || []).slice().reverse()) addEvent(ev);
        renderMetrics(msg.metrics);
      } else if (msg.kind === "metrics") { renderSettings(msg.settings); renderMetrics(msg.metrics); }
      else if (msg.kind === "event") addEvent(msg.event);
    };
  }

  loadSources().then(connect);
})();
