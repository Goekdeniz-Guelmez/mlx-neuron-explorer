import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { t } from "./i18n.js?v=5";

// live view: the model reads a text and writes a reply while every layer's MLP is watched, one token at a time
// (/api/live). Each layer's sheet arrives pooled into grid x grid cells (the strongest signed score in each cell,
// as a multiple of the neuron's typical range). Three views of the same frames:
//  brain:    the layers as coronal slices front to back (layer 0 at the back), each slice two hemispheres whose cells
//            run from the cortex inwards, glowing amber (up) or magenta (down)
//  maps:     every layer's grid side by side for the current token
//  timeline: tokens left to right, layers bottom to top, each layer's activity; scrub, play, step
const UP = [1.0, 0.71, 0.28], DOWN = [1.0, 0.31, 0.6];
const SPEEDS = [0.5, 1, 2, 4, 8];

export function createLive(root, arch, { onOpen }) {
  const $ = s => root.querySelector(s);
  root.innerHTML = `
    <form class="lv-bar">
      <input class="lv-text" spellcheck="false" autocomplete="off">
      <span class="seg lv-mode"><button type="button" data-m="write" class="on"></button><button type="button" data-m="read"></button></span>
      <label class="lv-chat dim"><input type="checkbox" checked> <span></span></label>
      <input class="lv-max mono" value="120" inputmode="numeric">
      <button type="submit" class="lv-go"></button>
      <span class="lv-state mono dim"></span>
    </form>
    <div class="lv-stage"><canvas class="lv-brain"></canvas><div class="lv-tip float"></div><div class="lv-legend mono dim"></div></div>
    <aside class="lv-side float">
      <div class="lv-head mono"><span class="lv-cur"></span></div>
      <canvas class="lv-maps"></canvas>
      <div class="lv-subhead dim"></div>
      <ol class="lv-top rows"></ol>
    </aside>
    <div class="lv-time float">
      <div class="lv-transport">
        <button type="button" class="lv-back" aria-label="step back">⏮</button>
        <button type="button" class="lv-play" aria-label="play">▶</button>
        <button type="button" class="lv-fwd" aria-label="step forward">⏭</button>
        <button type="button" class="lv-speed mono">1×</button>
        <button type="button" class="lv-follow mono on"></button>
        <span class="lv-pos mono dim"></span>
      </div>
      <canvas class="lv-heat"></canvas>
      <div class="lv-toks"></div>
    </div>`;
  const L = arch.n_layers;
  let G = 24, frames = [], heat = [], cur = -1, playing = false, follow = true, speed = 1, nPrompt = 0, ctrl = null, focusLayer = -1;
  const text = $(".lv-text"), chat = $(".lv-chat input"), maxEl = $(".lv-max");
  let mode = "write";

  function label() {
    text.placeholder = t("live.placeholder");
    $(".lv-mode [data-m=write]").textContent = t("live.write");
    $(".lv-mode [data-m=read]").textContent = t("live.read");
    $(".lv-chat span").textContent = t("live.chat");
    maxEl.title = t("live.maxTitle");
    $(".lv-go").textContent = ctrl ? t("live.stop") : t("live.run");
    $(".lv-follow").textContent = t("live.follow");
    $(".lv-subhead").textContent = t("live.strongest");
    $(".lv-legend").innerHTML = `<span style="color:var(--up)">●</span> ${t("live.up")} &nbsp; <span style="color:var(--down)">●</span> ${t("live.down")} &nbsp; ${t("live.legend")}`;
    if (!frames.length) $(".lv-state").textContent = t("live.empty");
  }
  root.querySelectorAll(".lv-mode button").forEach(b => b.onclick = () => {
    mode = b.dataset.m; root.querySelectorAll(".lv-mode button").forEach(x => x.classList.toggle("on", x === b));
  });

  // ---------- brain ----------
  const canvas = $(".lv-brain");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.05, 100);
  camera.position.set(4.4, 1.9, 2.2);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 0.35;
  canvas.addEventListener("pointerdown", () => { controls.autoRotate = false; });
  let geo = null, vals = null, shown = null, layerOf = null, cellOf = null, points = null;

  function slicePos(l, r, c) {
    // front-back position and size of this layer's slice: rounder at the back, a fuller frontal lobe
    const f = L > 1 ? l / (L - 1) : 0.5, z = (f - 0.5) * 3.4;
    const prof = Math.sqrt(Math.max(0.05, 1 - (z / 1.85) ** 2)) * (1 + 0.1 * Math.max(0, f - 0.55));
    const half = G / 2, hemi = c < half ? -1 : 1;
    const u = (c % half) / Math.max(1, half - 1);  // 0 = cortex, 1 = deep
    const th = (r / G) * Math.PI * 2 + 0.3;
    const gyri = 1 + 0.06 * Math.sin(7 * th + l * 0.9) + 0.03 * Math.sin(13 * th - l * 1.7);
    const R = (1 - 0.62 * u) * gyri;
    let x = 0.5 + 0.52 * R * Math.cos(th), y = 0.82 * R * Math.sin(th);
    if (x < 0.04) x = 0.04 + 0.02 * u;  // the flat medial wall between the hemispheres
    // a flatter base, except the temporal lobes, which hang lower in the middle third
    const temporal = Math.exp(-(((f - 0.48) / 0.16) ** 2)) * Math.max(0, Math.cos(th + 0.5 * Math.PI) * 1.3 - 0.2);
    if (y < 0) y *= 0.45 + 0.75 * temporal;
    // the dome: tallest just behind the middle
    const dome = 0.85 + 0.25 * Math.exp(-(((f - 0.42) / 0.3) ** 2));
    return [hemi * x * prof * 1.08, y * prof * dome + 0.1, z];
  }

  function buildBrain() {
    if (points) { scene.remove(points); geo.dispose(); }
    const n = L * G * G, pos = new Float32Array(n * 3);
    vals = new Float32Array(n); shown = new Float32Array(n); layerOf = new Int16Array(n); cellOf = new Int16Array(n);
    for (let l = 0, i = 0; l < L; l++) for (let r = 0; r < G; r++) for (let c = 0; c < G; c++, i++) {
      const p = slicePos(l, r, c);
      pos.set([p[0] + (Math.random() - 0.5) * 0.02, p[1] + (Math.random() - 0.5) * 0.02, p[2] + (Math.random() - 0.5) * 0.04], i * 3);
      layerOf[i] = l; cellOf[i] = r * G + c;
    }
    geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aVal", new THREE.BufferAttribute(shown, 1));
    const fl = new Float32Array(n); geo.setAttribute("aFocus", new THREE.BufferAttribute(fl, 1));
    points = new THREE.Points(geo, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { px: { value: 2.6 * Math.min(devicePixelRatio, 2) }, up: { value: new THREE.Vector3(...UP) }, down: { value: new THREE.Vector3(...DOWN) } },
      vertexShader: `uniform float px; attribute float aVal; attribute float aFocus; varying float vV; varying float vF;
        void main() { vV = aVal; vF = aFocus; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
          gl_PointSize = px * (1.0 + 2.6 * min(abs(aVal), 2.0) + 0.6 * aFocus) * (4.0 / -mv.z); }`,
      fragmentShader: `uniform vec3 up, down; varying float vV; varying float vF;
        void main() { vec2 d = gl_PointCoord - 0.5; float r = dot(d, d) * 4.0; if (r > 1.0) discard;
          float a = clamp(abs(vV), 0.0, 2.0) * 0.5; vec3 hot = vV > 0.0 ? up : down;
          vec3 c = mix(vec3(0.16, 0.19, 0.21) + 0.25 * vF, hot, smoothstep(0.03, 0.6, a)) * (0.45 + 1.5 * a);
          gl_FragColor = vec4(c * (1.0 - r) * (1.0 - r), 1.0); }`,
    }));
    scene.add(points);
  }

  function setFocusLayer(l) {
    focusLayer = l;
    const fa = geo.getAttribute("aFocus");
    for (let i = 0; i < fa.count; i++) fa.array[i] = layerOf[i] === l ? 1 : 0;
    fa.needsUpdate = true; drawMaps();
  }

  // hover: the cell under the pointer
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.04;
  const tip = $(".lv-tip");
  canvas.addEventListener("pointermove", e => {
    if (!points || cur < 0) { tip.style.display = "none"; return; }
    const b = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1), camera);
    const hit = ray.intersectObject(points).sort((a, b) => Math.abs(shown[b.index]) - Math.abs(shown[a.index]))[0];
    if (!hit || Math.abs(shown[hit.index]) < 0.15) { tip.style.display = "none"; return; }
    const i = hit.index, v = frames[cur].cells[i] / 64;
    tip.innerHTML = `<b class="mono">L${layerOf[i]}</b> <span class="mono dim">${t("live.cell")} ${cellOf[i]}</span><br><span class="mono" style="color:var(${v > 0 ? "--up" : "--down"})">${v > 0 ? "+" : ""}${v.toFixed(2)}×</span> <span class="dim">${t("live.typical")}</span>`;
    tip.style.display = "block"; tip.style.left = `${e.clientX - b.left + 14}px`; tip.style.top = `${e.clientY - b.top + 10}px`;
  });
  canvas.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  canvas.addEventListener("click", () => {
    if (tip.style.display !== "block") return;
    const m = tip.querySelector("b").textContent.slice(1); setFocusLayer(Number(m) === focusLayer ? -1 : Number(m));
  });

  // ---------- frames ----------
  function decode(b64) {
    const s = atob(b64), a = new Int8Array(s.length);
    for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i) << 24 >> 24;
    return a;
  }
  function layerHeat(cells) {  // each layer's activity: the mean of its strongest tenth of cells
    const out = new Float32Array(L), k = Math.max(1, Math.round(G * G / 10)), buf = new Float32Array(G * G);
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < G * G; j++) buf[j] = Math.abs(cells[l * G * G + j]) / 64;
      buf.sort(); let s = 0; for (let j = G * G - k; j < G * G; j++) s += buf[j]; out[l] = s / k;
    }
    return out;
  }

  async function run(e) {
    e?.preventDefault();
    if (ctrl) { ctrl.abort(); return; }
    const body = { text: text.value.trim() || t("live.placeholder"), chat: chat.checked, max_new: Number(maxEl.value) || 120, read_only: mode === "read" };
    frames = []; heat = []; cur = -1; follow = true; $(".lv-follow").classList.add("on"); $(".lv-toks").innerHTML = "";
    ctrl = new AbortController(); label();
    $(".lv-state").textContent = t("live.starting");
    try {
      const r = await fetch("/api/live", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      const rd = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let k;
        while ((k = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, k); buf = buf.slice(k + 2);
          if (!line.startsWith("data:")) continue;
          const ev = JSON.parse(line.slice(5));
          if (ev.meta) { G = ev.meta.grid; nPrompt = ev.meta.prompt; buildBrain(); resize(); continue; }
          if (ev.done) { if (ev.error) $(".lv-state").textContent = ev.error; continue; }
          addFrame({ ...ev, cells: decode(ev.cells) });
        }
      }
      $(".lv-state").textContent = t("live.done", { n: frames.length, p: nPrompt });
    } catch (err) {
      $(".lv-state").textContent = err.name === "AbortError" ? t("live.stopped", { n: frames.length }) : String(err);
    } finally { ctrl = null; label(); }
  }
  $(".lv-bar").onsubmit = run;

  function addFrame(f) {
    frames.push(f); heat.push(layerHeat(f.cells));
    const s = document.createElement("span");
    s.textContent = f.tok.replace(/\n/g, "⏎");
    s.className = f.prompt ? "p" : "g";
    if (/^<\|.*\|>$/.test(f.tok) || /^<\/?think>$/.test(f.tok)) s.classList.add("sp");
    s.onclick = () => { stopFollow(); go(f.i); };
    $(".lv-toks").append(s);
    if (follow) go(frames.length - 1);
    else drawHeat();
    $(".lv-state").textContent = f.prompt ? t("live.reading", { i: f.i + 1, p: nPrompt }) : t("live.writing", { n: frames.length - nPrompt });
  }

  function go(i) {
    if (!frames.length) return;
    cur = Math.max(0, Math.min(frames.length - 1, i));
    const f = frames[cur];
    for (let j = 0; j < vals.length; j++) vals[j] = f.cells[j] / 64;
    root.querySelectorAll(".lv-toks span").forEach((s, j) => s.classList.toggle("on", j === cur));
    root.querySelector(".lv-toks span.on")?.scrollIntoView({ block: "nearest", inline: "nearest" });
    $(".lv-cur").innerHTML = `<span class="dim">${f.prompt ? t("live.reads") : t("live.writes")}</span> <b>${esc(JSON.stringify(f.tok).slice(1, -1))}</b>`;
    $(".lv-pos").textContent = `${cur + 1} / ${frames.length}`;
    $(".lv-top").innerHTML = f.top.map(c => `<li data-l="${c.layer}" data-n="${c.n}"><span class="n">L${c.layer}N${c.n}</span>
      <span class="z" style="color:var(${c.score > 0 ? "--up" : "--down"})">${c.score > 0 ? "+" : ""}${c.score.toFixed(1)}</span>
      <span class="w">${(c.lens || []).map(x => `<b>${esc(x)}</b>`).join(" ")}</span></li>`).join("");
    root.querySelectorAll(".lv-top li").forEach(li => {
      li.onclick = () => onOpen(Number(li.dataset.l), Number(li.dataset.n));
      li.onmouseenter = () => setFocusLayer(Number(li.dataset.l));
    });
    drawMaps(); drawHeat();
  }
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------- maps: every layer's grid for the current token ----------
  const maps = $(".lv-maps"), mctx = maps.getContext("2d");
  let mapCols = 8;
  function color(v) {
    const a = Math.min(1, Math.abs(v) / 1.5), c = v > 0 ? UP : DOWN, b = 0.06 * (1 - a);
    return [255 * (b + c[0] * a), 255 * (b * 1.2 + c[1] * a), 255 * (b * 1.4 + c[2] * a)];
  }
  function drawMaps() {
    const w = maps.clientWidth, dpr = Math.min(devicePixelRatio, 2);
    mapCols = L > 24 ? 8 : 6;
    const rows = Math.ceil(L / mapCols), cell = Math.floor((w - (mapCols - 1) * 4) / mapCols);
    maps.width = w * dpr; maps.height = (rows * (cell + 14)) * dpr; maps.style.height = `${rows * (cell + 14)}px`;
    mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    mctx.clearRect(0, 0, w, rows * (cell + 14));
    if (cur < 0) return;
    const f = frames[cur], img = mctx.createImageData(G, G), tmp = document.createElement("canvas");
    tmp.width = tmp.height = G; const tctx = tmp.getContext("2d");
    mctx.imageSmoothingEnabled = false; mctx.font = "10px ui-monospace, Menlo, monospace";
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < G * G; j++) {
        const [r, g, b] = color(f.cells[l * G * G + j] / 64);
        img.data.set([r, g, b, 255], j * 4);
      }
      tctx.putImageData(img, 0, 0);
      const x = (l % mapCols) * (cell + 4), y = Math.floor(l / mapCols) * (cell + 14);
      mctx.drawImage(tmp, x, y + 12, cell, cell);
      mctx.fillStyle = l === focusLayer ? "#fff" : "#666"; mctx.fillText(`L${l}`, x, y + 9);
      if (l === focusLayer) { mctx.strokeStyle = "#fff"; mctx.strokeRect(x + 0.5, y + 12.5, cell - 1, cell - 1); }
    }
    maps._geom = { cell, rows };
  }
  maps.onclick = e => {
    const g = maps._geom; if (!g) return;
    const b = maps.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
    const l = Math.floor(y / (g.cell + 14)) * mapCols + Math.floor(x / (g.cell + 4));
    if (l >= 0 && l < L) setFocusLayer(l === focusLayer ? -1 : l);
  };

  // ---------- timeline ----------
  const hc = $(".lv-heat"), hctx = hc.getContext("2d");
  function drawHeat() {
    const w = hc.clientWidth, h = hc.clientHeight, dpr = Math.min(devicePixelRatio, 2);
    if (hc.width !== w * dpr || hc.height !== h * dpr) { hc.width = w * dpr; hc.height = h * dpr; }
    hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    hctx.fillStyle = "#000"; hctx.fillRect(0, 0, w, h);
    const T = frames.length; if (!T) return;
    const cw = w / T, rh = h / L, peak = new Float32Array(L).fill(1e-6), low = new Float32Array(L).fill(1e9);
    for (let i = 0; i < T; i++) for (let l = 0; l < L; l++) { peak[l] = Math.max(peak[l], heat[i][l]); low[l] = Math.min(low[l], heat[i][l]); }
    for (let i = 0; i < T; i++) for (let l = 0; l < L; l++) {  // each layer against its own range, so its changes show
      const v = Math.max(0, (heat[i][l] - low[l]) / Math.max(1e-6, peak[l] - low[l])) ** 1.3, g = frames[i].prompt ? 0.75 : 1;
      hctx.fillStyle = `rgb(${255 * v * g | 0},${(150 * v * v + 18 * v) * g | 0},${(60 * v * v + 30 * v) * g | 0})`;
      hctx.fillRect(i * cw, h - (l + 1) * rh, Math.ceil(cw), Math.ceil(rh));
    }
    if (nPrompt && nPrompt < T) { hctx.fillStyle = "#38c8ff"; hctx.fillRect(nPrompt * cw - 0.5, 0, 1, h); }
    if (focusLayer >= 0) { hctx.strokeStyle = "rgba(255,255,255,.5)"; hctx.strokeRect(0, h - (focusLayer + 1) * rh, w, rh); }
    if (cur >= 0) { hctx.fillStyle = "#fff"; hctx.fillRect(cur * cw + cw / 2 - 1, 0, 2, h); }
  }
  let dragging = false;
  const scrub = e => {
    const b = hc.getBoundingClientRect();
    stopFollow(); go(Math.floor(((e.clientX - b.left) / b.width) * frames.length));
  };
  hc.addEventListener("pointerdown", e => { dragging = true; hc.setPointerCapture(e.pointerId); pause(); scrub(e); });
  hc.addEventListener("pointermove", e => { if (dragging) scrub(e); });
  hc.addEventListener("pointerup", () => { dragging = false; });

  function stopFollow() { follow = false; $(".lv-follow").classList.remove("on"); }
  let last = 0, acc = 0;
  function play() {
    if (!frames.length) return;
    if (cur >= frames.length - 1) go(0);
    playing = true; stopFollow(); $(".lv-play").textContent = "⏸";
  }
  function pause() { playing = false; $(".lv-play").textContent = "▶"; }
  $(".lv-play").onclick = () => (playing ? pause() : play());
  $(".lv-back").onclick = () => { pause(); stopFollow(); go(cur - 1); };
  $(".lv-fwd").onclick = () => { pause(); stopFollow(); go(cur + 1); };
  $(".lv-speed").onclick = () => {
    speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]; $(".lv-speed").textContent = `${speed}×`;
  };
  $(".lv-follow").onclick = () => { follow = !follow; $(".lv-follow").classList.toggle("on", follow); if (follow) { pause(); go(frames.length - 1); } };
  addEventListener("keydown", e => {
    if (!root.classList.contains("on") || /INPUT|TEXTAREA/.test(document.activeElement?.tagName)) return;
    if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === "ArrowLeft") { pause(); stopFollow(); go(cur - 1); }
    else if (e.key === "ArrowRight") { pause(); stopFollow(); go(cur + 1); }
  });

  // ---------- frame loop ----------
  function resize() {
    const s = $(".lv-stage"), w = s.clientWidth, h = s.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    drawMaps(); drawHeat();
  }
  new ResizeObserver(resize).observe($(".lv-stage"));
  function tick(now) {
    requestAnimationFrame(tick);
    if (!root.classList.contains("on")) { last = now; return; }
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) {
      acc += dt * 6 * speed;  // 6 tokens a second at 1x
      while (acc >= 1) { acc -= 1; if (cur >= frames.length - 1) { pause(); break; } go(cur + 1); }
    }
    if (shown) {  // ease toward the current token's values, so a step reads as a pulse rather than a flicker
      const k = 1 - Math.exp(-dt * 14);
      for (let j = 0; j < shown.length; j++) shown[j] += (vals[j] - shown[j]) * k;
      geo.getAttribute("aVal").needsUpdate = true;
    }
    controls.update(); renderer.render(scene, camera);
  }
  buildBrain();
  requestAnimationFrame(tick);
  label();
  return { label, resize };
}
