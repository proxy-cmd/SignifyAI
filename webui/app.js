const presetGrid = document.getElementById("preset-grid");
const terminal = document.getElementById("terminal");
const runIndicator = document.getElementById("run-indicator");
const runText = document.getElementById("run-text");

let lastSeq = 0;
let autoScroll = true;

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    const msg = data.error || data.message || `Request failed: ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function appendLog(lines) {
  if (!Array.isArray(lines) || lines.length === 0) return;
  const chunk = lines.map((x) => `[${x.time}] ${x.line}`).join("\n");
  terminal.textContent += (terminal.textContent ? "\n" : "") + chunk;
  if (autoScroll) {
    terminal.scrollTop = terminal.scrollHeight;
  }
}

function setStatus(status, metrics) {
  const running = !!status.running;
  runIndicator.classList.toggle("live", running);
  runText.textContent = running
    ? `Running • ${status.command?.slice(2).join(" ") || "command"}`
    : `Idle${status.exit_code !== null && status.exit_code !== undefined ? ` • last exit: ${status.exit_code}` : ""}`;

  document.getElementById("metric-dataset").textContent = metrics.dataset_exists
    ? `Ready (${metrics.dataset_size_mb} MB)`
    : "Missing";
  document.getElementById("metric-frame").textContent = metrics.frame_model ? "Available" : "Missing";
  document.getElementById("metric-deep").textContent = metrics.deep_model ? "Available" : "Missing";
  document.getElementById("metric-temporal").textContent = metrics.temporal_model ? "Available" : "Missing";
}

function commandToast(message, isError = false) {
  const el = document.createElement("div");
  el.textContent = message;
  el.style.position = "fixed";
  el.style.right = "22px";
  el.style.bottom = "22px";
  el.style.padding = "10px 14px";
  el.style.borderRadius = "10px";
  el.style.border = `1px solid ${isError ? "rgba(255,110,110,.6)" : "rgba(22,214,176,.6)"}`;
  el.style.background = isError ? "rgba(120,25,25,.9)" : "rgba(9,71,58,.9)";
  el.style.boxShadow = "0 10px 30px rgba(0,0,0,.35)";
  el.style.zIndex = "99";
  el.style.fontWeight = "600";
  document.body.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity 300ms ease";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 300);
  }, 1800);
}

async function runPreset(id) {
  const data = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({ mode: "preset", id }),
  });
  commandToast(data.message || "Started");
}

async function runMainArgs(args) {
  const data = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({ mode: "main_args", args }),
  });
  commandToast(data.message || "Started");
}

function renderPresets(catalog) {
  presetGrid.innerHTML = "";
  for (const p of catalog.presets || []) {
    const card = document.createElement("article");
    card.className = "preset-card";
    card.innerHTML = `
      <h4>${p.title}</h4>
      <p>${p.description}</p>
      <button class="btn">Run</button>
    `;
    card.querySelector("button").addEventListener("click", async () => {
      try {
        await runPreset(p.id);
      } catch (err) {
        commandToast(err.message, true);
      }
    });
    presetGrid.appendChild(card);
  }
}

function bindControls() {
  document.getElementById("btn-stop").addEventListener("click", async () => {
    try {
      const data = await api("/api/stop", { method: "POST", body: "{}" });
      commandToast(data.message || "Stopped");
    } catch (err) {
      commandToast(err.message, true);
    }
  });

  document.getElementById("btn-collect").addEventListener("click", async () => {
    const label = document.getElementById("collect-label").value.trim() || "hello";
    const samples = document.getElementById("collect-samples").value.trim() || "250";
    const args = `collect --label ${JSON.stringify(label)} --samples ${samples}`;
    try {
      await runMainArgs(args);
    } catch (err) {
      commandToast(err.message, true);
    }
  });

  document.getElementById("btn-run-live").addEventListener("click", async () => {
    const profile = document.getElementById("run-profile").value;
    const mode = document.getElementById("run-mode").value;
    const args = `run --profile ${profile} --mode ${mode}`;
    try {
      await runMainArgs(args);
    } catch (err) {
      commandToast(err.message, true);
    }
  });

  document.getElementById("btn-advanced").addEventListener("click", async () => {
    const args = document.getElementById("advanced-args").value.trim();
    if (!args) {
      commandToast("Enter a command first.", true);
      return;
    }
    try {
      await runMainArgs(args);
    } catch (err) {
      commandToast(err.message, true);
    }
  });

  document.getElementById("btn-clear-log").addEventListener("click", () => {
    terminal.textContent = "";
    lastSeq = 0;
  });

  terminal.addEventListener("scroll", () => {
    autoScroll = terminal.scrollTop + terminal.clientHeight >= terminal.scrollHeight - 6;
  });
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    setStatus(data.status, data.metrics || {});
  } catch (err) {
    commandToast(err.message, true);
  }
}

async function refreshLogs() {
  try {
    const data = await api(`/api/logs?since=${lastSeq}`);
    const logs = data.logs || [];
    if (logs.length > 0) {
      lastSeq = Number(logs[logs.length - 1].seq || lastSeq);
      appendLog(logs);
    }
  } catch (err) {
    commandToast(err.message, true);
  }
}

async function init() {
  bindControls();
  try {
    const catalog = await api("/api/catalog");
    renderPresets(catalog);
  } catch (err) {
    commandToast(err.message, true);
  }
  await refreshStatus();
  await refreshLogs();
  setInterval(refreshStatus, 1000);
  setInterval(refreshLogs, 900);
}

// Subtle pointer tilt effect for panels.
function initTilt() {
  const cards = document.querySelectorAll("[data-tilt]");
  cards.forEach((card) => {
    card.addEventListener("mousemove", (e) => {
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width;
      const y = (e.clientY - r.top) / r.height;
      const rx = (0.5 - y) * 5;
      const ry = (x - 0.5) * 6;
      card.style.transform = `perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg)`;
    });
    card.addEventListener("mouseleave", () => {
      card.style.transform = "perspective(1000px) rotateX(0) rotateY(0)";
    });
  });
}

// Lightweight animated particles for atmosphere.
function initFxCanvas() {
  const canvas = document.getElementById("fx-canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let w = 0;
  let h = 0;
  const particles = [];
  const count = 60;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  for (let i = 0; i < count; i += 1) {
    particles.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 2.2 + 0.8,
      vx: (Math.random() - 0.5) * 0.26,
      vy: (Math.random() - 0.5) * 0.26,
      c: i % 3 === 0 ? "rgba(31,181,255,.65)" : i % 3 === 1 ? "rgba(22,214,176,.58)" : "rgba(255,200,87,.45)",
    });
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.c;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  draw();
}

init();
initTilt();
initFxCanvas();
