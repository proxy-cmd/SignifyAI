async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} => ${res.status}`);
  }
  return res.json();
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function refresh() {
  try {
    const [health, metrics, intents] = await Promise.all([
      fetchJson('/health'),
      fetchJson('/metrics'),
      fetchJson('/intents'),
    ]);
    document.getElementById('health').textContent = pretty(health);
    document.getElementById('metrics').textContent = pretty(metrics);
    document.getElementById('intents').textContent = pretty(intents.intents);
  } catch (err) {
    document.getElementById('health').textContent = String(err);
  }
}

refresh();
setInterval(refresh, 4000);
