const REFRESH_MS = 4000;

const healthEl = document.getElementById('health');
const metricsEl = document.getElementById('metrics');
const intentsEl = document.getElementById('intents');

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} -> ${response.status}`);
  }
  return response.json();
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function showData(health, metrics, intents) {
  healthEl.textContent = pretty(health);
  metricsEl.textContent = pretty(metrics);
  intentsEl.textContent = pretty(intents.intents);
}

function showError(err) {
  healthEl.textContent = String(err);
}

async function refreshPage() {
  try {
    const [health, metrics, intents] = await Promise.all([
      getJson('/health'),
      getJson('/metrics'),
      getJson('/intents'),
    ]);
    showData(health, metrics, intents);
  } catch (err) {
    showError(err);
  }
}

refreshPage();
setInterval(refreshPage, REFRESH_MS);
