import { MODE_OPTIONS, latencyState } from "../utils/constants.js";

export function createStatusStrip({ mode }) {
  const state = {
    mode,
    handDetected: false,
    handCount: 0,
    faceDetected: false,
    latencyMs: 0
  };

  const element = document.createElement("article");
  element.className = "panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">Status Strip</p>
        <h3>Frame Health</h3>
      </div>
      <span class="meta-tag">event.frame_status</span>
    </div>
    <div class="status-strip">
      <div class="status-item">
        <span class="field-label">Mode</span>
        <span class="status-value" data-mode>Realtime Translation</span>
      </div>
      <div class="status-item">
        <span class="field-label">Hand Detected</span>
        <span class="status-value" data-hand>No</span>
      </div>
      <div class="status-item">
        <span class="field-label">Hand Count</span>
        <span class="status-value" data-count>0</span>
      </div>
      <div class="status-item">
        <span class="field-label">Face Detected</span>
        <span class="status-value" data-face>No</span>
      </div>
      <div class="status-item">
        <span class="field-label">Latency</span>
        <span class="status-value" data-latency>0 ms</span>
      </div>
    </div>
  `;

  const modeEl = element.querySelector("[data-mode]");
  const handEl = element.querySelector("[data-hand]");
  const countEl = element.querySelector("[data-count]");
  const faceEl = element.querySelector("[data-face]");
  const latencyEl = element.querySelector("[data-latency]");

  function update(nextState) {
    Object.assign(state, nextState);
    modeEl.textContent = MODE_OPTIONS.find((item) => item.value === state.mode)?.label || state.mode;
    handEl.textContent = state.handDetected ? "Yes" : "No";
    countEl.textContent = String(state.handCount);
    faceEl.textContent = state.faceDetected ? "Yes" : "No";
    latencyEl.textContent = `${Number(state.latencyMs || 0).toFixed(0)} ms`;
    latencyEl.className = `status-value ${latencyState(state.latencyMs)}`;
  }

  update(state);

  return {
    element,
    update
  };
}
