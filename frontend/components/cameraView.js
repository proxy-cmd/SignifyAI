import { MODE_OPTIONS } from "../utils/constants.js";

export function createCameraView({ mode }) {
  const state = {
    mode,
    sessionActive: false,
    detectedSign: "Waiting for session",
    subtitle: "Start a session to begin live monitoring",
    recognitionState: "Camera standby",
    handDetected: false,
    faceDetected: false,
    latencyMs: 0
  };

  const element = document.createElement("article");
  element.className = "panel camera-panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">Camera View</p>
        <h3>Live Recognition Feed</h3>
        <p class="panel-copy">Real-time video stage with hand, face, and alignment guidance overlays.</p>
      </div>
      <div class="camera-toolbar">
        <span class="meta-tag" data-camera-mode></span>
        <span class="meta-tag" data-camera-session>Standby</span>
      </div>
    </div>

    <div class="camera-stage">
      <video class="camera-video" autoplay muted playsinline></video>
      <div class="camera-placeholder">
        <div class="camera-placeholder-card">
          <p class="guide-label">Clinical Recognition Feed</p>
          <h4 class="hero-detected" data-detected-sign>Waiting for session</h4>
          <p class="guide-note" data-subtitle>Start a session to begin live monitoring</p>
          <p class="guide-note" data-recognition-state>Camera standby</p>
        </div>
      </div>

      <div class="camera-guides">
        <div class="camera-guide guide-face" data-face-guide></div>
        <div class="camera-guide guide-hand" data-hand-guide></div>
        <div class="guide-scan" aria-hidden="true"></div>
        <div class="camera-guide-tip tip-face">Face zone</div>
        <div class="camera-guide-tip tip-hand">Hand zone</div>
        <div class="camera-guide-tip tip-align">Center gesture and keep palm visible</div>
      </div>
    </div>

    <div class="camera-footer">
      <div class="kpi-card">
        <p class="kpi-label">Recognition State</p>
        <span class="kpi-value" data-state-value>Camera standby</span>
      </div>
      <div class="kpi-card">
        <p class="kpi-label">Latency</p>
        <span class="kpi-value" data-latency-value>0 ms</span>
      </div>
      <div class="kpi-card">
        <p class="kpi-label">Alignment Tip</p>
        <span class="kpi-value" data-tip-value>Raise hand into guide area</span>
      </div>
    </div>
  `;

  const modeTag = element.querySelector("[data-camera-mode]");
  const sessionTag = element.querySelector("[data-camera-session]");
  const detectedSign = element.querySelector("[data-detected-sign]");
  const subtitle = element.querySelector("[data-subtitle]");
  const recognitionState = element.querySelector("[data-recognition-state]");
  const stateValue = element.querySelector("[data-state-value]");
  const latencyValue = element.querySelector("[data-latency-value]");
  const tipValue = element.querySelector("[data-tip-value]");
  const handGuide = element.querySelector("[data-hand-guide]");
  const faceGuide = element.querySelector("[data-face-guide]");
  const placeholder = element.querySelector(".camera-placeholder");
  const video = element.querySelector(".camera-video");

  function update(nextState) {
    Object.assign(state, nextState);

    const modeLabel = MODE_OPTIONS.find((item) => item.value === state.mode)?.label || state.mode;

    modeTag.textContent = modeLabel;
    sessionTag.textContent = state.sessionActive ? "Session live" : "Standby";
    detectedSign.textContent = state.detectedSign;
    subtitle.textContent = state.subtitle;
    recognitionState.textContent = state.recognitionState;
    stateValue.textContent = state.recognitionState;
    latencyValue.textContent = `${Number(state.latencyMs || 0).toFixed(0)} ms`;
    tipValue.textContent = state.handDetected
      ? "Maintain steady gesture for final prediction"
      : "Raise hand into guide area";
    handGuide.classList.toggle("active", state.handDetected);
    faceGuide.classList.toggle("active", state.faceDetected);
  }

  function attachStream(stream) {
    video.srcObject = stream;
    placeholder.classList.add("hidden");
  }

  function clearStream() {
    video.srcObject = null;
    placeholder.classList.remove("hidden");
  }

  update(state);

  return {
    element,
    update,
    attachStream,
    clearStream
  };
}
