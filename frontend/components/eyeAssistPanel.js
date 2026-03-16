export function createEyeAssistPanel({ mode }) {
  const state = {
    mode,
    leftRatio: "--",
    rightRatio: "--",
    gazeX: "--",
    gazeY: "--",
    mappedIntent: "No mapped intent"
  };

  const element = document.createElement("article");
  element.className = "panel eye-panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">Eye Assist Panel</p>
        <h3>Gaze and Blink Mapping</h3>
        <p class="panel-copy">Eye guidance becomes active in Eye Assist Mode for alternative patient communication.</p>
      </div>
      <span class="meta-tag">event.eye_status</span>
    </div>
    <div class="eye-grid">
      <div class="eye-box">
        <span class="field-label">Left Eye Ratio</span>
        <span class="field-value" data-left-ratio>--</span>
      </div>
      <div class="eye-box">
        <span class="field-label">Right Eye Ratio</span>
        <span class="field-value" data-right-ratio>--</span>
      </div>
      <div class="eye-box">
        <span class="field-label">Gaze X</span>
        <span class="field-value" data-gaze-x>--</span>
      </div>
      <div class="eye-box">
        <span class="field-label">Gaze Y</span>
        <span class="field-value" data-gaze-y>--</span>
      </div>
      <div class="eye-box">
        <span class="field-label">Mapped Intent</span>
        <span class="field-value" data-mapped-intent>No mapped intent</span>
      </div>
    </div>
  `;

  const leftRatio = element.querySelector("[data-left-ratio]");
  const rightRatio = element.querySelector("[data-right-ratio]");
  const gazeX = element.querySelector("[data-gaze-x]");
  const gazeY = element.querySelector("[data-gaze-y]");
  const mappedIntent = element.querySelector("[data-mapped-intent]");

  function syncModeClass() {
    const isEyeMode = state.mode === "eye";
    element.classList.toggle("idle", !isEyeMode);
  }

  function update(nextState) {
    Object.assign(state, nextState);
    leftRatio.textContent = state.leftRatio;
    rightRatio.textContent = state.rightRatio;
    gazeX.textContent = state.gazeX;
    gazeY.textContent = state.gazeY;
    mappedIntent.textContent = state.mappedIntent;
    syncModeClass();
  }

  function setMode(nextMode) {
    state.mode = nextMode;
    syncModeClass();
  }

  update(state);

  return {
    element,
    update,
    setMode
  };
}
