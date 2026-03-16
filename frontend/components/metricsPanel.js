export function createMetricsPanel() {
  const state = {
    e2e: "--",
    capture: "--",
    perception: "--",
    decode: "--",
    speech: "--"
  };

  const element = document.createElement("article");
  element.className = "panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">System Metrics</p>
        <h3>Performance Pipeline</h3>
        <p class="panel-copy">Latency breakdown across capture, perception, decode, and speech synthesis stages.</p>
      </div>
      <span class="meta-tag">event.health_metrics</span>
    </div>
    <div class="metrics-grid">
      <div class="metric">
        <span class="field-label">E2E Latency</span>
        <span class="metric-value" data-e2e>--</span>
      </div>
      <div class="metric">
        <span class="field-label">Capture Time</span>
        <span class="metric-value" data-capture>--</span>
      </div>
      <div class="metric">
        <span class="field-label">Perception Time</span>
        <span class="metric-value" data-perception>--</span>
      </div>
      <div class="metric">
        <span class="field-label">Decode Time</span>
        <span class="metric-value" data-decode>--</span>
      </div>
      <div class="metric">
        <span class="field-label">Speech Time</span>
        <span class="metric-value" data-speech>--</span>
      </div>
    </div>
  `;

  const fields = {
    e2e: element.querySelector("[data-e2e]"),
    capture: element.querySelector("[data-capture]"),
    perception: element.querySelector("[data-perception]"),
    decode: element.querySelector("[data-decode]"),
    speech: element.querySelector("[data-speech]")
  };

  function update(nextState) {
    Object.assign(state, nextState);
    Object.entries(fields).forEach(([key, node]) => {
      node.textContent = state[key];
    });
  }

  update(state);

  return {
    element,
    update
  };
}
