export function createEmergencyCard({ onAcknowledge }) {
  const state = {
    intent: "No active emergency",
    priority: "Monitor",
    confidence: 0,
    speech: "Emergency alerts will appear here immediately when detected.",
    requiresAck: false,
    mode: "default"
  };

  const element = document.createElement("article");
  element.className = "panel emergency-panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">Emergency Alert</p>
        <h3>Priority Response Panel</h3>
      </div>
      <span class="meta-tag" data-priority-tag>Monitor</span>
    </div>
    <p class="emergency-copy" data-speech-copy>Emergency alerts will appear here immediately when detected.</p>
    <div class="emergency-grid">
      <div class="emergency-box">
        <span class="field-label">Intent</span>
        <span class="field-value" data-intent>No active emergency</span>
      </div>
      <div class="emergency-box">
        <span class="field-label">Priority</span>
        <span class="field-value" data-priority>Monitor</span>
      </div>
      <div class="emergency-box">
        <span class="field-label">Confidence</span>
        <span class="field-value" data-confidence>0%</span>
      </div>
    </div>
    <div style="margin-top: 18px;">
      <button class="btn btn-danger" type="button" data-acknowledge hidden>ACKNOWLEDGE</button>
    </div>
  `;

  const priorityTag = element.querySelector("[data-priority-tag]");
  const speechCopy = element.querySelector("[data-speech-copy]");
  const intentEl = element.querySelector("[data-intent]");
  const priorityEl = element.querySelector("[data-priority]");
  const confidenceEl = element.querySelector("[data-confidence]");
  const acknowledgeButton = element.querySelector("[data-acknowledge]");

  acknowledgeButton.addEventListener("click", () => {
    onAcknowledge();
  });

  function update(nextState) {
    Object.assign(state, nextState);
    const alerting =
      state.requiresAck || /high|critical|emergency|urgent/i.test(String(state.priority));

    priorityTag.textContent = state.priority;
    speechCopy.textContent = state.speech;
    intentEl.textContent = state.intent;
    priorityEl.textContent = state.priority;
    confidenceEl.textContent = `${state.confidence}%`;
    acknowledgeButton.hidden = !state.requiresAck;
    element.classList.toggle("alerting", alerting);
  }

  function acknowledge() {
    update({
      intent: "Alert acknowledged",
      priority: "Acknowledged",
      confidence: state.confidence,
      speech: "Operator acknowledged the active emergency alert.",
      requiresAck: false
    });
  }

  function setMode(mode) {
    state.mode = mode;
  }

  update(state);

  return {
    element,
    update,
    acknowledge,
    setMode
  };
}
