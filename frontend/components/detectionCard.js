export function createDetectionCard() {
  const state = {
    detectedSign: "Awaiting recognition",
    confidence: 0,
    source: "vision_pipeline",
    speech: "Speech output will appear here."
  };

  const element = document.createElement("article");
  element.className = "panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">Detection Result</p>
        <h3>Prediction Output</h3>
        <p class="panel-copy">Finalized sign predictions, confidence score, and synthesized speech text.</p>
      </div>
      <span class="meta-tag">event.prediction_final</span>
    </div>

    <div class="detection-display">
      <div class="detection-hero">
        <p class="card-label">Sign Detected</p>
        <h4 data-sign-value>Awaiting recognition</h4>
      </div>

      <div class="detection-meta">
        <div class="field">
          <span class="field-label">Confidence</span>
          <span class="field-value" data-confidence-text>0%</span>
        </div>
        <div class="field">
          <span class="field-label">Detection Source</span>
          <span class="field-value" data-source-text>vision_pipeline</span>
        </div>
      </div>

      <div class="confidence-meter">
        <div class="meter-topline">
          <span class="field-label">Confidence Meter</span>
          <strong data-meter-label>0%</strong>
        </div>
        <div class="meter-track">
          <div class="meter-bar" data-meter-bar></div>
        </div>
      </div>

      <div class="speech-quote" data-speech-text>"Speech output will appear here."</div>
    </div>
  `;

  const signValue = element.querySelector("[data-sign-value]");
  const confidenceText = element.querySelector("[data-confidence-text]");
  const sourceText = element.querySelector("[data-source-text]");
  const meterLabel = element.querySelector("[data-meter-label]");
  const meterBar = element.querySelector("[data-meter-bar]");
  const speechText = element.querySelector("[data-speech-text]");

  function update(nextState) {
    Object.assign(state, nextState);
    signValue.textContent = String(state.detectedSign).toUpperCase();
    confidenceText.textContent = `${state.confidence}%`;
    sourceText.textContent = state.source;
    meterLabel.textContent = `${state.confidence}%`;
    meterBar.style.width = `${state.confidence}%`;
    speechText.textContent = `"${state.speech}"`;
  }

  update(state);

  return {
    element,
    update
  };
}
