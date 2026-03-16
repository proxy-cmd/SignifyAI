export function createTeachPanel({ mode, onTeach }) {
  const element = document.createElement("article");
  element.className = "panel teach-panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">Teach Sign Panel</p>
        <h3>Teach Current Gesture</h3>
        <p class="panel-copy">Available only in Realtime Translation mode to capture new sign labels.</p>
      </div>
      <span class="meta-tag">command.teach_sign</span>
    </div>

    <form class="teach-form">
      <label class="visually-hidden" for="gestureLabelInput">Gesture label</label>
      <input
        class="field-input"
        id="gestureLabelInput"
        name="gestureLabel"
        type="text"
        placeholder="Enter gesture label"
        autocomplete="off"
      />
      <button class="btn btn-primary" type="submit">Teach Current Gesture</button>
      <p class="teach-feedback" data-teach-feedback>Waiting for training request.</p>
    </form>
  `;

  const form = element.querySelector(".teach-form");
  const input = element.querySelector("#gestureLabelInput");
  const feedback = element.querySelector("[data-teach-feedback]");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const label = input.value.trim();
    if (!label) {
      showResult({
        success: false,
        message: "Enter a gesture label before sending the teach request."
      });
      return;
    }

    onTeach(label);
    showResult({
      success: true,
      message: `Teach request sent for "${label}". Awaiting backend confirmation.`
    });
    input.value = "";
  });

  function setMode(nextMode) {
    mode = nextMode;
    element.classList.toggle("hidden", mode !== "default");
  }

  function showResult(result) {
    feedback.textContent = result.message;
    feedback.className = `teach-feedback ${result.success ? "success" : "error"}`;
  }

  setMode(mode);

  return {
    element,
    setMode,
    showResult
  };
}
