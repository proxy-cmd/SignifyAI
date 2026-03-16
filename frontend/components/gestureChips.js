import { humanizeGesture, MODE_OPTIONS } from "../utils/constants.js";

export function createGestureChips({ mode, gestures }) {
  const element = document.createElement("article");
  element.className = "panel";
  element.innerHTML = `
    <div class="panel-header">
      <div>
        <p class="section-label">Supported Gestures</p>
        <h3>Mode Vocabulary</h3>
        <p class="panel-copy" data-mode-description></p>
      </div>
      <span class="meta-tag" data-mode-badge></span>
    </div>
    <div class="gesture-panel">
      <div>
        <p class="chip-group-label">Available gestures</p>
        <div class="chip-group" data-chip-group></div>
      </div>
    </div>
  `;

  const modeDescription = element.querySelector("[data-mode-description]");
  const modeBadge = element.querySelector("[data-mode-badge]");
  const chipGroup = element.querySelector("[data-chip-group]");
  let activeGesture = "";

  function render(nextMode, nextGestures) {
    const modeInfo = MODE_OPTIONS.find((item) => item.value === nextMode);
    modeDescription.textContent = modeInfo?.description || "";
    modeBadge.textContent = modeInfo?.shortLabel || nextMode;
    chipGroup.innerHTML = "";

    nextGestures.forEach((gesture) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = humanizeGesture(gesture);
      chip.dataset.gesture = gesture.toLowerCase();
      chip.classList.toggle("emphasis", gesture.includes("need") || gesture.includes("call"));
      chip.classList.toggle("active", chip.dataset.gesture === activeGesture);
      chipGroup.appendChild(chip);
    });
  }

  function setMode(nextMode, nextGestures) {
    mode = nextMode;
    gestures = nextGestures;
    render(mode, gestures);
  }

  function highlight(gesture) {
    activeGesture = String(gesture).trim().toLowerCase().replace(/\s+/g, "_");
    chipGroup.querySelectorAll(".chip").forEach((chip) => {
      chip.classList.toggle("active", chip.dataset.gesture === activeGesture);
    });
  }

  render(mode, gestures);

  return {
    element,
    setMode,
    highlight
  };
}
