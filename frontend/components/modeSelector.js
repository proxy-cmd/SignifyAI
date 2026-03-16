export function createModeSelector({ modes, activeMode, onSelect }) {
  const element = document.createElement("div");
  element.className = "mode-selector";

  const cards = new Map();

  modes.forEach((mode) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mode-card";
    button.setAttribute("data-mode", mode.value);
    button.innerHTML = `
      <span class="mode-card-icon" aria-hidden="true">${mode.shortLabel}</span>
      <div class="mode-card-copy">
        <div>
          <p class="card-label">${mode.value}</p>
          <h3 class="mode-card-title">${mode.label}</h3>
        </div>
        <p class="mode-card-desc">${mode.description}</p>
      </div>
      <span class="mode-card-foot">Switch to ${mode.shortLabel}</span>
    `;

    button.addEventListener("click", () => {
      onSelect(mode.value);
    });

    cards.set(mode.value, button);
    element.appendChild(button);
  });

  function update(nextMode) {
    cards.forEach((card, modeValue) => {
      const active = modeValue === nextMode;
      card.classList.toggle("active", active);
      card.setAttribute("aria-pressed", String(active));
    });
  }

  update(activeMode);

  return {
    element,
    update
  };
}
