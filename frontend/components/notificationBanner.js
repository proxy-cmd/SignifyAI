export function createNotificationBanner() {
  const bannerElement = document.createElement("div");
  bannerElement.className = "banner-stack";
  bannerElement.innerHTML = `
    <div class="banner" data-warning-banner>
      <p class="banner-title">Warning</p>
      <p class="banner-text" data-warning-text></p>
    </div>
  `;

  const errorElement = document.createElement("article");
  errorElement.className = "error-card";
  errorElement.innerHTML = `
    <p class="banner-title" style="color: #b91c1c;">Error Panel</p>
    <p class="error-copy" data-error-message>System errors will appear here.</p>
    <div class="error-grid">
      <div class="error-field">
        <p class="error-title">Error Code</p>
        <p class="error-value" data-error-code>--</p>
      </div>
      <div class="error-field">
        <p class="error-title">Message</p>
        <p class="error-value" data-error-message-detail>No active errors.</p>
      </div>
      <div class="error-field">
        <p class="error-title">Suggested Recovery</p>
        <p class="error-value" data-error-recovery>Reconnect the backend and verify camera access.</p>
      </div>
    </div>
  `;

  const warningBanner = bannerElement.querySelector("[data-warning-banner]");
  const warningText = bannerElement.querySelector("[data-warning-text]");
  const errorMessage = errorElement.querySelector("[data-error-message]");
  const errorCode = errorElement.querySelector("[data-error-code]");
  const errorMessageDetail = errorElement.querySelector("[data-error-message-detail]");
  const errorRecovery = errorElement.querySelector("[data-error-recovery]");

  function showWarning(message) {
    warningText.textContent = message;
    warningBanner.classList.add("visible");
  }

  function clearWarning() {
    warningBanner.classList.remove("visible");
    warningText.textContent = "";
  }

  function showError({ code, message, recovery }) {
    errorMessage.textContent = message;
    errorCode.textContent = code;
    errorMessageDetail.textContent = message;
    errorRecovery.textContent = recovery;
    errorElement.classList.add("visible");
  }

  function clearError() {
    errorElement.classList.remove("visible");
  }

  return {
    bannerElement,
    errorElement,
    showWarning,
    clearWarning,
    showError,
    clearError
  };
}
