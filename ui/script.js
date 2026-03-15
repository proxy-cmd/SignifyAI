const ui = {
  startButton: document.getElementById("startSessionBtn"),
  stopButton: document.getElementById("stopSessionBtn"),
  acknowledgeButton: document.getElementById("acknowledgeEmergencyBtn"),
  cameraConnectionStatus: document.getElementById("cameraConnectionStatus"),
  modeBadge: document.getElementById("modeBadge"),
  sessionState: document.getElementById("sessionState"),
  sessionId: document.getElementById("sessionId"),
  feedIntent: document.getElementById("feedIntent"),
  feedHint: document.getElementById("feedHint"),
  detectedIntent: document.getElementById("detectedIntent"),
  confidenceValue: document.getElementById("confidenceValue"),
  recognitionState: document.getElementById("recognitionState"),
  resultBadge: document.getElementById("resultBadge"),
  resultIntent: document.getElementById("resultIntent"),
  resultConfidence: document.getElementById("resultConfidence"),
  resultMode: document.getElementById("resultMode"),
  lastUpdated: document.getElementById("lastUpdated"),
  confidenceBar: document.getElementById("confidenceBar"),
  emergencyPanel: document.getElementById("emergencyPanel"),
  emergencyMessage: document.getElementById("emergencyMessage"),
  emergencyIntent: document.getElementById("emergencyIntent"),
  emergencyConfidence: document.getElementById("emergencyConfidence"),
  predictionLog: document.getElementById("predictionLog"),
  statuses: {
    camera: document.getElementById("statusCamera"),
    hand: document.getElementById("statusHand"),
    face: document.getElementById("statusFace"),
    eye: document.getElementById("statusEye"),
    backend: document.getElementById("statusBackend"),
    latency: document.getElementById("statusLatency")
  },
  metrics: {
    latency: document.getElementById("metricLatency"),
    capture: document.getElementById("metricCapture"),
    processing: document.getElementById("metricProcessing"),
    drops: document.getElementById("metricDrops"),
    cameraHealth: document.getElementById("metricCameraHealth"),
    modelSource: document.getElementById("metricModelSource")
  }
};

const mockModes = ["Gesture + Eye Assist", "Gesture Assist", "Eye Blink Assist"];

const mockPredictions = [
  {
    type: "prediction_final",
    intent: "Need Water",
    confidence: 96,
    mode: "Gesture Assist",
    state: "Gesture recognized successfully"
  },
  {
    type: "prediction_final",
    intent: "Call Nurse",
    confidence: 92,
    mode: "Gesture + Eye Assist",
    state: "Patient assistance requested"
  },
  {
    type: "prediction_final",
    intent: "I Am OK",
    confidence: 89,
    mode: "Gesture Assist",
    state: "Reassurance intent detected"
  },
  {
    type: "prediction_final",
    intent: "Need Help",
    confidence: 94,
    mode: "Gesture + Eye Assist",
    state: "Support needed at bedside"
  }
];

const mockFrameStatuses = [
  {
    type: "frame_status",
    camera: "Active",
    handDetected: "Yes",
    faceDetected: "Yes",
    eyeTracking: "Active",
    backend: "Connected",
    latency: "142 ms"
  },
  {
    type: "frame_status",
    camera: "Active",
    handDetected: "No",
    faceDetected: "Yes",
    eyeTracking: "Active",
    backend: "Connected",
    latency: "157 ms"
  },
  {
    type: "frame_status",
    camera: "Active",
    handDetected: "Yes",
    faceDetected: "Yes",
    eyeTracking: "Inactive",
    backend: "Connected",
    latency: "168 ms"
  }
];

const mockMetrics = [
  {
    type: "health_metrics",
    latency: "142 ms",
    capture: "22 ms",
    processing: "78 ms",
    drops: "1.2%",
    cameraHealth: "Optimal",
    modelSource: "Local Gesture + Eye Model"
  },
  {
    type: "health_metrics",
    latency: "156 ms",
    capture: "24 ms",
    processing: "82 ms",
    drops: "1.6%",
    cameraHealth: "Stable",
    modelSource: "Python OpenCV Stream"
  },
  {
    type: "health_metrics",
    latency: "149 ms",
    capture: "21 ms",
    processing: "79 ms",
    drops: "1.1%",
    cameraHealth: "Optimal",
    modelSource: "Vision Pipeline v1"
  }
];

const emergencySample = {
  type: "emergency_alert",
  intent: "Emergency",
  confidence: 98,
  message: "Double blink or emergency gesture detected. Immediate caregiver response recommended."
};

let appState = {
  isSessionActive: false,
  mockTimer: null,
  sessionCounter: 1000
};

function initializeDashboard() {
  seedPredictionLog();
  setSessionActive(true);
  connectWebSocket();
  bindEvents();
}

function bindEvents() {
  ui.startButton.addEventListener("click", () => setSessionActive(true));
  ui.stopButton.addEventListener("click", () => setSessionActive(false));
  ui.acknowledgeButton.addEventListener("click", acknowledgeEmergencyAlert);
}

function setSessionActive(isActive) {
  const wasActive = appState.isSessionActive;
  appState.isSessionActive = isActive;

  if (isActive) {
    if (!wasActive) {
      appState.sessionCounter += 1;
    }
    ui.sessionId.textContent = `SIG-${appState.sessionCounter}`;
    ui.sessionState.textContent = "Session Active";
    ui.recognitionState.textContent = "Monitoring patient input";
    ui.resultBadge.textContent = "Live";
    ui.resultMode.textContent = ui.modeBadge.textContent;
    setChipState(ui.cameraConnectionStatus, "Camera Connected", "live");
    startMockStream();
  } else {
    ui.sessionState.textContent = "Session Stopped";
    ui.recognitionState.textContent = "Paused by operator";
    ui.resultBadge.textContent = "Paused";
    ui.feedIntent.textContent = "Session paused";
    ui.feedHint.textContent = "Live recognition will resume when the session starts again.";
    ui.detectedIntent.textContent = "Standby";
    ui.resultIntent.textContent = "No active prediction";
    ui.resultConfidence.textContent = "0%";
    ui.confidenceValue.textContent = "0%";
    ui.confidenceBar.style.width = "0%";
    ui.resultMode.textContent = "Idle";
    setChipState(ui.cameraConnectionStatus, "Camera Inactive", "inactive");
    updateSystemStatus({
      camera: "Inactive",
      handDetected: "No",
      faceDetected: "No",
      eyeTracking: "Inactive",
      backend: "Disconnected",
      latency: "--"
    });
    stopMockStream();
  }
}

function setChipState(element, label, variant) {
  element.classList.remove("live", "inactive");
  element.classList.add(variant);
  element.innerHTML = `
    <span class="status-dot ${variant === "live" ? "pulse" : ""}"></span>
    <span>${label}</span>
  `;
}

function startMockStream() {
  stopMockStream();
  runMockCycle();
  appState.mockTimer = window.setInterval(runMockCycle, 3200);
}

function stopMockStream() {
  if (appState.mockTimer) {
    window.clearInterval(appState.mockTimer);
    appState.mockTimer = null;
  }
}

function runMockCycle() {
  if (!appState.isSessionActive) {
    return;
  }

  const prediction = getRandomItem(mockPredictions);
  const frameStatus = getRandomItem(mockFrameStatuses);
  const metrics = getRandomItem(mockMetrics);
  const nextMode = getRandomItem(mockModes);

  updatePrediction(prediction);
  updateSystemStatus(frameStatus);
  updateHealthMetrics(metrics);
  updateMode(nextMode);

  if (Math.random() > 0.72) {
    showEmergencyAlert(emergencySample);
  }
}

function updateMode(mode) {
  ui.modeBadge.textContent = mode;
  ui.resultMode.textContent = mode;
}

function updatePrediction(data) {
  const timestamp = formatTime(new Date());

  ui.feedIntent.textContent = data.intent;
  ui.feedHint.textContent = data.state;
  ui.detectedIntent.textContent = data.intent;
  ui.confidenceValue.textContent = `${data.confidence}%`;
  ui.recognitionState.textContent = data.state;
  ui.resultIntent.textContent = data.intent;
  ui.resultConfidence.textContent = `${data.confidence}%`;
  ui.lastUpdated.textContent = timestamp;
  ui.confidenceBar.style.width = `${data.confidence}%`;

  addLogEntry(timestamp, data.intent, data.mode, data.confidence);
}

function updateSystemStatus(data) {
  applyStatus(ui.statuses.camera, data.camera);
  applyStatus(ui.statuses.hand, data.handDetected);
  applyStatus(ui.statuses.face, data.faceDetected);
  applyStatus(ui.statuses.eye, data.eyeTracking);
  applyStatus(ui.statuses.backend, data.backend);
  applyStatus(ui.statuses.latency, data.latency);
}

function updateHealthMetrics(data) {
  ui.metrics.latency.textContent = data.latency;
  ui.metrics.capture.textContent = data.capture;
  ui.metrics.processing.textContent = data.processing;
  ui.metrics.drops.textContent = data.drops;
  ui.metrics.cameraHealth.textContent = data.cameraHealth;
  ui.metrics.modelSource.textContent = data.modelSource;
}

function showEmergencyAlert(data) {
  ui.emergencyPanel.classList.add("is-active");
  ui.emergencyPanel.classList.remove("is-hidden");
  ui.emergencyMessage.textContent = data.message;
  ui.emergencyIntent.textContent = data.intent;
  ui.emergencyConfidence.textContent = `${data.confidence}%`;
  addLogEntry(formatTime(new Date()), data.intent, "Emergency Path", data.confidence);
}

function acknowledgeEmergencyAlert() {
  ui.emergencyPanel.classList.remove("is-active");
  ui.emergencyPanel.classList.add("is-hidden");
  ui.emergencyMessage.textContent =
    "Emergency reviewed by operator. The panel remains ready for the next high-priority event.";
  ui.emergencyIntent.textContent = "Acknowledged";
  ui.emergencyConfidence.textContent = "--";
}

function applyStatus(element, value) {
  element.textContent = value;
  element.className = "status-pill";

  if (/active|connected|yes|optimal|stable/i.test(value)) {
    element.classList.add("active");
  } else if (/inactive|disconnected|error|emergency/i.test(value)) {
    element.classList.add("danger");
  } else if (/\d+\s?ms/i.test(value)) {
    element.classList.add("info");
  } else if (/warning|limited/i.test(value)) {
    element.classList.add("warning");
  } else {
    element.classList.add("neutral");
  }
}

function addLogEntry(time, intent, mode, confidence) {
  const item = document.createElement("div");
  item.className = "log-entry";
  item.innerHTML = `
    <div>
      <span class="log-entry-time">${time}</span>
      <div class="log-entry-intent">${intent}</div>
    </div>
    <div class="log-entry-time">${mode} | ${confidence}%</div>
  `;

  ui.predictionLog.prepend(item);

  while (ui.predictionLog.children.length > 8) {
    ui.predictionLog.removeChild(ui.predictionLog.lastChild);
  }
}

function seedPredictionLog() {
  [
    ["12:40 PM", "Need Water", "Gesture Assist", 96],
    ["12:41 PM", "Call Nurse", "Gesture + Eye Assist", 92],
    ["12:42 PM", "I Am OK", "Gesture Assist", 89],
    ["12:43 PM", "Need Help", "Gesture + Eye Assist", 94]
  ].forEach(([time, intent, mode, confidence]) => addLogEntry(time, intent, mode, confidence));
}

function getRandomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function formatTime(date) {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

/*
  Backend integration point:
  Replace this mock connector with a real WebSocket to your Python backend.
  Expected message types:
  - prediction_final
  - emergency_alert
  - frame_status
  - health_metrics
  - eye_status
  - warning
  - error
*/
function connectWebSocket() {
  // Example:
  // const socket = new WebSocket("ws://localhost:8000/ws");
  // socket.onmessage = (event) => handleBackendMessage(JSON.parse(event.data));
}

/*
  Backend integration point:
  Use this dispatcher for WebSocket, REST polling, or local mock messages.
*/
function handleBackendMessage(message) {
  switch (message.type) {
    case "prediction_final":
      updatePrediction(message);
      break;
    case "emergency_alert":
      showEmergencyAlert(message);
      break;
    case "frame_status":
    case "eye_status":
      updateSystemStatus(message);
      break;
    case "health_metrics":
      updateHealthMetrics(message);
      break;
    case "warning":
    case "error":
      console.warn("Backend message:", message);
      break;
    default:
      console.info("Unhandled backend message:", message);
  }
}

initializeDashboard();
