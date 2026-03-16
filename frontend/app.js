import { createCameraView } from "./components/cameraView.js";
import { createDetectionCard } from "./components/detectionCard.js";
import { createEmergencyCard } from "./components/emergencyCard.js";
import { createEyeAssistPanel } from "./components/eyeAssistPanel.js";
import { createGestureChips } from "./components/gestureChips.js";
import { createMetricsPanel } from "./components/metricsPanel.js";
import { createModeSelector } from "./components/modeSelector.js";
import { createNotificationBanner } from "./components/notificationBanner.js";
import { createStatusStrip } from "./components/statusStrip.js";
import { createTeachPanel } from "./components/teachPanel.js";
import { createEventRouter } from "./utils/eventRouter.js";
import {
  DEFAULT_SOCKET_URL,
  DEFAULT_START_PAYLOAD,
  MODE_OPTIONS,
  createRequestId,
  createSessionId,
  getGesturesForMode,
  humanizeGesture
} from "./utils/constants.js";
import { createEnvelope } from "./utils/schemaValidator.js";
import { SocketClient } from "./utils/socketClient.js";

const ui = typeof document !== "undefined"
  ? {
      connectionStatus: document.getElementById("connectionStatus"),
      sessionBadge: document.getElementById("sessionBadge"),
      sessionMeta: document.getElementById("sessionMeta"),
      modeSelectorMount: document.getElementById("modeSelectorMount"),
      notificationMount: document.getElementById("notificationMount"),
      cameraViewMount: document.getElementById("cameraViewMount"),
      detectionCardMount: document.getElementById("detectionCardMount"),
      statusStripMount: document.getElementById("statusStripMount"),
      gestureChipsMount: document.getElementById("gestureChipsMount"),
      teachPanelMount: document.getElementById("teachPanelMount"),
      emergencyCardMount: document.getElementById("emergencyCardMount"),
      eyeAssistPanelMount: document.getElementById("eyeAssistPanelMount"),
      metricsPanelMount: document.getElementById("metricsPanelMount"),
      errorPanelMount: document.getElementById("errorPanelMount"),
      startSessionBtn: document.getElementById("startSessionBtn"),
      stopSessionBtn: document.getElementById("stopSessionBtn")
    }
  : null;

const appState = {
  mode: "default",
  sessionId: createSessionId(),
  sessionActive: false,
  connectionState: "connecting",
  connectionLabel: "Connecting backend",
  lastPrediction: null,
  mediaStream: null
};

let components = null;
let router = null;
let client = null;

function bootstrap() {
  if (!ui) {
    return;
  }

  router = createEventRouter();
  components = mountComponents();
  bindUiActions();
  bindRouteHandlers();
  updateHeader();

  client = new SocketClient({ url: readSocketUrl() });
  client.on("status", handleConnectionStatus);
  client.on("message", (message) => {
    router.route(message);
  });
  client.connect();
}

function mountComponents() {
  const notificationBanner = createNotificationBanner();
  ui.notificationMount.appendChild(notificationBanner.bannerElement);
  ui.errorPanelMount.appendChild(notificationBanner.errorElement);

  const modeSelector = createModeSelector({
    modes: MODE_OPTIONS,
    activeMode: appState.mode,
    onSelect: handleModeChange
  });
  ui.modeSelectorMount.appendChild(modeSelector.element);

  const cameraView = createCameraView({ mode: appState.mode });
  ui.cameraViewMount.appendChild(cameraView.element);

  const detectionCard = createDetectionCard();
  ui.detectionCardMount.appendChild(detectionCard.element);

  const statusStrip = createStatusStrip({ mode: appState.mode });
  ui.statusStripMount.appendChild(statusStrip.element);

  const gestureChips = createGestureChips({
    mode: appState.mode,
    gestures: getGesturesForMode(appState.mode)
  });
  ui.gestureChipsMount.appendChild(gestureChips.element);

  const teachPanel = createTeachPanel({
    mode: appState.mode,
    onTeach: handleTeachGesture
  });
  ui.teachPanelMount.appendChild(teachPanel.element);

  const emergencyCard = createEmergencyCard({
    onAcknowledge: acknowledgeEmergency
  });
  ui.emergencyCardMount.appendChild(emergencyCard.element);

  const eyeAssistPanel = createEyeAssistPanel({ mode: appState.mode });
  ui.eyeAssistPanelMount.appendChild(eyeAssistPanel.element);

  const metricsPanel = createMetricsPanel();
  ui.metricsPanelMount.appendChild(metricsPanel.element);

  return {
    notificationBanner,
    modeSelector,
    cameraView,
    detectionCard,
    statusStrip,
    gestureChips,
    teachPanel,
    emergencyCard,
    eyeAssistPanel,
    metricsPanel
  };
}

function bindUiActions() {
  ui.startSessionBtn.addEventListener("click", startSession);
  ui.stopSessionBtn.addEventListener("click", stopSession);
}

function bindRouteHandlers() {
  router.on("event.prediction_final", ({ payload }) => {
    const rawSign = payload.sign || payload.label || payload.detected_sign || "Awaiting gesture";
    const detectedSign = humanizeGesture(rawSign);
    const confidence = toPercent(payload.confidence, 0);
    const source = payload.source || payload.detection_source || "vision_pipeline";
    const speech = payload.speech_text || payload.speech || detectedSign;

    appState.lastPrediction = rawSign;
    components.detectionCard.update({
      detectedSign,
      confidence,
      source,
      speech
    });
    components.cameraView.update({
      detectedSign,
      subtitle: speech,
      recognitionState: `Source: ${source}`
    });
    components.gestureChips.highlight(rawSign);
  });

  router.on("event.frame_status", ({ payload }) => {
    const frameMode = payload.mode || appState.mode;
    const latencyMs = Number(payload.latency_ms ?? payload.latency ?? 0);
    const handDetected = Boolean(payload.hand_detected);
    const faceDetected = Boolean(payload.face_detected);
    const handCount = Number(payload.hand_count ?? 0);

    components.statusStrip.update({
      mode: frameMode,
      handDetected,
      handCount,
      faceDetected,
      latencyMs
    });
    components.cameraView.update({
      handDetected,
      faceDetected,
      latencyMs
    });
  });

  router.on("event.emergency_alert", ({ payload }) => {
    components.emergencyCard.update({
      intent: humanizeGesture(payload.intent || "Emergency"),
      priority: payload.priority || "High",
      confidence: toPercent(payload.confidence, 0),
      speech: payload.speech_text || payload.speech || "Emergency assistance required",
      requiresAck: Boolean(payload.requires_ack)
    });
  });

  router.on("event.teach_result", ({ payload }) => {
    components.teachPanel.showResult({
      success: Boolean(payload.success),
      message: payload.message || `Gesture "${payload.label || "gesture"}" saved`
    });
  });

  router.on("event.eye_status", ({ payload }) => {
    components.eyeAssistPanel.update({
      leftRatio: asFixed(payload.left_eye_ratio, 2),
      rightRatio: asFixed(payload.right_eye_ratio, 2),
      gazeX: asFixed(payload.gaze_x, 2),
      gazeY: asFixed(payload.gaze_y, 2),
      mappedIntent: payload.mapped_intent || "No mapped intent"
    });
  });

  router.on("event.health_metrics", ({ payload }) => {
    components.metricsPanel.update({
      e2e: formatMilliseconds(payload.e2e_latency_ms),
      capture: formatMilliseconds(payload.capture_ms),
      perception: formatMilliseconds(payload.perception_ms),
      decode: formatMilliseconds(payload.decode_ms),
      speech: formatMilliseconds(payload.speech_ms)
    });
  });

  router.on("event.warning", ({ payload }) => {
    components.notificationBanner.showWarning(payload.message || "Warning received from backend");
  });

  router.on("event.error", ({ payload }) => {
    components.notificationBanner.showError({
      code: payload.code || "UNKNOWN",
      message: payload.message || "An unknown error occurred",
      recovery: payload.recovery || payload.suggested_recovery || "Check camera, backend, and network connectivity"
    });
  });
}

function handleConnectionStatus(status) {
  appState.connectionState = status.state;
  appState.connectionLabel = status.label;
  updateHeader();

  if (status.state === "mock" && status.detail) {
    components.notificationBanner.showWarning(status.detail);
  }

  if (status.state === "connected") {
    components.notificationBanner.clearWarning();
  }
}

function handleModeChange(nextMode) {
  appState.mode = nextMode;
  components.modeSelector.update(nextMode);
  components.statusStrip.update({ mode: nextMode });
  components.cameraView.update({ mode: nextMode });
  components.gestureChips.setMode(nextMode, getGesturesForMode(nextMode));
  components.teachPanel.setMode(nextMode);
  components.eyeAssistPanel.setMode(nextMode);
  components.emergencyCard.setMode(nextMode);
  updateHeader();

  sendCommand("command.switch_mode", { mode: nextMode });
}

async function startSession() {
  if (appState.sessionActive) {
    return;
  }

  appState.sessionActive = true;
  appState.sessionId = createSessionId();
  updateHeader();
  components.cameraView.update({
    sessionActive: true,
    detectedSign: "Session started",
    subtitle: "Waiting for backend predictions",
    recognitionState: `Mode: ${resolveModeLabel(appState.mode)}`
  });

  await startLocalPreview();

  sendCommand("command.start_session", {
    ...DEFAULT_START_PAYLOAD,
    mode: appState.mode
  });
}

async function stopSession() {
  if (!appState.sessionActive) {
    return;
  }

  appState.sessionActive = false;
  updateHeader();
  stopLocalPreview();
  components.cameraView.update({
    sessionActive: false,
    detectedSign: "Session paused",
    subtitle: "Camera feed stopped",
    recognitionState: "Awaiting restart"
  });
  components.statusStrip.update({
    handDetected: false,
    handCount: 0,
    faceDetected: false,
    latencyMs: 0
  });

  sendCommand("command.stop_session", {});
}

function acknowledgeEmergency() {
  sendCommand("command.emergency_ack", {});
  components.emergencyCard.acknowledge();
}

function handleTeachGesture(label) {
  sendCommand("command.teach_sign", {
    label,
    confirm: true
  });
}

function sendCommand(messageType, payload) {
  if (!client) {
    return;
  }

  const envelope = createEnvelope({
    messageType,
    sessionId: appState.sessionId,
    requestId: createRequestId("cmd"),
    payload
  });

  client.send(envelope);
}

async function startLocalPreview() {
  if (!navigator.mediaDevices?.getUserMedia) {
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: DEFAULT_START_PAYLOAD.width,
        height: DEFAULT_START_PAYLOAD.height,
        frameRate: DEFAULT_START_PAYLOAD.fps,
        facingMode: "user"
      },
      audio: false
    });

    appState.mediaStream = stream;
    components.cameraView.attachStream(stream);
  } catch (error) {
    components.notificationBanner.showWarning(
      "Camera preview permission was not granted. Backend events will still be displayed."
    );
  }
}

function stopLocalPreview() {
  if (appState.mediaStream) {
    appState.mediaStream.getTracks().forEach((track) => track.stop());
    appState.mediaStream = null;
  }
  components.cameraView.clearStream();
}

function updateHeader() {
  const modeLabel = resolveModeLabel(appState.mode);

  ui.connectionStatus.textContent = appState.connectionLabel;
  ui.connectionStatus.className = `pill pill-status ${appState.connectionState}`;
  ui.sessionBadge.textContent = appState.sessionActive
    ? `Live session | ${modeLabel}`
    : `Session idle | ${modeLabel}`;
  ui.sessionMeta.textContent = appState.sessionActive
    ? `${appState.sessionId} | ${modeLabel} active`
    : `${modeLabel} ready | awaiting start`;
  ui.startSessionBtn.disabled = appState.sessionActive;
  ui.stopSessionBtn.disabled = !appState.sessionActive;
}

function resolveModeLabel(mode) {
  return MODE_OPTIONS.find((item) => item.value === mode)?.label || mode;
}

function readSocketUrl() {
  if (typeof window === "undefined") {
    return DEFAULT_SOCKET_URL;
  }

  const url = new URL(window.location.href);
  return url.searchParams.get("ws") || DEFAULT_SOCKET_URL;
}

function formatMilliseconds(value) {
  const numeric = Number(value ?? 0);
  return Number.isFinite(numeric) && numeric > 0 ? `${numeric.toFixed(0)} ms` : "--";
}

function toPercent(value, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function asFixed(value, fractionDigits) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(fractionDigits) : "--";
}

bootstrap();

export { bootstrap };
