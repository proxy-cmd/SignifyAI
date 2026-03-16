export const APP_SCHEMA_VERSION = "2.0";
export const DEFAULT_SOCKET_URL = "ws://localhost:8000/ws";

export const DEFAULT_START_PAYLOAD = {
  camera: 0,
  width: 960,
  height: 540,
  fps: 30,
  voice: true
};

export const MODE_OPTIONS = [
  {
    value: "default",
    label: "Realtime Translation",
    shortLabel: "RT",
    description: "Continuous sign interpretation with live teach mode for new gestures."
  },
  {
    value: "demo",
    label: "Demo Mode",
    shortLabel: "DM",
    description: "Curated high-confidence gesture set for presentations and public demos."
  },
  {
    value: "aid",
    label: "Emergency Mode",
    shortLabel: "ER",
    description: "Priority-focused alerting for urgent medical communication intents."
  },
  {
    value: "eye",
    label: "Eye Assist Mode",
    shortLabel: "EA",
    description: "Gaze and blink-assisted communication when hand signing is limited."
  }
];

export const GESTURES_BY_MODE = {
  default: ["one", "two", "i", "y", "q", "l", "r", "v", "w", "yes", "no"],
  demo: [
    "hello",
    "thank_you",
    "please",
    "sorry",
    "yes",
    "no",
    "smile",
    "call_me",
    "one",
    "two",
    "three",
    "four",
    "five",
    "stop",
    "heart"
  ],
  aid: [
    "need_water",
    "need_food",
    "need_toilet",
    "call_family",
    "hospital_help",
    "severe_pain",
    "cannot_breathe",
    "bleeding",
    "head_injury",
    "chest_pain",
    "yes",
    "no"
  ],
  eye: ["emergency", "yes", "need_water", "no", "call_family", "need_food"]
};

export function getGesturesForMode(mode) {
  return GESTURES_BY_MODE[mode] || GESTURES_BY_MODE.default;
}

export function humanizeGesture(gesture) {
  return String(gesture)
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function latencyState(latencyMs) {
  const value = Number(latencyMs ?? 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "";
  }
  if (value < 50) {
    return "good";
  }
  if (value < 100) {
    return "warn";
  }
  return "danger";
}

export function createRequestId(prefix = "req") {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
}

export function createSessionId() {
  return `session_${Date.now()}`;
}
