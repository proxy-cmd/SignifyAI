import { createRequestId, getGesturesForMode } from "./constants.js";
import { createEnvelope, validateEnvelope } from "./schemaValidator.js";

const MOCK_INTERVAL_MS = 2200;

export class SocketClient {
  constructor({ url }) {
    this.url = url;
    this.socket = null;
    this.listeners = {
      status: new Set(),
      message: new Set()
    };
    this.mockMode = false;
    this.mockTimer = null;
    this.mockState = {
      mode: "default",
      sessionId: "",
      sessionActive: false
    };
  }

  on(eventName, handler) {
    this.listeners[eventName]?.add(handler);
    return () => this.listeners[eventName]?.delete(handler);
  }

  emit(eventName, payload) {
    this.listeners[eventName]?.forEach((handler) => handler(payload));
  }

  connect() {
    if (typeof WebSocket === "undefined") {
      this.activateMock("WebSocket is unavailable in this environment. Demo stream enabled.");
      return;
    }

    this.emit("status", {
      state: "connecting",
      label: "Connecting backend"
    });

    let opened = false;
    let fallbackHandled = false;

    try {
      this.socket = new WebSocket(this.url);
    } catch (error) {
      this.activateMock("Backend socket could not be opened. Demo stream enabled.");
      return;
    }

    const fallbackTimer = window.setTimeout(() => {
      if (!opened) {
        fallbackHandled = true;
        try {
          this.socket?.close();
        } catch (error) {
          // Ignore close errors during fallback.
        }
        this.activateMock("Backend not reachable. Demo stream enabled for presentation use.");
      }
    }, 1400);

    this.socket.addEventListener("open", () => {
      opened = true;
      window.clearTimeout(fallbackTimer);
      this.mockMode = false;
      this.emit("status", {
        state: "connected",
        label: "Backend connected"
      });
    });

    this.socket.addEventListener("message", (event) => {
      const parsed = this.safeParse(event.data);
      const envelope = validateEnvelope(parsed);
      if (!envelope) {
        return;
      }
      this.emit("message", envelope);
    });

    this.socket.addEventListener("error", () => {
      if (!opened && !fallbackHandled) {
        window.clearTimeout(fallbackTimer);
        this.activateMock("Socket error while connecting. Demo stream enabled.");
      } else {
        this.emit("status", {
          state: "error",
          label: "Backend error"
        });
      }
    });

    this.socket.addEventListener("close", () => {
      if (!opened && !fallbackHandled) {
        this.activateMock("Backend connection closed before opening. Demo stream enabled.");
      } else if (!this.mockMode) {
        this.emit("status", {
          state: "error",
          label: "Backend disconnected"
        });
      }
    });
  }

  send(envelope) {
    if (this.mockMode || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.handleMockCommand(envelope);
      return;
    }

    this.socket.send(JSON.stringify(envelope));
  }

  safeParse(raw) {
    try {
      return JSON.parse(raw);
    } catch (error) {
      return null;
    }
  }

  activateMock(detail) {
    this.mockMode = true;
    this.emit("status", {
      state: "mock",
      label: "Demo stream active",
      detail
    });
  }

  handleMockCommand(envelope) {
    const { message_type: messageType, session_id: sessionId, payload } = envelope;
    this.mockState.sessionId = sessionId;

    switch (messageType) {
      case "command.switch_mode":
        this.mockState.mode = payload.mode || this.mockState.mode;
        if (this.mockState.sessionActive) {
          this.emitMockTick();
        }
        break;
      case "command.start_session":
        this.mockState.mode = payload.mode || this.mockState.mode;
        this.mockState.sessionActive = true;
        this.startMockLoop();
        this.emitMockTick();
        break;
      case "command.stop_session":
        this.mockState.sessionActive = false;
        this.stopMockLoop();
        break;
      case "command.emergency_ack":
        break;
      case "command.teach_sign":
        this.emitMockMessage("event.teach_result", {
          success: true,
          label: payload.label,
          message: `Gesture "${payload.label}" added to the local teaching queue.`
        });
        break;
      default:
        break;
    }
  }

  startMockLoop() {
    this.stopMockLoop();
    this.mockTimer = window.setInterval(() => {
      this.emitMockTick();
    }, MOCK_INTERVAL_MS);
  }

  stopMockLoop() {
    if (this.mockTimer) {
      window.clearInterval(this.mockTimer);
      this.mockTimer = null;
    }
  }

  emitMockTick() {
    if (!this.mockState.sessionActive) {
      return;
    }

    const gestures = getGesturesForMode(this.mockState.mode);
    const selectedGesture = gestures[Math.floor(Math.random() * gestures.length)];
    const latencyMs = randomBetween(34, this.mockState.mode === "aid" ? 122 : 92);
    const confidence = randomBetween(82, 98);

    this.emitMockMessage("event.frame_status", {
      mode: this.mockState.mode,
      hand_detected: this.mockState.mode !== "eye",
      hand_count: this.mockState.mode === "eye" ? 0 : randomBetween(1, 2),
      face_detected: true,
      latency_ms: latencyMs
    });

    this.emitMockMessage("event.prediction_final", {
      sign: selectedGesture,
      confidence,
      source: this.mockState.mode === "eye" ? "eye_decoder" : "gesture_classifier",
      speech_text: humanizeSpeech(selectedGesture)
    });

    if (this.mockState.mode === "eye") {
      this.emitMockMessage("event.eye_status", {
        left_eye_ratio: randomFloat(0.26, 0.49),
        right_eye_ratio: randomFloat(0.27, 0.52),
        gaze_x: randomFloat(-0.84, 0.88),
        gaze_y: randomFloat(-0.72, 0.76),
        mapped_intent: humanizeSpeech(selectedGesture)
      });
    }

    this.emitMockMessage("event.health_metrics", {
      e2e_latency_ms: latencyMs,
      capture_ms: randomBetween(10, 22),
      perception_ms: randomBetween(14, 34),
      decode_ms: randomBetween(8, 18),
      speech_ms: randomBetween(6, 20)
    });

    if (this.mockState.mode === "aid" || Math.random() > 0.8) {
      this.emitMockMessage("event.emergency_alert", {
        intent: this.mockState.mode === "aid" ? humanizeSpeech(selectedGesture) : "Manual assistance required",
        priority: this.mockState.mode === "aid" ? "Critical" : "High",
        confidence: randomBetween(89, 99),
        speech_text: `Attention required: ${humanizeSpeech(selectedGesture)}`,
        requires_ack: true
      });
    }

    if (Math.random() > 0.88) {
      this.emitMockMessage("event.warning", {
        message: "Lighting variance detected. Stable frontal lighting will improve confidence."
      });
    }
  }

  emitMockMessage(messageType, payload) {
    this.emit(
      "message",
      createEnvelope({
        messageType,
        sessionId: this.mockState.sessionId || `session_mock_${Date.now()}`,
        requestId: createRequestId("evt"),
        payload
      })
    );
  }
}

function randomBetween(min, max) {
  return Math.round(Math.random() * (max - min) + min);
}

function randomFloat(min, max) {
  return Number((Math.random() * (max - min) + min).toFixed(2));
}

function humanizeSpeech(value) {
  return String(value)
    .split("_")
    .join(" ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
