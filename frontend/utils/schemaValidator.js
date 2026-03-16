import { APP_SCHEMA_VERSION, createRequestId, createSessionId } from "./constants.js";

export function validateEnvelope(candidate) {
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
    return null;
  }

  const {
    schema_version: schemaVersion,
    message_type: messageType,
    session_id: sessionId,
    request_id: requestId,
    timestamp_ms: timestampMs,
    payload
  } = candidate;

  if (typeof schemaVersion !== "string") {
    return null;
  }

  if (typeof messageType !== "string" || !messageType.trim()) {
    return null;
  }

  if (typeof sessionId !== "string" || typeof requestId !== "string") {
    return null;
  }

  if (typeof timestampMs !== "number" || !Number.isFinite(timestampMs)) {
    return null;
  }

  return {
    schema_version: schemaVersion,
    message_type: messageType,
    session_id: sessionId,
    request_id: requestId,
    timestamp_ms: timestampMs,
    payload: payload && typeof payload === "object" && !Array.isArray(payload) ? payload : {}
  };
}

export function createEnvelope({
  messageType,
  sessionId = createSessionId(),
  requestId = createRequestId(),
  payload = {}
}) {
  return {
    schema_version: APP_SCHEMA_VERSION,
    message_type: messageType,
    session_id: sessionId,
    request_id: requestId,
    timestamp_ms: Date.now(),
    payload
  };
}
