export function createEventRouter() {
  const handlers = new Map();

  function on(messageType, handler) {
    if (!handlers.has(messageType)) {
      handlers.set(messageType, new Set());
    }
    handlers.get(messageType).add(handler);
    return () => off(messageType, handler);
  }

  function off(messageType, handler) {
    handlers.get(messageType)?.delete(handler);
  }

  function route(message) {
    const messageHandlers = handlers.get(message.message_type);
    if (!messageHandlers) {
      return;
    }
    messageHandlers.forEach((handler) => handler(message));
  }

  return {
    on,
    off,
    route
  };
}
