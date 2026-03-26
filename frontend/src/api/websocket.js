export function createTeleprompterSocket(url, handlers = {}) {
  const ws = new WebSocket(url);

  ws.onopen = () => handlers.onOpen?.();
  ws.onerror = (event) => handlers.onError?.(event);
  ws.onclose = () => handlers.onClose?.();
  ws.onmessage = (event) => {
    let parsed = null;
    try {
      parsed = JSON.parse(event.data);
    } catch {
      parsed = { type: 'raw', payload: event.data };
    }
    handlers.onMessage?.(parsed);
  };

  return ws;
}

export function sendSocketCommand(socket, action, payload = {}) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ action, ...payload }));
}
