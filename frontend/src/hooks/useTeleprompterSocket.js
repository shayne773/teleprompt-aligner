import { useCallback, useMemo, useRef, useState } from 'react';
import { createTeleprompterSocket, sendSocketCommand } from '../api/websocket';

const DEFAULT_STATE = {
  scriptLines: [],
  currentLineIndex: 0,
  currentWordIndex: 0,
  confidence: null,
  isRunning: false
};

function toArrayScript(scriptLike) {
  if (Array.isArray(scriptLike)) return scriptLike;
  if (typeof scriptLike === 'string') {
    return scriptLike
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
  }
  return [];
}

function coerceNumber(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

export default function useTeleprompterSocket() {
  const socketRef = useRef(null);
  const socketUrl = useMemo(
    () => import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/teleprompter',
    []
  );

  const [connectionState, setConnectionState] = useState('disconnected');
  const [state, setState] = useState(DEFAULT_STATE);

  const applyPayload = useCallback((payload) => {
    if (!payload || typeof payload !== 'object') return;

    const root = payload.data && typeof payload.data === 'object' ? payload.data : payload;

    setState((prev) => {
      const merged = {
        ...prev,
        scriptLines:
          toArrayScript(root.script_lines ?? root.script ?? root.lines).length > 0
            ? toArrayScript(root.script_lines ?? root.script ?? root.lines)
            : prev.scriptLines,
        currentLineIndex: coerceNumber(
          root.current_line_index ?? root.line_index ?? root.active_line_index,
          prev.currentLineIndex
        ),
        currentWordIndex: coerceNumber(
          root.current_word_index ?? root.word_index ?? root.active_word_index,
          prev.currentWordIndex
        ),
        confidence:
          root.confidence == null ? prev.confidence : coerceNumber(root.confidence, prev.confidence),
        isRunning:
          root.is_running == null ? prev.isRunning : Boolean(root.is_running)
      };

      return merged;
    });
  }, []);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN) return;

    setConnectionState('connecting');
    const socket = createTeleprompterSocket(socketUrl, {
      onOpen: () => setConnectionState('connected'),
      onClose: () => {
        socketRef.current = null;
        setConnectionState('disconnected');
      },
      onError: () => setConnectionState('error'),
      onMessage: applyPayload
    });

    socketRef.current = socket;
  }, [applyPayload, socketUrl]);

  const disconnect = useCallback(() => {
    if (!socketRef.current) return;
    socketRef.current.close();
    socketRef.current = null;
    setConnectionState('disconnected');
  }, []);

  const start = useCallback(() => {
    console.log('Starting teleprompter...');
    sendSocketCommand(socketRef.current, 'start');
    setState((prev) => ({ ...prev, isRunning: true }));
  }, []);

  const stop = useCallback(() => {
    sendSocketCommand(socketRef.current, 'stop');
    setState((prev) => ({ ...prev, isRunning: false }));
  }, []);

  return {
    connectionState,
    socketUrl,
    connect,
    disconnect,
    start,
    stop,
    ...state
  };
}
