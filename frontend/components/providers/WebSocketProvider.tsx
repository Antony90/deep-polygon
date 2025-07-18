"use client";
import { setLiveFrame } from "@/lib/features/liveFrameSlice";
import { setTrainingProgress } from "@/lib/features/trainingProgressSlice";
import { useAppDispatch } from "@/lib/hooks";
import { LiveFrame, WsMessage } from "@/types/message";
import { useEffect, useRef, useState } from "react";

/**
 * This component manages the lifecycle of the WebSocket connection,
 * establishing it on mount and cleaning up on unmount. It listens to incoming
 * WebSocket messages and updates the global Zustand store accordingly.
 *
 * Unlike typical React providers, it does NOT use React context to expose values
 * because global state is handled by Zustand. Instead, it acts as a side-effect
 * manager and should be mounted once near the root of the app.
 *
 * The reason this is a separate component is to preserve Next.js server components’
 * benefits. Since WebSocket connections require client-side APIs, this component
 * is a client component that can be mounted alongside or within server components
 * without making the entire tree a client component.
 *
 * Usage:
 * Place <WebSocketProvider /> at the top level of your component tree
 * to ensure a single WebSocket connection shared throughout the app.
 */

function getWebSocketUrl(path = "/ws"): string {
  // Catch Server Side Rendering, we only care about client
  if (typeof window === "undefined") {
    return "";
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const hostname = window.location.hostname;

  const isDevEnv =
    hostname === "localhost" ||
    hostname.startsWith("192.168.") ||
    hostname.startsWith("10.") ||
    hostname === "127.0.0.1";

  if (isDevEnv) {
    return `${protocol}://${hostname}:8000${path}`;
  }

  // Docker compose environment
  if (hostname === "frontend") {
    return `${protocol}://rl_backend:8000${path}`;
  }

  // Production
  return `${protocol}://${hostname}${path}`;
}



export default function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const wsUrl = getWebSocketUrl();
  
  const dispatch = useAppDispatch();
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    wsRef.current = new WebSocket(wsUrl);
    const ws = wsRef.current; // For simplicity

    ws.onopen = () => {
      setConnected(true);
      console.log("WebSocket connected");
    };

    ws.onclose = () => {
      setConnected(false);
      console.log("WebSocket disconnected");
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      ws.close();
    };

    ws.onmessage = (event) => {
      try {
        const { type, payload }: WsMessage = JSON.parse(event.data);
        console.log(event.data);

        switch (type) {
          case "live_frame":
            dispatch(setLiveFrame(payload));
            break;

          case "training_progress":
            dispatch(setTrainingProgress(payload));
            break;

          // case "mean_statistics":
          //   setMeanStatistics(payload);
          //   break;

          // case "leaderboard":
          //   setLeaderboard(payload);
          //   break;

          // case "graph_update":
          //   updateGraphs(payload);
          //   break;

          default:
            console.warn("Unknown message type:", type);
        }
      } catch (err) {
        console.warn("Invalid message received:", event.data);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  return children;
}
