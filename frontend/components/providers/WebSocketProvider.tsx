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
export default function WebSocketProvider({ url, children }: { url: string, children: React.ReactNode }) {
  const dispatch = useAppDispatch();
  
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  
  // const setLatestFrame = (x: any) => {};
  // const setTrainingProgress = (x: any) => {};
  // const setMeanStatistics = (x: any) => {};
  // const setLeaderboard = (x: any) => {};
  // const updateGraphs = (x: any) => {};

  useEffect(() => {
    wsRef.current = new WebSocket(url);
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

