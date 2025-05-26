"use client";

import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
const THROTTLE_FPS = 1000;
const THROTTLE_MS = 1000 / THROTTLE_FPS;

export function LivePlayerView() {
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [reward, setReward] = useState(0);

  const live = connected && imgSrc;

  const lastUpdateRef = useRef(0);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onmessage = (e) => {
      // const now = Date.now();
      // if (now - lastUpdateRef.current >= THROTTLE_MS) {
      //   lastUpdateRef.current = now;

      // }
      const { image, reward } = JSON.parse(e.data);
      setImgSrc(image);
    };

    ws.onopen = (e) => {
      setConnected(true);
    };

    ws.onclose = (e) => {
      setConnected(false);
    };

    ws.onerror = (e) => {
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle className="text-lg">Live Player</CardTitle>
        <div
          className={`rounded-md  p-1 flex items-center gap-2 px-2 ${
            live
              ? "bg-red-300/50"
              : connected
              ? "bg-amber-300/50"
              : "bg-neutral-300/50"
          }`}
        >
          <div
            className={`rounded-full size-2 animate-pulse ${
              live
                ? "bg-red-500"
                : connected
                ? "bg-amber-500"
                : "bg-neutral-500"
            }`}
          />
          <p className="font-bold">{connected ? "LIVE" : "CONNECTING"}</p>
        </div>
      </CardHeader>
      <CardContent>
        {imgSrc ? (
          <img
            className="rounded-md ring-1 ring-neutral-300"
            src={imgSrc}
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <div className="animate-pulse rounded-md bg-neutral-300 aspect-square"></div>
        )}
        <div className="flex">
          <div>{reward}</div>
        </div>
      </CardContent>
    </Card>
  );
}
