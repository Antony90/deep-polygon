import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getWebSocketUrl(path = "/ws"): string {
  'use client';
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

