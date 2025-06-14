import { TrainingProgress, WsMessage, WsMessageType } from "@/types/message";

async function getLatestOfType<T extends WsMessageType>(
  msg_type: T
): Promise<Extract<WsMessage, { type: T }>["payload"]> {
  const res = await fetch(`http://localhost:8000/latest/${msg_type}`);
  return res.json();
}

export async function getTrainingProgress(): Promise<TrainingProgress> {
  const r = await getLatestOfType("training_progress");
  r.runtime = "lkashdfasdf";
  return r;
}
