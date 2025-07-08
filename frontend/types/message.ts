// Unfortunately we have to use python snake case

export interface LiveFrame {
  img: string;
  reward: number;
  total_reward: number;
  ep_length: number;
  kills: number;
  land_captured: number;
  rank: number;
}

export interface TrainingProgress {
  steps: number;
  percent_done: number;
  eta: string;
  rate: number;
  runtime: string;
  gpu_util: number | null;
  cpu_util: number | null;
  epsilon: number;
}

export interface StatisticUpdate {
  avg_ep_reward: {
    mean: number;
    values: number[];
    num_kills: number;
    land_captured: number;
  };
  avg_ep_length: {
    mean: number;
    values: number[];
    max: number;
    min: number;
  };
  avg_loss: {
    mean: number;
    values: number[];
    max: number;
    std_dev: number;
  };
}

// export interface LeaderboardUpdate {
//   agent_name: string;
//   total_reward: number;
//   kills: number;
//   land: number;
//   rank: number;
//   timestamp: number;
// }

// export interface GraphUpdate {
//   graph_name: string;
//   value: number;
// }
export type WsMessage =
  | { type: "live_frame"; payload: LiveFrame }
  | { type: "training_progress"; payload: TrainingProgress }
  | { type: "mean_statistics"; payload: StatisticUpdate }

export type WsMessageType = WsMessage["type"]

  // | { type: "leaderboard"; payload: LeaderboardUpdate }
  // | { type: "graph_update"; payload: GraphUpdate };
