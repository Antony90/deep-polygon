// Reinforcement learning objects

type Agent = {
  name: string;
  players: Player[];
};

type Player = {
  reward: number;
  episodeLength: number;
  kills: number;
  land: number;
  rank: number;
};

// Types for our data
type ReplayInfo = {
  id: string;
  agentName: string;
  timestamp: number;
  totalReward: number;
  episodeLength: number;
  kills: number;
  land: number;
};