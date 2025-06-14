"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  ArrowUpIcon,
  Users,
  MapPin,
  Radio,
  Clock,
  Timer,
  Trophy,
  Flag,
  Sword,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Switch } from "../ui/switch";
import { useAppSelector } from "@/lib/hooks";

// Types for our data
type Agent = {
  id: string;
  name: string;
  players: Player[];
};

type Player = {
  id: string;
  name: string;
  reward: number;
  episodeLength: number;
  kills: number;
  land: number;
  rank: number;
};

type ReplayMeta = {
  id: number;
  agentName: string;
  timestamp: Date;
  finalReward: number;
  episodeLength: number;
  kills: number;
  land: number;
  rank: number;
};

function ReplayLeaderboardSidebar({ replays }: { replays: ReplayMeta[] }) {
  const [selectedReplayId, setSelectedReplayId] = useState<number | null>(null);

  const handleReplaySelect = (id: number) => {
    setSelectedReplayId(id);
  };

  // Sort replays by finalReward in descending order
  const sortedReplays = [...replays].sort(
    (a, b) => b.finalReward - a.finalReward
  );

  return (
    <div>
      <div className="flex gap-2 items-center mb-3 px-2 pt-2">
        <Trophy className="size-4 text-amber-500" />
        <h3 className="text-sm font-medium">Leaderboard</h3>
      </div>
      <div className="flex flex-col gap-2">
        {sortedReplays.map((replay, index) => {
          // Determine background color based on position
          // Gold, Silver, Bronze

          const bgColor = [
            "bg-amber-50 dark:bg-amber-900/40",
            "bg-slate-50 dark:bg-slate-800/40",
            "bg-orange-50 dark:bg-orange-900/30",
          ];

          const hoverColors = [
            "hover:bg-amber-100 dark:hover:bg-amber-800",
            "hover:bg-slate-100 dark:hover:bg-slate-700",
            "hover:bg-orange-100 dark:hover:bg-orange-800",
          ];

          const hoverColor =
            hoverColors[index] || "hover:bg-muted/60 dark:hover:bg-muted/30";

          // Format timestamp
          const formattedDate = format(
            new Date(replay.timestamp),
            "MMM d, HH:mm"
          );

          return (
            <div
              key={replay.id}
              className={cn(
                "p-2 rounded-md cursor-pointer text-sm transition-colors inset-ring-muted-foreground/20 ",
                bgColor[index],
                hoverColor,
                selectedReplayId === replay.id
                  ? "inset-ring-1" // Highlight selected
                  : "inset-ring-0"
              )}
              onClick={() => handleReplaySelect(replay.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="font-medium text-xs text-muted-foreground">
                    #{index + 1}
                  </span>
                  <span className="font-medium truncate max-w-[80px]">
                    {replay.finalReward.toFixed(1)}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {formattedDate}
                </span>
              </div>
              <div className="mt-1 text-xs text-muted-foreground truncate pl-1">
                {replay.agentName}
              </div>

              {/* Stats row */}
              <div className="flex justify-between mt-1.5 px-1">
                <div className="flex items-center">
                  <Sword
                    className="size-3.5 text-red-500/50 mr-1"
                    strokeWidth={2.5}
                  />
                  <span className="text-xs">{replay.kills}</span>
                </div>
                <div className="flex items-center">
                  <Flag
                    className="size-3.5 text-blue-500/80 mr-1"
                    strokeWidth={2.5}
                  />
                  <span className="text-xs">{replay.land}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PlayerSelectSidebar({ agents }: { agents: Agent[] }) {
  const [selectedAgentId, setSelectedAgentId] = useState<string>("agent-1");
  const [selectedPlayerId, setSelectedPlayerId] =
    useState<string>("agent-1-player-1");

  const [autoFollowBest, setAutoFollowBest] = useState(false);

  const [bestPlayer, setBestPlayer] = useState<Player | null>(null);
  useEffect(() => {
    setBestPlayer(agents[0]?.players[0]);
  }, [agents]);

  const handlePlayerSelect = (playerId: string) => {
    if (!autoFollowBest) {
      setSelectedPlayerId(playerId);
    }
  };

  const currentAgent = agents.find((a) => a.id === selectedAgentId);

  return (
    <div className="flex flex-col gap-4">
      {/* Auto-follow toggle */}
      <div className="rounded-md border bg-card text-card-foreground">
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="size-3.5 text-primary" />
              <span className="text-xs font-medium">Auto-follow Best</span>
            </div>
            <Switch
              checked={autoFollowBest}
              onCheckedChange={setAutoFollowBest}
            />
          </div>

          {/* Animated content that appears when auto-follow is enabled */}
          <div
            className={cn(
              "grid transition-all duration-300 ease-in-out",
              autoFollowBest && bestPlayer
                ? "grid-rows-[1fr] mt-3 opacity-100"
                : "grid-rows-[0fr] mt-0 opacity-0"
            )}
          >
            <div className="overflow-hidden">
              {bestPlayer && (
                <div className="text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Following:</span>
                    <span className="font-medium">{bestPlayer.name}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Agent:</span>
                    <span>{currentAgent?.name}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Reward:</span>
                    <span>{bestPlayer.reward.toFixed(1)}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Agent selector - blurred when auto-following */}
      <div
        className={cn(
          "transition-all duration-300",
          autoFollowBest ? "opacity-60 blur-[1px]" : ""
        )}
      >
        <label
          htmlFor="agent-select"
          className="text-xs font-medium mb-1.5 block"
        >
          Agent
        </label>
        <Select
          value={selectedAgentId}
          onValueChange={setSelectedAgentId}
          disabled={autoFollowBest}
        >
          <SelectTrigger id="agent-select">
            <SelectValue placeholder="Select agent" />
          </SelectTrigger>
          <SelectContent>
            {agents.map((agent) => (
              <SelectItem key={agent.id} value={agent.id}>
                {agent.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Players list */}
      <div
        className={cn(
          "transition-all duration-300",
          autoFollowBest ? "opacity-60 blur-[1px]" : ""
        )}
      >
        <label className="text-xs font-medium mb-1.5 block">Players</label>
        <div className="flex flex-col gap-2">
          {currentAgent?.players.map((player) => (
            <button
              key={player.id}
              className={cn(
                "w-full text-left p-2 rounded-md text-xs flex items-center justify-between",
                "transition-colors duration-200",
                "hover:bg-accent hover:text-accent-foreground",
                player.id === selectedPlayerId
                  ? "bg-accent text-accent-foreground"
                  : "bg-background text-foreground",
                autoFollowBest && "pointer-events-none"
              )}
              onClick={() => handlePlayerSelect(player.id)}
              disabled={autoFollowBest}
            >
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    "size-1.5 rounded-full",
                    player.id === selectedPlayerId
                      ? "bg-primary"
                      : "bg-primary/20"
                  )}
                />
                <span>{player.name}</span>
              </div>
              <span>{player.reward.toFixed(1)}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatsFooter({
  totalReward,
  rank,
  episodeLength,
  kills,
  land,
}: {
  [key: string]: number;
}) {
  return (
    <div className="grid grid-cols-4 gap-2 mt-2">
      <div>
        <div className="flex items-center gap-1">
          <Trophy className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Reward</span>
          <ArrowUpIcon className="h-3 w-3 text-green-500" />
        </div>
        <span className="text-base font-medium tabular-nums">
          {totalReward.toFixed(2)}
        </span>
      </div>
      {/* TODO: Animation when in top 3 */}
      {/* <div className="">
        <div className="flex items-center gap-1">
          <Hash className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Rank</span>
        </div>
        <span className="block text-base font-medium tabular-nums">{rank}</span>
      </div> */}
      <div>
        <div className="flex items-center gap-1">
          <Timer className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Length</span>
        </div>
        <span className="block text-base font-medium tabular-nums">
          {episodeLength}
        </span>
      </div>
      <div>
        <div className="flex items-center gap-1">
          <Users className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Kills</span>
        </div>
        <span className="block text-base font-medium tabular-nums">
          {kills}
        </span>
      </div>
      <div>
        <div className="flex items-center gap-1">
          <MapPin className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Land</span>
        </div>
        <span className="block text-base font-medium tabular-nums">{land}</span>
      </div>
    </div>
  );
}

// Render the main content area with image and stats
function StateView({
  totalReward,
  rank,
  episodeLength,
  kills,
  land,
  isLive,
  isLiveView,
  imgSrc,
}: {
  totalReward: number;
  rank: number;
  episodeLength: number;
  kills: number;
  land: number;
  isLiveView: boolean;
  isLive: boolean;
  imgSrc: string | null;
}) {
  return (
    <div className="flex-1 flex flex-col min-w-0">
      <div className="relative aspect-square w-full mb-2">
        {isLiveView && (
          <Badge
            variant={isLive ? "default" : "outline"}
            className="absolute top-2 right-2 z-10 gap-1"
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                isLive ? "bg-red-500 animate-pulse" : "bg-gray-500"
              }`}
            />
            {isLive ? "LIVE" : "CONNECTING"}
          </Badge>
        )}

        {imgSrc ? (
          <div className="h-full w-full overflow-hidden rounded-md border border-border">
            <img
              className="h-full w-full object-cover"
              src={imgSrc || "/placeholder.svg"}
              alt={isLiveView ? "Live agent view" : "Replay view"}
              style={{ imageRendering: "pixelated" }}
            />
          </div>
        ) : (
          <div className="h-full w-full animate-pulse rounded-md bg-muted"></div>
        )}
      </div>

      <StatsFooter
        totalReward={totalReward}
        rank={23}
        episodeLength={episodeLength}
        kills={kills}
        land={land}
      />
    </div>
  );
}

export function SpectatorCard() {
  const liveFrame = useAppSelector((s) => s.liveFrame);
  const { img, reward, total_reward: totalReward, ep_length: episodeLength, kills, land_captured: landCaptured, rank } =
    liveFrame;

  // State for view mode
  const [activeView, setActiveView] = useState<"live" | "replay">("live");

  // Live view state
  // const [connected, setConnected] = useState<boolean>(false);

  // Replay view state
  // Shared state (will be populated based on selected player/replay)

  // const [totalReward, setTotalReward] = useState<number>(0);
  // const [episodeLength, setEpisodeLength] = useState<number>(0);
  // const [kills, setKills] = useState<number>(0);
  // const [landCaptured, setLandCaptured] = useState<number>(0);
  // const [rank, setRank] = useState<number>(0);

  const [agents, setAgents] = useState<Agent[]>([]);
  const [replays, setReplays] = useState<ReplayMeta[]>([]);
  // Determine if we're live
  // const isLive = activeView === "live" && connected && !!imgSrc;
  const isLive = true;

  // // Mock data initialization
  // useEffect(() => {
  //   // Create mock agents and players
  //   const mockAgents: Agent[] = [
  //     {
  //       id: "agent-1",
  //       name: "Agent Alpha",
  //       players: Array.from({ length: 5 }, (_, i) => ({
  //         id: `agent-1-player-${i + 1}`,
  //         name: `Player ${i + 1}`,
  //         reward: 100 + Math.random() * 200,
  //         episodeLength: 100 + Math.floor(Math.random() * 200),
  //         kills: Math.floor(Math.random() * 10),
  //         land: Math.floor(Math.random() * 1000),
  //         rank: i + 1,
  //       })),
  //     },
  //     {
  //       id: "agent-2",
  //       name: "Agent Beta",
  //       players: Array.from({ length: 5 }, (_, i) => ({
  //         id: `agent-2-player-${i + 1}`,
  //         name: `Player ${i + 1}`,
  //         reward: 100 + Math.random() * 200,
  //         episodeLength: 100 + Math.floor(Math.random() * 200),
  //         kills: Math.floor(Math.random() * 10),
  //         land: Math.floor(Math.random() * 1000),
  //         rank: i + 6,
  //       })),
  //     },
  //   ];

  //   // Create mock replays
  //   const mockReplays: ReplayMeta[] = Array.from({ length: 10 }, (_, i) => ({
  //     id: i + 1,
  //     playerName: `Player ${Math.floor(Math.random() * 5) + 1}`,
  //     agentName: Math.random() > 0.5 ? "Agent Alpha" : "Agent Beta",
  //     timestamp: new Date(Date.now() - Math.random() * 86400000 * 7),
  //     finalReward: 200 + Math.random() * 300,
  //     episodeLength: 150 + Math.floor(Math.random() * 300),
  //     kills: Math.floor(Math.random() * 15),
  //     land: Math.floor(Math.random() * 1500),
  //     rank: i + 1,
  //   })).sort((a, b) => b.finalReward - a.finalReward);

  //   setAgents(mockAgents);
  //   setReplays(mockReplays);

  //   // Set initial selections
  //   if (mockAgents.length > 0) {
  //     const firstAgent = mockAgents[0];

  //     if (firstAgent.players.length > 0) {
  //       const firstPlayer = firstAgent.players[0];
  //       setReward(firstPlayer.reward);
  //       setEpisodeLength(firstPlayer.episodeLength);
  //       setKills(firstPlayer.kills);
  //       setLand(firstPlayer.land);
  //       setRank(firstPlayer.rank);
  //     }
  //   }

  //   // Mock connection
  //   const mockConnect = setTimeout(() => {
  //     setConnected(true);
  //     setImgSrc(
  //       "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
  //     );
  //   }, 1500);

  //   return () => {
  //     clearTimeout(mockConnect);
  //   };
  // }, []);

  return (
    <Tabs
      value={activeView}
      onValueChange={(value) => setActiveView(value as "live" | "replay")}
    >
      <Card>
        <CardHeader className="flex justify-between items-center">
          <CardTitle className="text-lg">Spectate Player</CardTitle>
          <TabsList className="grid w-[240px] grid-cols-2">
            <TabsTrigger value="live" className="flex items-center gap-1">
              <Radio className="size-3" />
              Live
            </TabsTrigger>
            <TabsTrigger value="replay" className="flex items-center gap-1">
              <Clock className="size-3" />
              Replay
            </TabsTrigger>
          </TabsList>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="w-56 relative">
              <div className="absolute inset-0 overflow-y-auto">
                <TabsContent value="live">
                  {/* Agent/Player Selection */}
                  {/* <PlayerSelectSidebar agents={agents} /> */}
                  {/* Main Content */}
                </TabsContent>
                <TabsContent value="replay">
                  {/* Replay Selection */}
                  {/* <ReplayLeaderboardSidebar replays={replays} /> */}
                </TabsContent>
              </div>
              <div className="absolute bottom-0 inset-x-0 h-6 bg-gradient-to-t from-card" />
            </div>
            <StateView
              isLiveView={activeView == "live"}
              isLive={isLive}
              totalReward={totalReward}
              rank={rank}
              episodeLength={episodeLength}
              kills={kills}
              land={landCaptured}
              imgSrc={img}
            />
          </div>
        </CardContent>
      </Card>
    </Tabs>
  );
}
