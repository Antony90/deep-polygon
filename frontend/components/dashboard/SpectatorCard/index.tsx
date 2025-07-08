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
  ArrowDownIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import { useAppSelector } from "@/lib/hooks";


function ReplayLeaderboardSidebar({ replays }: { replays: ReplayInfo[] }) {
  const [selectedReplayId, setSelectedReplayId] = useState<string | null>(null);

  // Sort replays by totalReward in descending order
  const sortedReplays = [...replays].sort(
    (a, b) => b.totalReward - a.totalReward
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
              onClick={() => setSelectedReplayId(replay.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="font-medium text-xs text-muted-foreground">
                    #{index + 1}
                  </span>
                  <span className="font-medium truncate max-w-[80px]">
                    {replay.totalReward.toFixed(1)}
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
  const [selectedAgentIdx, setSelectedAgentIdx] = useState(0);
  const [selectedPlayerIdx, setSelectedPlayerIdx] = useState(0);

  const [autoFollowBest, setAutoFollowBest] = useState(false);

  // TODO: read ws redux state
  const [bestAgentIdx, setBestAgentIdx] = useState(0); // Agent of the player who currently has the highest total reward
  const [bestPlayerIdx, setBestPlayerIdx] = useState(0); // Current highest total reward

  const bestAgent = agents[bestAgentIdx];
  const bestPlayer = bestAgent.players[bestPlayerIdx];

  const selectedAgent = agents[selectedAgentIdx];
  const selectedPlayer = selectedAgent.players[selectedPlayerIdx];

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
              autoFollowBest
                ? "grid-rows-[1fr] mt-3 opacity-100"
                : "grid-rows-[0fr] mt-0 opacity-0"
            )}
          >
            <div className="overflow-hidden">
              <div className="text-xs space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Following:</span>
                  <span className="font-medium">Player {bestPlayerIdx + 1}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Agent:</span>
                  <span>{bestAgent.name}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Reward:</span>
                  <span>{bestPlayer.reward.toFixed(1)}</span>
                </div>
              </div>
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
          value={selectedAgent.name}
          onValueChange={(agentName) => {
            setSelectedAgentIdx(
              agents.findIndex((ag) => ag.name === agentName)
            );
            setSelectedPlayerIdx(0);
          }}
          disabled={autoFollowBest}
        >
          <SelectTrigger id="agent-select">
            <SelectValue placeholder="Select agent" />
          </SelectTrigger>
          <SelectContent>
            {agents.map((agent) => (
              <SelectItem key={agent.name} value={agent.name}>
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
          {selectedAgent.players.map((player, i) => (
            <button
              key={i}
              className={cn(
                "w-full text-left p-2 rounded-md text-xs flex items-center justify-between",
                "transition-colors duration-200",
                "hover:bg-accent hover:text-accent-foreground",
                i === selectedPlayerIdx
                  ? "bg-accent text-accent-foreground"
                  : "bg-background text-foreground",
                autoFollowBest && "pointer-events-none"
              )}
              onClick={() => setSelectedPlayerIdx(i)}
              disabled={autoFollowBest}
            >
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    "size-1.5 rounded-full",
                    i === selectedPlayerIdx
                      ? "bg-primary"
                      : "bg-primary/20"
                  )}
                />
                <span>Player {i + 1}</span>
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
  reward,
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
          {reward > 0 ? (
            <ArrowUpIcon className="h-3 w-3 text-green-500" />
          ) : (
            <ArrowDownIcon className="h-3 w-3 text-red-500" />
          )}
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
  reward,
  rank,
  episodeLength,
  kills,
  land,
  isLive,
  isLiveView,
  imgSrc,
}: {
  totalReward: number;
  reward: number;
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
        reward={reward}
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
  const {
    img,
    reward,
    total_reward: totalReward,
    ep_length: episodeLength,
    kills,
    land_captured: landCaptured,
    rank,
  } = liveFrame;

  // State for view mode
  const [activeView, setActiveView] = useState<"live" | "replay">("live");

  const [agents, setAgents] = useState<Agent[]>([
    {
      name: "Builder",
      players: [{ reward: 0, episodeLength: 0, kills: 0, land: 0, rank: 0 }],
    },
  ]);
  const [replays, setReplays] = useState<ReplayInfo[]>([
    {
      id: "1",
      timestamp: Date.now(),
      agentName: "Builder",
      totalReward: 123.4,
      episodeLength: 258,
      kills: 5,
      land: 12
    }
  ]);
  // Determine if we're live
  // const isLive = activeView === "live" && connected && !!imgSrc;
  const isLive = true;

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
                  <PlayerSelectSidebar agents={agents} />
                </TabsContent>
                <TabsContent value="replay">
                  {/* Replay Selection */}
                  <ReplayLeaderboardSidebar replays={replays} />
                </TabsContent>
              </div>
              <div className="absolute bottom-0 inset-x-0 h-6 bg-gradient-to-t from-card" />
            </div>
            {/* Main Content */}
            <StateView
              isLiveView={activeView == "live"}
              isLive={isLive}
              totalReward={totalReward}
              reward={reward}
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
