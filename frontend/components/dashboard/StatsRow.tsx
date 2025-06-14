"use client";
import { StatCard } from "@/components/dashboard/StatCard";
import { TrophyIcon, Timer, NetworkIcon } from "lucide-react";
import { use, useEffect, useState } from "react";

export function StatsRow() {
  
  return (
    <>
      <StatCard
        title="Avg Total Reward"
        value={"912.17"}
        delta="+12.5%"
        metrics={[
          { label: "Kills", value: "4.2" },
          { label: "Area Captured", value: "852" },
        ]}
        chartColor="#22c55e"
        icon={<TrophyIcon className="size-5 text-muted-foreground" />}
        tooltip="Average total reward across all episodes, indicating overall agent performance"
      />
      <StatCard
        title="Avg Episode Length"
        value="153"
        delta="+8.1%"
        metrics={[
          { label: "Max Length", value: "312" },
          { label: "Min Length", value: "87" },
        ]}
        chartColor="#3b82f6"
        icon={<Timer className="size-5 text-muted-foreground" />}
        tooltip="Average number of steps per episode, longer episodes typically indicate better survival"
      />
      <StatCard
        title="Value Loss"
        value="0.0342"
        delta="-2.7%"
        negative
        metrics={[
          { label: "Max Loss", value: "0.047" },
          { label: "Standard Deviation", value: "0.019" },
        ]}
        chartColor="#ff0000"
        icon={<NetworkIcon className="size-5 text-muted-foreground" />}
        tooltip="Average value function loss, lower values indicate more stable learning"
      />
    </>
  );
}
