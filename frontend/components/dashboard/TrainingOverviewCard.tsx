"use client";

import { use, useState } from "react";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  CopyIcon,
  PlayIcon,
  PauseIcon,
  SaveIcon,
  ZapIcon,
  CpuIcon,
  GaugeIcon,
  HashIcon,
  BarChart3Icon,
  Gpu,
} from "lucide-react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Progress } from "../ui/progress";
import { useAppSelector } from "@/lib/hooks";
import { TrainingProgress } from "@/types/message";

interface StatComponentProps {
  icon: React.ReactNode;
  label: string;
  value: string | React.ReactNode;
}

const MinorStatPill = ({ icon, label, value }: StatComponentProps) => (
  <div
    className={`flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1`}
  >
    {icon}
    <span className="text-xs text-muted-foreground">{label}</span>
    <span className="text-xs font-medium tabular-nums">{value}</span>
  </div>
);

const StatCard = ({ icon, label, value }: StatComponentProps) => (
  <div className="flex">
    <div className="mr-2 flex size-8 shrink-0 items-center justify-center rounded-md ">
      {icon}
    </div>
    <div>
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <p className="text-xs font-semibold tabular-nums">{value}</p>
    </div>
  </div>
);

export function TrainingOverviewCard() {
  const [paused, setPaused] = useState(false);

  const trainingProgress = useAppSelector(s => s.trainingProgress);
  const {
    steps,
    percent_done: percentComplete,
    eta,
    rate,
    runtime,
    gpu_util: gpuUtil,
    cpu_util: cpuUtil,
    epsilon,
  } = trainingProgress;

  const seed = "12345";
  // const eta = paused ? "Paused" : "02:10:00";
  const totalSteps = 500000;
  const meanInterval = 100;

  const milestones = [10000, 20000, 30000, 40000, 50000].map((m) => ({
    position: (m / totalSteps) * 100,
    completed: steps >= m,
  }));

  const statCards = [
    {
      icon: <ZapIcon className="h-4 w-4 text-primary" />,
      label: "Speed",
      value: `${rate} steps/s`,
      highlight: paused,
    },
    {
      icon: <span className="text-primary font-medium text-lg">ε</span>,
      label: "Exploration",
      value: epsilon.toFixed(2),
    },
    {
      icon: <Gpu className="h-4 w-4 text-primary" />,
      label: "GPU Util",
      value: gpuUtil ? `${gpuUtil}%` : "N/A",
      highlight: paused,
    },
    {
      icon: <CpuIcon className="h-4 w-4 text-primary" />,
      label: "CPU Util",
      value: cpuUtil ? `${cpuUtil}%` : "N/A",
    },
  ];

  return (
    <TooltipProvider>
      <Card>
        <CardHeader className="gap-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold tracking-tight">
              Training Progress
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant={"outline"}
                onClick={() => setPaused(!paused)}
                className={`cursor-pointer h-8 px-2.5 text-xs transition-all ease-in-out duration-300 ${
                  paused ? "bg-amber-500 hover:bg-amber-600 text-white border-none" : ""
                }`}
              >
                {paused ? (
                  <>
                    <PlayIcon className="lg:mr-1 size-4" />
                    <div className="hidden lg:block">Resume</div>
                  </>
                ) : (
                  <>
                    <PauseIcon className="lg:mr-1 size-4" />
                    <div className="hidden lg:block">Pause</div>
                  </>
                )}
              </Button>
              <Button
                size="sm"
                variant="default"
                className="cursor-pointer h-8 px-2.5 text-xs bg-primary text-primary-foreground hover:bg-primary/90"
              >
                <SaveIcon className="lg:mr-1 size-4" />
                <div className="hidden lg:block">Checkpoint</div>
              </Button>
            </div>
          </div>

          <div className="flex flex-col items-start lg:flex-row gap-2 lg:items-center justify-between">
            <MinorStatPill
              icon={<BarChart3Icon className="h-3.5 w-3.5 text-primary" />}
              label="Mean interval:"
              value={`${meanInterval} episodes`}
            />
            <div
              className="cursor-pointer"
              onClick={() => navigator.clipboard.writeText(seed)}
              title="Click to copy seed"
              role="button"
              tabIndex={0}
              onKeyDown={(e) =>
                e.key === "Enter" && navigator.clipboard.writeText(seed)
              }
            >
              <MinorStatPill
                icon={<HashIcon className="h-3.5 w-3.5 text-primary" />}
                label="Seed:"
                value={
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono">{seed}</span>{" "}
                    <CopyIcon className="h-3 w-3 text-muted-foreground" />
                  </div>
                }
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border bg-card/50 p-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-bold tabular-nums text-primary">
                  {percentComplete}%
                </span>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span className="-mt-0.5">Step</span>
                  <span className="font-mono tabular-nums">
                    {steps.toLocaleString()}
                  </span>
                  <span>/</span>
                  <span className="font-mono tabular-nums">
                    {totalSteps.toLocaleString()}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <div className="flex items-center gap-1">
                  <span className="text-muted-foreground">Runtime:</span>
                  <span className="font-medium tabular-nums">{runtime}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-muted-foreground">ETA:</span>
                  <span
                    className={`font-medium tabular-nums ${
                      paused ? "text-amber-500" : ""
                    }`}
                  >
                    {eta}
                  </span>
                </div>
              </div>
            </div>
            <Progress value={percentComplete} />
          </div>
        </CardContent>
        <CardFooter>
          <div className="w-full grid grid-cols-2 lg:grid-cols-4 gap-2">
            {statCards.map((stat, i) => (
              <StatCard key={i} {...stat} />
            ))}
          </div>
        </CardFooter>
      </Card>
    </TooltipProvider>
  );
}
