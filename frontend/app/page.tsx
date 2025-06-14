import { StatsRow } from "@/components/dashboard/StatsRow";
import { TrainingOverviewCard } from "@/components/dashboard/TrainingOverviewCard";
import { MLOverviewCard } from "@/components/dashboard/MLOverviewCard";
import { SpectatorCard } from "@/components/dashboard/SpectatorCard";
import { getTrainingProgress } from "@/lib/api-client";

export default function Home() {
  
  return (
    <div className="max-w-[1600px] mx-auto px-4">
      {/* Heading */}
      <div className="flex justify-between items-center py-4">
        <div className="space-y-2">
          <div className="text-3xl font-bold">Deep Polygon Dashboard</div>
          <div className="text-sm text-muted-foreground">
            A Multi-Agent Deep Reinforcement Learning Simulation
          </div>
        </div>
        <div className="bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 text-xs font-medium px-3 py-1.5 rounded-full flex items-center gap-2">
          <div className="rounded-full bg-green-700 animate-pulse size-2"></div>
          <div>Simulation Running</div>
        </div>
      </div>
      <div className="grid grid-cols-12 gap-4">
        {/* Stats Top Row */}
        <div className="col-span-12 grid grid-cols-1 lg:grid-cols-3 gap-4">
          <StatsRow />
        </div>

        <div className="col-start-1 col-end-7">
          <SpectatorCard />
        </div>
        <div className="col-start-7 col-end-13 space-y-4">
          <TrainingOverviewCard initialTrainingProgress={getTrainingProgress()} />
          <MLOverviewCard />
        </div>

        <div className="col-start-7 col-end-13"></div>
      </div>
    </div>
  );
}
