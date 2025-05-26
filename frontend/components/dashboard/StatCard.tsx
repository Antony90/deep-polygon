"use client";

import { Card, CardContent } from "@/components/ui/card";
import { ArrowUpIcon, ArrowDownIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { LineChart, Line, ResponsiveContainer, ReferenceLine } from "recharts";
import type { LucideIcon } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InfoIcon } from "lucide-react";

interface Metric {
  label: string;
  value: string;
}

interface StatCardCompactProps {
  title: string;
  value: string;
  delta?: string;
  metrics?: Metric[];
  chartColor?: string;
  icon?: React.ReactNode;
  tooltip?: string;
  negative?: boolean; // for statistics where a lower value is better
}

// Helper function to generate fake time series data
function generateTimeSeriesData(
  length: number,
  initialValue: number,
  volatility: number,
  trend = 0
) {
  const data = [];
  let value = initialValue;

  for (let i = 0; i < length; i++) {
    // Add some random noise and a slight trend
    value = Math.max(0, value + (Math.random() * 2 - 1) * volatility + trend);
    data.push({
      time: i,
      value: value,
    });
  }

  return data;
}

export function StatCard({
  title,
  value,
  delta,
  metrics = [],
  chartColor = "#3b82f6",
  icon,
  tooltip,
  negative,
}: StatCardCompactProps) {
  const negativeValue = delta?.trim().startsWith("-");
  const negativeTrend = negativeValue && !negative;

  // Generate fake data for the chart
  const data = generateTimeSeriesData(
    30,
    Number.parseFloat(value),
    Number.parseFloat(value) * 0.1,
    negativeValue ? -0.05 : 0.05
  );

  // Calculate the average value for the reference line
  const avgValue =
    data.reduce((sum, item) => sum + item.value, 0) / data.length;

  return (
    <Card>
      <CardContent>
        <div className="flex items-center justify-between">
          <div className="space-y-1 flex-30 mr-2 lg:mr-0">
            {tooltip && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-2 w-fit">
                      {icon}
                      <h3 className="text-md font-medium text-nowrap">
                        {title}
                      </h3>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="w-[300px] text-md">{tooltip}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold">{value}</div>
              {delta && (
                <div
                  className={cn(
                    "text-sm font-medium flex items-center gap-1",
                    negativeTrend
                      ? "text-red-600 dark:text-red-400"
                      : "text-green-600 dark:text-green-400"
                  )}
                >
                  {negativeValue ? (
                    <ArrowDownIcon className="size-4" />
                  ) : (
                    <ArrowUpIcon className="size-4" />
                  )}
                  {delta}
                </div>
              )}
            </div>
          </div>

          {/* Mini Chart */}
          <div className="h-16 flex-20 lg:flex-10">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data}
                margin={{ top: 0, right: 0, left: 0, bottom: 0 }}
              >
                <ReferenceLine
                  y={avgValue}
                  stroke="#888"
                  strokeDasharray="3 3"
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={chartColor}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Additional Metrics */}
        {metrics.length > 0 && (
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-3">
            {metrics.map((metric, index) => (
              <div key={index} className="flex items-baseline justify-between">
                <span className="text-xs text-muted-foreground text-nowrap">
                  {metric.label}
                </span>
                <span className="text-sm font-medium">{metric.value}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
