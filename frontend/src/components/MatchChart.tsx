import { Donor } from "@/data/mockData";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { TrendingUp } from "lucide-react";

interface MatchChartProps {
  donors: Donor[];
}

export function MatchChart({ donors }: MatchChartProps) {
  const chartData = donors.slice(0, 5).map((d) => ({
    name: d.name.split(" ")[0],
    score: d.score,
    full: d.name,
  }));

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden" style={{ boxShadow: "var(--shadow-card)" }}>
      <div className="flex items-center gap-2 px-5 py-3.5 border-b border-border bg-gradient-to-r from-card to-muted/30">
        <TrendingUp className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Top Donors by Score</h2>
      </div>
      <div className="p-5">
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
              <XAxis type="number" domain={[60, 100]} tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
              <YAxis dataKey="name" type="category" width={70} tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
              <Tooltip
                contentStyle={{
                  borderRadius: "10px",
                  border: "1px solid hsl(var(--border))",
                  boxShadow: "var(--shadow-card)",
                  fontSize: "12px",
                }}
                formatter={(value: number, _: string, props: any) => [`${value}%`, props.payload.full]}
              />
              <Bar dataKey="score" radius={[0, 6, 6, 0]} barSize={20}>
                {chartData.map((_, index) => (
                  <Cell
                    key={index}
                    fill={index === 0 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.35)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
