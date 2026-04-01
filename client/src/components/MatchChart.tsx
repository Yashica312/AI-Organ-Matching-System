import { Donor } from "@/data/mockData";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { TrendingUp } from "lucide-react";


interface MatchChartProps {
  donors: Donor[];
}


export function MatchChart({ donors }: MatchChartProps) {
  const chartData = donors.slice(0, 5).map((donor) => ({
    name: donor.name,
    score: donor.score,
    full: donor.name,
  }));

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden" style={{ boxShadow: "var(--shadow-card)" }}>
      <div className="flex items-center gap-2 px-5 py-3.5 border-b border-border bg-gradient-to-r from-card to-muted/30">
        <TrendingUp className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Top 5 Donors by Match Score</h2>
      </div>
      <div className="p-5">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                angle={-15}
                textAnchor="end"
                interval={0}
                height={60}
              />
              <YAxis tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  borderRadius: "10px",
                  border: "1px solid hsl(var(--border))",
                  boxShadow: "var(--shadow-card)",
                  fontSize: "12px",
                }}
                formatter={(value: number, _: string, props: any) => [`${value}%`, props.payload.full]}
              />
              <Bar dataKey="score" radius={[8, 8, 0, 0]} barSize={34}>
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
