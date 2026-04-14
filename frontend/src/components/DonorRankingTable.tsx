import { Donor } from "@/data/mockData";
import { Badge } from "@/components/ui/badge";
import { Sparkles } from "lucide-react";

interface DonorRankingTableProps {
  donors: Donor[];
}

export function DonorRankingTable({ donors }: DonorRankingTableProps) {
  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden" style={{ boxShadow: "var(--shadow-card)" }}>
      <div className="flex items-center gap-2 px-5 py-4 border-b border-border bg-gradient-to-r from-card to-muted/30">
        <Sparkles className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Ranked Donors</h2>
        <span className="text-[11px] text-muted-foreground ml-1">({donors.length} compatible)</span>
      </div>
      {donors.length === 0 ? (
        <p className="p-5 text-sm text-muted-foreground">No compatible donors found for this recipient.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                {["#", "Donor", "Blood", "HLA Match", "Location", "Score"].map((h) => (
                  <th key={h} className="px-5 py-2.5 text-[10px] font-semibold text-muted-foreground uppercase tracking-[0.08em] text-left">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {donors.map((d, i) => (
                <tr
                  key={d.id}
                  className={`border-b border-border/40 last:border-0 transition-colors duration-200 ${
                    i === 0 ? "bg-primary/[0.04]" : i === 1 ? "bg-warning/[0.02]" : "hover:bg-primary/[0.02]"
                  }`}
                >
                  <td className="px-5 py-3">
                    {i === 0 ? (
                      <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary-glow text-primary-foreground text-[11px] font-bold shadow-sm">1</span>
                    ) : i === 1 ? (
                      <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-warning/15 text-warning text-[11px] font-bold border border-warning/20">2</span>
                    ) : (
                      <span className="text-muted-foreground text-[12px] pl-2">{i + 1}</span>
                    )}
                  </td>
                  <td className="px-5 py-3">
                    <div>
                      <p className="font-semibold text-foreground text-[13px]">{d.name}</p>
                      <p className="text-[11px] text-muted-foreground">Age {d.age}</p>
                    </div>
                  </td>
                  <td className="px-5 py-3">
                    <Badge variant="outline" className="font-mono text-[11px] bg-muted/50">{d.bloodGroup}</Badge>
                  </td>
                  <td className="px-5 py-3">
                    <div className="flex items-center gap-2.5">
                      <div className="h-1.5 w-14 rounded-full bg-muted overflow-hidden">
                        <div className="h-full rounded-full bg-gradient-to-r from-primary to-primary-glow transition-all duration-500" style={{ width: `${d.hlaMatch}%` }} />
                      </div>
                      <span className="text-[11px] text-muted-foreground font-mono">{d.hlaMatch}%</span>
                    </div>
                  </td>
                  <td className="px-5 py-3 text-muted-foreground text-[12px]">{d.location}</td>
                  <td className="px-5 py-3">
                    <span className={`text-sm font-extrabold ${i === 0 ? "text-gradient-primary" : i === 1 ? "text-warning" : "text-foreground"}`}>
                      {d.score}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
