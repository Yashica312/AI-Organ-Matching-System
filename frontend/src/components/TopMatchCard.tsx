import { Donor } from "@/data/mockData";
import { Trophy, Droplets, Activity, User, MapPin, CheckCircle2, Lightbulb } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface TopMatchCardProps {
  donor: Donor;
  explanation?: string;
}

const matchReasons = [
  { icon: CheckCircle2, text: "Perfect blood group compatibility" },
  { icon: CheckCircle2, text: "High urgency priority applied" },
  { icon: CheckCircle2, text: "Close geographic proximity" },
  { icon: CheckCircle2, text: "Strong ML prediction confidence" },
];

export function TopMatchCard({ donor, explanation }: TopMatchCardProps) {
  return (
    <div className="rounded-xl border-2 border-primary/25 bg-gradient-to-br from-accent via-card to-primary-muted p-5 relative overflow-hidden">
      {/* Decorative elements */}
      <div className="absolute top-0 right-0 w-40 h-40 bg-primary/[0.04] rounded-full -translate-y-1/2 translate-x-1/2" />
      <div className="absolute bottom-0 left-0 w-24 h-24 bg-primary/[0.03] rounded-full translate-y-1/2 -translate-x-1/2" />

      <div className="relative">
        <div className="flex items-center gap-2.5 mb-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary-glow text-primary-foreground animate-pulse-glow shadow-lg shadow-primary/25">
            <Trophy className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-foreground">Best Match</h2>
            <p className="text-[10px] text-muted-foreground">AI-recommended donor</p>
          </div>
          <Badge className="bg-primary/15 text-primary border-primary/25 ml-auto text-xs font-bold px-2.5" variant="outline">
            {donor.score}%
          </Badge>
        </div>

        <p className="text-lg font-extrabold text-foreground mb-3 tracking-tight">{donor.name}</p>

        <div className="grid grid-cols-2 gap-2.5">
          {[
            { icon: Droplets, label: "Blood Group", value: donor.bloodGroup },
            { icon: Activity, label: "HLA Match", value: `${donor.hlaMatch}%` },
            { icon: User, label: "Age", value: `${donor.age} yrs` },
            { icon: MapPin, label: "Location", value: donor.location },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-2 bg-card/60 rounded-lg px-3 py-2 border border-border/50">
              <item.icon className="h-3.5 w-3.5 text-primary shrink-0" />
              <div className="min-w-0">
                <p className="text-[9px] text-muted-foreground uppercase tracking-wider font-medium">{item.label}</p>
                <p className="text-[12px] font-bold text-foreground truncate">{item.value}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Why this donor section */}
        <div className="mt-4 p-3.5 rounded-xl bg-card border border-border">
          <div className="flex items-center gap-2 mb-2.5">
            <Lightbulb className="h-3.5 w-3.5 text-primary" />
            <span className="text-[11px] font-bold text-foreground uppercase tracking-wider">Why This Donor</span>
          </div>
          {explanation ? (
            <p className="text-[12px] text-muted-foreground leading-relaxed">{explanation}</p>
          ) : (
            <div className="space-y-1.5">
              {matchReasons.map((reason, i) => (
                <div key={i} className="flex items-center gap-2">
                  <reason.icon className="h-3 w-3 text-success shrink-0" />
                  <span className="text-[11px] text-muted-foreground">{reason.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
