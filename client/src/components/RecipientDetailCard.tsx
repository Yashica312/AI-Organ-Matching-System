import { Recipient } from "@/data/mockData";
import { User } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface RecipientDetailCardProps {
  recipient: Recipient;
}

const urgencyColor: Record<string, string> = {
  Critical: "bg-destructive/10 text-destructive border-destructive/20",
  High: "bg-warning/10 text-warning border-warning/20",
  Medium: "bg-primary/10 text-primary border-primary/20",
  Low: "bg-muted text-muted-foreground border-border",
};

export function RecipientDetailCard({ recipient }: RecipientDetailCardProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-5" style={{ boxShadow: "var(--shadow-card)" }}>
      <div className="flex items-center gap-2 mb-4">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-muted">
          <User className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
        <h2 className="text-sm font-semibold text-foreground">Recipient Details</h2>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: "Name", value: recipient.name },
          { label: "Age", value: `${recipient.age} years` },
          { label: "Blood Group", value: recipient.bloodGroup },
          { label: "Organ Needed", value: recipient.organ },
          { label: "Location", value: recipient.location },
          { label: "Wait Time", value: `${recipient.waitTime} days` },
        ].map((item) => (
          <div key={item.label} className="space-y-0.5">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{item.label}</p>
            <p className="text-[13px] font-semibold text-foreground">{item.value}</p>
          </div>
        ))}
        <div className="space-y-0.5 col-span-2">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Urgency</p>
          <Badge className={`${urgencyColor[recipient.urgency]} text-[11px] font-medium`} variant="outline">
            {recipient.urgency}
          </Badge>
        </div>
      </div>
    </div>
  );
}
