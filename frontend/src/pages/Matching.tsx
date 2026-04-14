import { useEffect, useMemo, useState } from "react";
import { Donor, Recipient } from "@/data/mockData";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sparkles } from "lucide-react";
import { TopMatchCard } from "@/components/TopMatchCard";
import { RecipientDetailCard } from "@/components/RecipientDetailCard";
import { DonorRankingTable } from "@/components/DonorRankingTable";
import { MatchChart } from "@/components/MatchChart";
import { api } from "@/lib/api";

const urgencyLabel = (value: number): Recipient["urgency"] => {
  if (value >= 8) return "Critical";
  if (value >= 6) return "High";
  if (value >= 4) return "Medium";
  return "Low";
};

const toPercent = (value: number) => (value <= 1 ? Math.round(value * 100) : Math.round(value));

const mapRecipient = (recipient: any): Recipient => ({
  id: recipient.id,
  name: recipient.recipient_name,
  age: Number(recipient.recipient_age),
  bloodGroup: recipient.recipient_bg,
  organ: recipient.required_organ,
  urgency: urgencyLabel(Number(recipient.urgency_score)),
  hlaType: "N/A",
  location: "Unknown",
  waitTime: 0,
});

const mapDonor = (donor: any, index: number): Donor => ({
  id: index + 1,
  name: donor.donor_name,
  age: Number(donor.donor_age),
  bloodGroup: donor.donor_bg,
  organ: donor.donor_organ,
  hlaMatch: Math.round(Number(donor.health_score ?? 0) * 100),
  location: donor.distance !== undefined ? `${donor.distance} km` : "Unknown",
  waitTime: 0,
  score: toPercent(Number(donor.predicted_score ?? 0)),
});

const buildExplanation = (donor: any) => {
  const reasons: string[] = [];
  if (Number(donor.blood_compat_score) === 1) reasons.push("Blood match");
  if (Number(donor.health_score ?? 0) >= 0.8) reasons.push("Strong donor health profile");
  if (Number(donor.age_diff ?? 99) < 10) reasons.push("Age compatibility");
  if (reasons.length === 0) reasons.push("High ranking from the matching model");
  return reasons.join(". ");
};

const Matching = () => {
  const [recipients, setRecipients] = useState<Recipient[]>([]);
  const [matches, setMatches] = useState<Donor[]>([]);
  const [bestMatch, setBestMatch] = useState<Donor | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedRecipientId, setSelectedRecipientId] = useState<string>("");
  const [topExplanation, setTopExplanation] = useState("");

  useEffect(() => {
    api.getRecipients()
      .then((data) => setRecipients(data.map(mapRecipient)))
      .catch(() => setError("Failed to fetch recipients"));
  }, []);

  const selectedRecipient = useMemo(
    () => recipients.find((r) => r.id.toString() === selectedRecipientId),
    [selectedRecipientId, recipients]
  );

  const handleMatch = async (recipientId: number) => {
    setLoading(true);
    setError("");
    try {
      const result = await api.getMatch(recipientId);
      const mappedMatches = Array.isArray(result.matches) ? result.matches.map(mapDonor) : [];
      setMatches(mappedMatches);
      setBestMatch(result.best_match ? mapDonor(result.best_match, 0) : null);
      setTopExplanation(result.best_match ? buildExplanation(result.best_match) : "");
      if (result.message && mappedMatches.length === 0) {
        setError(result.message);
      }
    } catch (err) {
      console.error(err);
      setError("Matching failed");
      setMatches([]);
      setBestMatch(null);
      setTopExplanation("");
    } finally {
      setLoading(false);
    }
  };

  const rankedDonors = matches;
  const topDonor = bestMatch ?? rankedDonors[0];

  return (
    <div className="space-y-5 max-w-[1200px]">
      <div className="rounded-xl border border-border bg-card p-5" style={{ boxShadow: "var(--shadow-card)" }}>
        <label className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground mb-2.5 block">
          Select Recipient
        </label>
        <Select
          value={selectedRecipientId}
          onValueChange={(value) => {
            setSelectedRecipientId(value);
            void handleMatch(Number(value));
          }}
        >
          <SelectTrigger className="w-full max-w-lg h-11 rounded-lg text-[13px]">
            <SelectValue placeholder="Choose a recipient to match..." />
          </SelectTrigger>
          <SelectContent>
            {recipients.map((r) => (
              <SelectItem key={r.id} value={r.id.toString()} className="text-[13px]">
                <span className="font-medium">{r.name}</span>
                <span className="text-muted-foreground ml-2">
                  Age {r.age} - {r.bloodGroup} - {r.organ} - {r.urgency}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {loading && <p className="text-sm text-muted-foreground mt-3">Loading matches...</p>}
        {error && <p className="text-sm text-destructive mt-3">{error}</p>}
      </div>

      {selectedRecipient && (
        <div className="animate-fade-up space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <RecipientDetailCard recipient={selectedRecipient} />
            {topDonor && (
              <TopMatchCard donor={topDonor} explanation={topExplanation} />
            )}
          </div>

          {rankedDonors.length > 0 && <MatchChart donors={rankedDonors} />}

          <DonorRankingTable donors={rankedDonors} />
        </div>
      )}

      {!selectedRecipient && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-muted to-muted/50 flex items-center justify-center mb-4 border border-border">
            <Sparkles className="h-7 w-7 text-muted-foreground" />
          </div>
          <p className="text-sm font-semibold text-foreground mb-1">Select a recipient to begin</p>
          <p className="text-[12px] text-muted-foreground max-w-xs leading-relaxed">
            Choose a patient from the dropdown above to see AI-ranked donor matches with compatibility analysis
          </p>
        </div>
      )}
    </div>
  );
};

export default Matching;
