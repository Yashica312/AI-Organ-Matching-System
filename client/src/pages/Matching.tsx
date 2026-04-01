import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/components/ui/use-toast";
import { MatchChart } from "@/components/MatchChart";
import { recipients } from "@/data/mockData";
import { Activity, BrainCircuit, Filter, Loader2, MapPin, Sparkles, Trophy } from "lucide-react";

type Compatibility = "High" | "Medium" | "Low";

type MatchResult = {
  donor: string;
  match_score: number;
  compatibility: Compatibility;
  hlaMatch: number;
  waitTime: number;
  explanation: string[];
};

type MatchForm = {
  blood_group: string;
  age: number;
  organ: string;
  urgency: string;
};

type ChartDonor = {
  id: number;
  name: string;
  age: number;
  bloodGroup: string;
  organ: string;
  hlaMatch: number;
  location: string;
  waitTime: number;
  score: number;
};

const API_URL = import.meta.env.VITE_API_URL;
const bloodGroups = ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"];
const organTypes = ["kidney", "heart", "liver", "lung", "pancreas"];
const urgencyLevels = ["low", "medium", "high", "critical"];

const demoMatches: MatchResult[] = [
  {
    donor: "Donor_KID_101",
    match_score: 94,
    compatibility: "High",
    hlaMatch: 91,
    waitTime: 3,
    explanation: ["Blood match", "High HLA similarity", "Age compatibility", "Urgency priority"],
  },
  {
    donor: "Donor_KID_204",
    match_score: 88,
    compatibility: "High",
    hlaMatch: 86,
    waitTime: 4,
    explanation: ["Blood match", "High HLA similarity", "Urgency priority"],
  },
  {
    donor: "Donor_KID_318",
    match_score: 83,
    compatibility: "Medium",
    hlaMatch: 82,
    waitTime: 6,
    explanation: ["Age compatibility", "Urgency priority"],
  },
  {
    donor: "Donor_KID_412",
    match_score: 77,
    compatibility: "Medium",
    hlaMatch: 79,
    waitTime: 8,
    explanation: ["Blood match", "Age compatibility"],
  },
  {
    donor: "Donor_KID_507",
    match_score: 69,
    compatibility: "Low",
    hlaMatch: 73,
    waitTime: 10,
    explanation: ["Urgency priority"],
  },
];

function compatibilityClass(level: Compatibility) {
  if (level === "High") return "bg-success/15 text-success border-success/30";
  if (level === "Medium") return "bg-warning/15 text-warning border-warning/30";
  return "bg-destructive/10 text-destructive border-destructive/20";
}

function compatibilityDot(level: Compatibility) {
  if (level === "High") return "bg-success";
  if (level === "Medium") return "bg-warning";
  return "bg-destructive";
}

function toChartDonors(matches: MatchResult[], organ: string): ChartDonor[] {
  return matches.map((match, index) => ({
    id: index + 1,
    name: match.donor,
    age: 30 + index * 4,
    bloodGroup: "A+",
    organ: organ.charAt(0).toUpperCase() + organ.slice(1),
    hlaMatch: match.hlaMatch,
    location: `Zone ${index + 1}`,
    waitTime: match.waitTime,
    score: match.match_score,
  }));
}

const Matching = () => {
  const { toast } = useToast();
  const [demoMode, setDemoMode] = useState(true);
  const [selectedRecipientId, setSelectedRecipientId] = useState<string>(recipients[0]?.id.toString() ?? "");
  const [form, setForm] = useState<MatchForm>({
    blood_group: recipients[0]?.bloodGroup ?? "A+",
    age: recipients[0]?.age ?? 45,
    organ: recipients[0]?.organ.toLowerCase() ?? "kidney",
    urgency: recipients[0]?.urgency.toLowerCase() ?? "high",
  });
  const [matches, setMatches] = useState<MatchResult[]>(demoMatches);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [compatibilityFilter, setCompatibilityFilter] = useState<string>("all");
  const [organFilter, setOrganFilter] = useState<string>("all");

  useEffect(() => {
    if (!demoMode) return;
    const selectedRecipient = recipients.find((recipient) => recipient.id.toString() === selectedRecipientId);
    if (!selectedRecipient) return;

    setForm({
      blood_group: selectedRecipient.bloodGroup,
      age: selectedRecipient.age,
      organ: selectedRecipient.organ.toLowerCase(),
      urgency: selectedRecipient.urgency.toLowerCase(),
    });
  }, [demoMode, selectedRecipientId]);

  const filteredMatches = useMemo(() => {
    return matches.filter((match) => {
      const compatibilityOk = compatibilityFilter === "all" || match.compatibility === compatibilityFilter;
      const organOk = organFilter === "all" || form.organ === organFilter;
      return compatibilityOk && organOk;
    });
  }, [matches, compatibilityFilter, organFilter, form.organ]);

  const topMatch = filteredMatches[0];
  const chartDonors = useMemo(() => toChartDonors(filteredMatches, form.organ), [filteredMatches, form.organ]);

  const handleSubmit = async () => {
    setLoading(true);
    setError("");

    try {
      if (demoMode) {
        await new Promise((resolve) => setTimeout(resolve, 650));
        setMatches(demoMatches);
        toast({
          title: "Demo results loaded",
          description: "Showing preloaded realistic donor rankings for presentation mode.",
        });
        return;
      }

      const response = await fetch(`${API_URL}/match-multiple`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const payload = await response.json();
      const liveMatches: MatchResult[] = Array.isArray(payload.matches) ? payload.matches : [];

      if (liveMatches.length === 0) {
        throw new Error("No donor matches returned from the API.");
      }

      setMatches(liveMatches);
      toast({
        title: "Live ranking updated",
        description: `Loaded ${liveMatches.length} donor recommendations from the API.`,
      });
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message : "Unable to fetch donor rankings.";
      setError(message);
      toast({
        title: "API request failed",
        description: message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5 max-w-[1240px]">
      <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-5">
        <Card className="border-border/80 shadow-[var(--shadow-elevated)] overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-card via-card to-primary-muted/30">
            <div className="flex items-start justify-between gap-4">
              <div>
                <CardTitle className="text-xl flex items-center gap-2">
                  <BrainCircuit className="h-5 w-5 text-primary" />
                  AI Match Console
                </CardTitle>
                <CardDescription>
                  Multi-donor ranking with explainable AI signals and demo-ready workflow.
                </CardDescription>
              </div>
              <div className="flex items-center gap-3 rounded-xl border border-border bg-card/80 px-3 py-2">
                <div className="space-y-0.5">
                  <p className="text-xs font-semibold text-foreground">Demo Mode</p>
                  <p className="text-[11px] text-muted-foreground">
                    {demoMode ? "Preloaded donor rankings" : "Live FastAPI scoring"}
                  </p>
                </div>
                <Switch checked={demoMode} onCheckedChange={setDemoMode} />
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-6 space-y-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Recipient Preset</Label>
                <Select value={selectedRecipientId} onValueChange={setSelectedRecipientId}>
                  <SelectTrigger className="h-11 rounded-xl">
                    <SelectValue placeholder="Choose recipient" />
                  </SelectTrigger>
                  <SelectContent>
                    {recipients.map((recipient) => (
                      <SelectItem key={recipient.id} value={recipient.id.toString()}>
                        {recipient.name} - {recipient.organ}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Blood Group</Label>
                <Select
                  value={form.blood_group}
                  onValueChange={(value) => setForm((current) => ({ ...current, blood_group: value }))}
                >
                  <SelectTrigger className="h-11 rounded-xl">
                    <SelectValue placeholder="Select blood group" />
                  </SelectTrigger>
                  <SelectContent>
                    {bloodGroups.map((group) => (
                      <SelectItem key={group} value={group}>
                        {group}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Age</Label>
                <Input
                  type="number"
                  min={1}
                  max={120}
                  value={form.age}
                  onChange={(event) =>
                    setForm((current) => ({ ...current, age: Number(event.target.value) || 0 }))
                  }
                  className="h-11 rounded-xl"
                />
              </div>
              <div className="space-y-2">
                <Label>Organ Type</Label>
                <Select
                  value={form.organ}
                  onValueChange={(value) => setForm((current) => ({ ...current, organ: value }))}
                >
                  <SelectTrigger className="h-11 rounded-xl">
                    <SelectValue placeholder="Select organ" />
                  </SelectTrigger>
                  <SelectContent>
                    {organTypes.map((organ) => (
                      <SelectItem key={organ} value={organ}>
                        {organ.charAt(0).toUpperCase() + organ.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2 md:col-span-2">
                <Label>Urgency</Label>
                <Select
                  value={form.urgency}
                  onValueChange={(value) => setForm((current) => ({ ...current, urgency: value }))}
                >
                  <SelectTrigger className="h-11 rounded-xl">
                    <SelectValue placeholder="Select urgency" />
                  </SelectTrigger>
                  <SelectContent>
                    {urgencyLevels.map((urgency) => (
                      <SelectItem key={urgency} value={urgency}>
                        {urgency.charAt(0).toUpperCase() + urgency.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="flex flex-col sm:flex-row sm:items-center gap-3">
              <Button
                onClick={handleSubmit}
                disabled={loading}
                className="h-11 rounded-xl px-5 text-sm font-semibold shadow-lg shadow-primary/20"
              >
                {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Sparkles className="h-4 w-4 mr-2" />}
                Find Best Match
              </Button>
              <p className="text-[12px] text-muted-foreground">
                {demoMode
                  ? "Demo mode keeps the presentation stable with realistic preloaded rankings."
                  : "Live mode requests ranked donors from the FastAPI backend."}
              </p>
            </div>
            {error && (
              <div className="rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive animate-in-page">
                {error}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-border/80 shadow-[var(--shadow-card)]">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Filter className="h-4 w-4 text-primary" />
              Match Filters
            </CardTitle>
            <CardDescription>Refine the visible donor ranking for the final demo.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Compatibility Level</Label>
              <Select value={compatibilityFilter} onValueChange={setCompatibilityFilter}>
                <SelectTrigger className="h-11 rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="High">High</SelectItem>
                  <SelectItem value="Medium">Medium</SelectItem>
                  <SelectItem value="Low">Low</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Organ Type</Label>
              <Select value={organFilter} onValueChange={setOrganFilter}>
                <SelectTrigger className="h-11 rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  {organTypes.map((organ) => (
                    <SelectItem key={organ} value={organ}>
                      {organ.charAt(0).toUpperCase() + organ.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="rounded-xl border border-border bg-muted/25 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground mb-2">
                Live Summary
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[11px] text-muted-foreground">Visible matches</p>
                  <p className="text-xl font-bold text-foreground">{filteredMatches.length}</p>
                </div>
                <div>
                  <p className="text-[11px] text-muted-foreground">Top score</p>
                  <p className="text-xl font-bold text-primary">{topMatch ? `${topMatch.match_score}%` : "--"}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {filteredMatches.length > 0 ? (
        <div className="space-y-5 animate-in-page">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
            {filteredMatches.map((match, index) => (
              <Card
                key={`${match.donor}-${index}`}
                className={`border-border/80 transition-all duration-300 hover:-translate-y-1 hover:shadow-[var(--shadow-card-hover)] ${
                  index === 0 ? "ring-2 ring-primary/30 bg-gradient-to-br from-card via-card to-primary-muted/35" : ""
                }`}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <div
                        className={`h-9 w-9 rounded-xl flex items-center justify-center text-sm font-bold ${
                          index === 0
                            ? "bg-gradient-to-br from-primary to-primary-glow text-primary-foreground shadow-lg shadow-primary/25"
                            : "bg-muted text-foreground"
                        }`}
                      >
                        #{index + 1}
                      </div>
                      <div>
                        <CardTitle className="text-sm leading-tight">{match.donor}</CardTitle>
                        <CardDescription>{index === 0 ? "Top ranked donor" : "Ranked donor option"}</CardDescription>
                      </div>
                    </div>
                    {index === 0 ? <Trophy className="h-4 w-4 text-primary" /> : null}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Match Score</p>
                      <p className="text-2xl font-extrabold text-foreground">{match.match_score}%</p>
                    </div>
                    <Badge variant="outline" className={compatibilityClass(match.compatibility)}>
                      <span className={`mr-1.5 h-2 w-2 rounded-full ${compatibilityDot(match.compatibility)}`} />
                      {match.compatibility}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-muted/35 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">HLA Match</p>
                      <p className="text-sm font-semibold text-foreground">{match.hlaMatch}%</p>
                    </div>
                    <div className="rounded-lg bg-muted/35 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">Wait Time</p>
                      <p className="text-sm font-semibold text-foreground">{match.waitTime} days</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                      Explanation
                    </p>
                    <ul className="space-y-1.5">
                      {match.explanation.map((reason) => (
                        <li key={reason} className="flex items-start gap-2 text-[12px] text-muted-foreground">
                          <span className="mt-1 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                          <span>{reason}</span>
                        </li>
                      ))}
                    </ul>
                    <div className="rounded-lg bg-accent/60 px-3 py-2 text-[11px] text-accent-foreground">
                      Top factor: HLA match ({Math.max(22, Math.min(65, Math.round(match.hlaMatch / 2)))}%)
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <MatchChart donors={chartDonors} />

          <Card className="border-border/80 shadow-[var(--shadow-card)]">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                Ranking Summary
              </CardTitle>
              <CardDescription>Top 5 donor overview for quick clinician review.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {filteredMatches.map((match, index) => (
                <div
                  key={`summary-${match.donor}`}
                  className={`flex flex-col gap-3 rounded-xl border border-border px-4 py-4 md:flex-row md:items-center md:justify-between ${
                    index === 0 ? "bg-primary/[0.05]" : "bg-card"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted text-sm font-bold">
                      #{index + 1}
                    </div>
                    <div>
                      <p className="font-semibold text-foreground">{match.donor}</p>
                      <div className="flex items-center gap-3 text-[12px] text-muted-foreground">
                        <span className="inline-flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          Priority donor
                        </span>
                        <span>{match.waitTime} days estimated wait</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className={compatibilityClass(match.compatibility)}>
                      {match.compatibility}
                    </Badge>
                    <p className="text-lg font-bold text-foreground">{match.match_score}%</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card className="border-dashed border-border/80">
          <CardContent className="py-16 text-center">
            <Sparkles className="h-10 w-10 text-primary mx-auto mb-4" />
            <p className="text-sm font-semibold text-foreground">No visible matches</p>
            <p className="text-[12px] text-muted-foreground mt-1">
              Adjust your filters or run the ranking again to populate donor cards.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Matching;
