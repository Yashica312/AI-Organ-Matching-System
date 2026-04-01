import { useState } from "react";
import { Heart, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useData } from "@/context/DataContext";

const bloodGroups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];
const organs = ["Kidney", "Liver", "Heart", "Lungs", "Pancreas"];

const AddDonor = () => {
  const { toast } = useToast();
  const { addDonor } = useData();
  const [form, setForm] = useState({
    name: "",
    age: "",
    bloodGroup: "",
    organ: "",
    healthScore: [80],
    distance: [50],
    location: "",
  });
  const [errors, setErrors] = useState<Record<string, boolean>>({});

  const validate = () => {
    const e: Record<string, boolean> = {};
    if (!form.name.trim()) e.name = true;
    if (!form.age || Number(form.age) < 1 || Number(form.age) > 100) e.age = true;
    if (!form.bloodGroup) e.bloodGroup = true;
    if (!form.organ) e.organ = true;
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) {
      toast({ title: "Validation Error", description: "Please fill in all required fields correctly.", variant: "destructive" });
      return;
    }
    addDonor({
      name: form.name,
      age: Number(form.age),
      bloodGroup: form.bloodGroup,
      organ: form.organ,
      hlaMatch: form.healthScore[0],
      location: form.location || "Unknown",
      waitTime: 0,
      score: Math.round(form.healthScore[0] * 0.6 + (100 - form.distance[0] / 5) * 0.4),
    });
    toast({
      title: "Donor Registered",
      description: `${form.name} has been added. Go to Matching to see updated results.`,
    });
    setForm({ name: "", age: "", bloodGroup: "", organ: "", healthScore: [80], distance: [50], location: "" });
    setErrors({});
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="rounded-xl border border-border bg-card overflow-hidden" style={{ boxShadow: "var(--shadow-elevated)" }}>
        <div className="flex items-center gap-3 p-5 border-b border-border bg-gradient-to-r from-primary/5 to-transparent">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary-glow shadow-sm">
            <Heart className="h-4.5 w-4.5 text-primary-foreground" />
          </div>
          <div>
            <h2 className="text-[15px] font-bold text-foreground">Register New Donor</h2>
            <p className="text-[11px] text-muted-foreground">Enter donor medical details below</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="p-5 space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="name" className="text-[12px] font-semibold">Full Name <span className="text-destructive">*</span></Label>
              <Input id="name" placeholder="Enter full name" value={form.name}
                onChange={(e) => { setForm({ ...form, name: e.target.value }); setErrors({ ...errors, name: false }); }}
                className={`rounded-lg h-10 text-[13px] ${errors.name ? "border-destructive ring-1 ring-destructive/20" : ""}`} />
              {errors.name && <p className="text-[11px] text-destructive">Name is required</p>}
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="age" className="text-[12px] font-semibold">Age <span className="text-destructive">*</span></Label>
              <Input id="age" type="number" placeholder="e.g. 30" min={1} max={100} value={form.age}
                onChange={(e) => { setForm({ ...form, age: e.target.value }); setErrors({ ...errors, age: false }); }}
                className={`rounded-lg h-10 text-[13px] ${errors.age ? "border-destructive ring-1 ring-destructive/20" : ""}`} />
              {errors.age && <p className="text-[11px] text-destructive">Valid age required (1-100)</p>}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label className="text-[12px] font-semibold">Blood Group <span className="text-destructive">*</span></Label>
              <Select value={form.bloodGroup} onValueChange={(v) => { setForm({ ...form, bloodGroup: v }); setErrors({ ...errors, bloodGroup: false }); }}>
                <SelectTrigger className={`rounded-lg h-10 text-[13px] ${errors.bloodGroup ? "border-destructive ring-1 ring-destructive/20" : ""}`}>
                  <SelectValue placeholder="Select blood group" />
                </SelectTrigger>
                <SelectContent>{bloodGroups.map((bg) => <SelectItem key={bg} value={bg}>{bg}</SelectItem>)}</SelectContent>
              </Select>
              {errors.bloodGroup && <p className="text-[11px] text-destructive">Blood group is required</p>}
            </div>
            <div className="space-y-1.5">
              <Label className="text-[12px] font-semibold">Organ <span className="text-destructive">*</span></Label>
              <Select value={form.organ} onValueChange={(v) => { setForm({ ...form, organ: v }); setErrors({ ...errors, organ: false }); }}>
                <SelectTrigger className={`rounded-lg h-10 text-[13px] ${errors.organ ? "border-destructive ring-1 ring-destructive/20" : ""}`}>
                  <SelectValue placeholder="Select organ" />
                </SelectTrigger>
                <SelectContent>{organs.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}</SelectContent>
              </Select>
              {errors.organ && <p className="text-[11px] text-destructive">Organ is required</p>}
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="location" className="text-[12px] font-semibold">Location</Label>
            <Input id="location" placeholder="e.g. Mumbai" value={form.location}
              onChange={(e) => setForm({ ...form, location: e.target.value })} className="rounded-lg h-10 text-[13px]" />
          </div>

          <div className="space-y-2.5 p-4 rounded-xl bg-muted/40 border border-border/50">
            <div className="flex items-center justify-between">
              <Label className="text-[12px] font-semibold">Health Score</Label>
              <span className="text-[13px] font-bold text-primary font-mono">{form.healthScore[0]}%</span>
            </div>
            <Slider value={form.healthScore} onValueChange={(v) => setForm({ ...form, healthScore: v })} max={100} min={0} step={1} />
            <p className="text-[11px] text-muted-foreground">Overall health assessment score</p>
          </div>

          <div className="space-y-2.5 p-4 rounded-xl bg-muted/40 border border-border/50">
            <div className="flex items-center justify-between">
              <Label className="text-[12px] font-semibold">Distance</Label>
              <span className="text-[13px] font-bold text-primary font-mono">{form.distance[0]} km</span>
            </div>
            <Slider value={form.distance} onValueChange={(v) => setForm({ ...form, distance: v })} max={500} min={0} step={5} />
            <p className="text-[11px] text-muted-foreground">Distance from medical center</p>
          </div>

          <Button type="submit" className="w-full rounded-lg h-11 text-[13px] font-bold bg-gradient-to-r from-primary to-primary-glow hover:shadow-[var(--shadow-glow)] transition-all duration-300">
            <CheckCircle2 className="h-4 w-4 mr-2" />
            Register Donor
          </Button>
        </form>
      </div>
    </div>
  );
};

export default AddDonor;
