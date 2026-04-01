export interface Donor {
  id: number;
  name: string;
  age: number;
  bloodGroup: string;
  organ: string;
  hlaMatch: number;
  location: string;
  waitTime: number; // days
  score: number;
}

export interface Recipient {
  id: number;
  name: string;
  age: number;
  bloodGroup: string;
  organ: string;
  urgency: "Critical" | "High" | "Medium" | "Low";
  hlaType: string;
  location: string;
  waitTime: number;
}

export const recipients: Recipient[] = [
  { id: 1, name: "Ananya Sharma", age: 34, bloodGroup: "O+", organ: "Kidney", urgency: "Critical", hlaType: "A2, B7, DR4", location: "Mumbai", waitTime: 540 },
  { id: 2, name: "Ravi Kumar", age: 52, bloodGroup: "A+", organ: "Liver", urgency: "High", hlaType: "A1, B8, DR3", location: "Delhi", waitTime: 320 },
  { id: 3, name: "Priya Patel", age: 28, bloodGroup: "B+", organ: "Kidney", urgency: "Medium", hlaType: "A3, B35, DR11", location: "Bangalore", waitTime: 180 },
  { id: 4, name: "Suresh Reddy", age: 61, bloodGroup: "AB+", organ: "Heart", urgency: "Critical", hlaType: "A24, B44, DR7", location: "Hyderabad", waitTime: 410 },
  { id: 5, name: "Meera Joshi", age: 45, bloodGroup: "O-", organ: "Liver", urgency: "High", hlaType: "A2, B27, DR1", location: "Pune", waitTime: 265 },
];

export const donors: Donor[] = [
  { id: 1, name: "Vikram Singh", age: 30, bloodGroup: "O+", organ: "Kidney", hlaMatch: 92, location: "Mumbai", waitTime: 0, score: 95 },
  { id: 2, name: "Neha Gupta", age: 26, bloodGroup: "O+", organ: "Kidney", hlaMatch: 85, location: "Pune", waitTime: 0, score: 88 },
  { id: 3, name: "Arjun Nair", age: 35, bloodGroup: "O-", organ: "Kidney", hlaMatch: 78, location: "Chennai", waitTime: 0, score: 81 },
  { id: 4, name: "Kavita Desai", age: 29, bloodGroup: "A+", organ: "Liver", hlaMatch: 90, location: "Delhi", waitTime: 0, score: 93 },
  { id: 5, name: "Rajesh Iyer", age: 40, bloodGroup: "A+", organ: "Liver", hlaMatch: 82, location: "Mumbai", waitTime: 0, score: 84 },
  { id: 6, name: "Deepa Menon", age: 33, bloodGroup: "B+", organ: "Kidney", hlaMatch: 88, location: "Bangalore", waitTime: 0, score: 90 },
  { id: 7, name: "Amit Verma", age: 38, bloodGroup: "AB+", organ: "Heart", hlaMatch: 94, location: "Hyderabad", waitTime: 0, score: 96 },
  { id: 8, name: "Sunita Rao", age: 31, bloodGroup: "O-", organ: "Liver", hlaMatch: 86, location: "Pune", waitTime: 0, score: 89 },
  { id: 9, name: "Manoj Pillai", age: 44, bloodGroup: "AB+", organ: "Heart", hlaMatch: 76, location: "Chennai", waitTime: 0, score: 79 },
  { id: 10, name: "Lakshmi Bhat", age: 27, bloodGroup: "B+", organ: "Kidney", hlaMatch: 80, location: "Bangalore", waitTime: 0, score: 83 },
];

export const matchExplanations: Record<number, string> = {
  1: "Vikram Singh is the top match due to 92% HLA compatibility, same blood group (O+), geographic proximity (Mumbai), and young age ensuring organ viability.",
  4: "Kavita Desai ranks highest for liver transplant with 90% HLA match, compatible A+ blood group, and location proximity to the recipient in Delhi.",
  7: "Amit Verma is the optimal heart donor with 94% HLA compatibility, matching AB+ blood group, and being located in the same city (Hyderabad).",
};

export const stats = {
  totalDonors: 10,
  totalRecipients: 5,
  matchesCompleted: 3,
  pendingMatches: 2,
  avgMatchScore: 87.8,
  criticalCases: 2,
};
