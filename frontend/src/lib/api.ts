const BASE_URL = import.meta.env.VITE_API_URL || "";

export const api = {
  login: async (username: string, password: string) => {
    const res = await fetch(`${BASE_URL}/api/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) throw new Error("Invalid credentials");
    return res.json();
  },

  register: async (username: string, password: string) => {
    const res = await fetch(`${BASE_URL}/api/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) throw new Error("Registration failed");
    return res.json();
  },

  getDonors: async () => {
    const res = await fetch(`${BASE_URL}/api/donors`);
    if (!res.ok) throw new Error("Failed to fetch donors");
    return res.json();
  },

  addDonor: async (donor: {
    name: string;
    age: number;
    blood_group: string;
    organ: string;
    health_score: number;
    distance: number;
  }) => {
    const res = await fetch(`${BASE_URL}/api/donors`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(donor),
    });
    if (!res.ok) throw new Error("Failed to add donor");
    return res.json();
  },

  getRecipients: async () => {
    const res = await fetch(`${BASE_URL}/api/recipients`);
    if (!res.ok) throw new Error("Failed to fetch recipients");
    return res.json();
  },

  addRecipient: async (recipient: {
    name: string;
    age: number;
    blood_group: string;
    organ: string;
    urgency_score: number;
  }) => {
    const res = await fetch(`${BASE_URL}/api/recipients`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(recipient),
    });
    if (!res.ok) throw new Error("Failed to add recipient");
    return res.json();
  },

  getMatch: async (recipient_id: number) => {
    const res = await fetch(`${BASE_URL}/api/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recipient_id }),
    });
    if (!res.ok) throw new Error("Matching failed");
    return res.json();
  },

  getInsights: async () => {
    const res = await fetch(`${BASE_URL}/api/model/insights`);
    if (!res.ok) throw new Error("Failed to fetch insights");
    return res.json();
  },
};
