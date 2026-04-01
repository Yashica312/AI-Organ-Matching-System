import { createContext, useContext, useState, ReactNode, useCallback } from "react";
import { Donor, Recipient, donors as initialDonors, recipients as initialRecipients } from "@/data/mockData";

interface DataContextType {
  donors: Donor[];
  recipients: Recipient[];
  addDonor: (donor: Omit<Donor, "id">) => void;
  addRecipient: (recipient: Omit<Recipient, "id">) => void;
}

const DataContext = createContext<DataContextType | null>(null);

export function DataProvider({ children }: { children: ReactNode }) {
  const [donors, setDonors] = useState<Donor[]>(initialDonors);
  const [recipients, setRecipients] = useState<Recipient[]>(initialRecipients);

  const addDonor = useCallback((donor: Omit<Donor, "id">) => {
    setDonors((prev) => [...prev, { ...donor, id: Date.now() }]);
  }, []);

  const addRecipient = useCallback((recipient: Omit<Recipient, "id">) => {
    setRecipients((prev) => [...prev, { ...recipient, id: Date.now() }]);
  }, []);

  return (
    <DataContext.Provider value={{ donors, recipients, addDonor, addRecipient }}>
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error("useData must be used within DataProvider");
  return ctx;
}
