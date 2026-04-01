import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Login from "./pages/Login";
import Matching from "./pages/Matching";
import AddDonor from "./pages/AddDonor";
import AddRecipient from "./pages/AddRecipient";
import NotFound from "./pages/NotFound";
import { DashboardLayout } from "./components/DashboardLayout";
import { DataProvider } from "./context/DataContext";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <DataProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Login />} />
            <Route element={<DashboardLayout />}>
              <Route path="/matching" element={<Matching />} />
              <Route path="/add-donor" element={<AddDonor />} />
              <Route path="/add-recipient" element={<AddRecipient />} />
              <Route path="/dashboard" element={<Matching />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </DataProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
