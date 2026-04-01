import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Outlet, useLocation } from "react-router-dom";


const pageTitles: Record<string, { title: string; subtitle: string }> = {
  "/matching": { title: "AI Organ Matching System", subtitle: "Multi-donor ranking with explainable AI insights" },
  "/add-donor": { title: "Donors", subtitle: "Register and manage organ donor entries" },
  "/add-recipient": { title: "Recipients", subtitle: "Register and manage transplant recipients" },
  "/dashboard": { title: "AI Organ Matching System", subtitle: "Multi-donor ranking with explainable AI insights" },
};


export function DashboardLayout() {
  const location = useLocation();
  const page = pageTitles[location.pathname] || { title: "AI Organ Matching System", subtitle: "" };

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        <AppSidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <header className="h-14 flex items-center border-b border-border px-5 bg-card/80 backdrop-blur-sm shrink-0 sticky top-0 z-10">
            <SidebarTrigger className="text-muted-foreground hover:text-foreground transition-colors mr-3" />
            <div className="h-5 w-px bg-border mr-3 hidden sm:block" />
            <div className="hidden sm:block">
              <h1 className="text-sm font-semibold text-foreground leading-tight">{page.title}</h1>
              <p className="text-[11px] text-muted-foreground">{page.subtitle}</p>
            </div>
          </header>
          <main className="flex-1 p-5 bg-background overflow-auto">
            <div className="animate-in-page">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}
