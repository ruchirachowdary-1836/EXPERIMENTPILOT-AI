import { useQuery } from "@tanstack/react-query";
import { BookOpen, FlaskConical, FolderKanban, Menu, Moon, Settings, Sun, X } from "lucide-react";
import { useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "@/api/client";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { Logo } from "./Logo";

const NAV = [
  { to: "/app/projects", label: "Workspace", icon: FolderKanban },
  { to: "/app/benchmark", label: "Benchmark lab", icon: FlaskConical },
  { to: "/app/docs", label: "Documentation", icon: BookOpen },
  { to: "/app/settings", label: "Settings", icon: Settings },
];

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      className="grid h-9 w-9 place-items-center rounded-xl border border-line/15 text-muted transition hover:border-line/30 hover:text-fg">
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

function ApiStatus() {
  const q = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000, retry: false });
  const ok = q.isSuccess;
  return (
    <div className="flex items-center gap-2 text-xs text-muted" title={ok ? `API v${q.data?.version}` : "API unreachable"}>
      <span className={cn("h-2 w-2 rounded-full", ok ? "bg-good shadow-[0_0_8px] shadow-good" : q.isLoading ? "bg-warn" : "bg-bad")} />
      {ok ? "API online" : q.isLoading ? "Connecting…" : "API offline"}
    </div>
  );
}

export function AppShell() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const nav = (
    <nav className="flex flex-col gap-1" aria-label="Primary">
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink key={to} to={to} onClick={() => setOpen(false)}
          className={({ isActive }) => cn("group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition",
            isActive || (to === "/app/projects" && loc.pathname.startsWith("/app/sessions")) ? "bg-accent/12 text-fg ring-1 ring-accent/25" : "text-muted hover:bg-line/10 hover:text-fg")}>
          <Icon className="h-4 w-4 text-faint transition group-hover:text-accent" />{label}
        </NavLink>
      ))}
    </nav>
  );
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_1fr]">
      <aside className="sticky top-0 hidden h-screen flex-col justify-between border-r border-line/10 bg-surface/60 p-4 backdrop-blur lg:flex">
        <div className="space-y-8">
          <Link to="/" className="block px-2 pt-2"><Logo /></Link>
          {nav}
        </div>
        <div className="space-y-3 rounded-xl border border-line/10 bg-sunken/60 p-3">
          <ApiStatus />
          <p className="text-[11px] leading-relaxed text-faint">Local-first. Uploaded data never leaves this machine.</p>
        </div>
      </aside>
      <div className="min-w-0">
        <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-line/10 bg-bg/80 px-4 backdrop-blur-xl sm:px-6 lg:justify-end">
          <button className="lg:hidden rounded-lg p-2 text-muted hover:bg-line/10" onClick={() => setOpen(true)} aria-label="Open navigation"><Menu className="h-5 w-5" /></button>
          <Link to="/" className="lg:hidden"><Logo compact /></Link>
          <div className="flex items-center gap-3"><div className="hidden sm:block"><ApiStatus /></div><ThemeToggle /></div>
        </header>
        {open && (
          <div className="fixed inset-0 z-40 lg:hidden">
            <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
            <div className="absolute left-0 top-0 h-full w-72 bg-surface p-4 shadow-2xl">
              <div className="mb-6 flex items-center justify-between"><Logo /><button onClick={() => setOpen(false)} aria-label="Close navigation" className="p-1.5 text-muted"><X className="h-5 w-5" /></button></div>
              {nav}
            </div>
          </div>
        )}
        <main className="mx-auto w-full max-w-[1280px] px-4 py-6 sm:px-6 sm:py-8"><Outlet /></main>
      </div>
    </div>
  );
}
