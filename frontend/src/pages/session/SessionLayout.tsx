import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, FileText, GitBranch, Microscope } from "lucide-react";
import { Link, NavLink, Outlet, useOutletContext, useParams } from "react-router-dom";
import { api } from "@/api/client";
import { Badge, ErrorState, Skeleton } from "@/components/ui";
import { cn, shortDate, statusTone } from "@/lib/utils";
import type { Session } from "@/types/api";

export interface SessionCtx { session: Session; setSession: (s: Session) => void; refetch: () => void }
export const useSessionCtx = () => useOutletContext<SessionCtx>();

const TABS = [
  { to: ".", end: true, label: "Live diagnosis", icon: Activity },
  { to: "tree", label: "Decision tree", icon: GitBranch },
  { to: "evidence", label: "Evidence", icon: Microscope },
  { to: "report", label: "Report", icon: FileText },
];

export default function SessionLayout() {
  const { id = "" } = useParams(); const qc = useQueryClient();
  const q = useQuery({ queryKey: ["session", id], queryFn: () => api.session(id) });
  if (q.isLoading) return <div className="space-y-4"><Skeleton className="h-12 w-80" /><Skeleton className="h-96" /></div>;
  if (q.isError || !q.data) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const s = q.data;
  const setSession = (n: Session) => {
    qc.setQueryData(["session", id], n);
    for (const k of ["history", "tree", "evidence", "report"]) qc.invalidateQueries({ queryKey: [k, id] });
  };
  return (
    <div>
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="eyebrow mb-1"><Link to={`/app/projects/${s.project_id}`} className="hover:text-fg">Project</Link> / Session</div>
          <h1 className="h-page flex flex-wrap items-center gap-3">{s.name}<Badge tone={statusTone(s.status)} dot={s.status === "active"}>{s.status}</Badge></h1>
          <p className="mt-1 text-sm text-muted">Started {shortDate(s.created_at)} · planner <span className="font-medium text-fg">{s.recommendation?.planner_label ?? s.planner}</span></p>
        </div>
      </div>
      <nav className="mb-6 flex gap-1 overflow-x-auto border-b border-line/10" aria-label="Session sections">
        {TABS.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => cn("relative flex items-center gap-2 whitespace-nowrap px-4 py-2.5 text-sm font-medium transition", isActive ? "text-fg" : "text-muted hover:text-fg")}>
            {({ isActive }) => <><Icon className="h-4 w-4" />{label}{isActive && <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-accent" />}</>}
          </NavLink>))}
      </nav>
      <Outlet context={{ session: s, setSession, refetch: () => q.refetch() } satisfies SessionCtx} />
    </div>
  );
}
