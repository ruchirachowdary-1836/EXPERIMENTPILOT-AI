import { Handle, Position, type NodeProps } from "@xyflow/react";
import { CircleDot, FlaskConical, Flag, Hourglass } from "lucide-react";
import { cn, hypColor, pct, title } from "@/lib/utils";

const shell = "w-[250px] rounded-2xl border bg-surface px-3.5 py-3 shadow-card text-left";
const handles = (<><Handle type="target" position={Position.Left} className="!h-2 !w-2 !border-0 !bg-faint" /><Handle type="source" position={Position.Right} className="!h-2 !w-2 !border-0 !bg-faint" /></>);

function Top({ rows }: { rows?: { hypothesis: string; probability: number }[] }) {
  return (
    <div className="mt-2 space-y-1">
      {(rows ?? []).slice(0, 3).map((r) => (
        <div key={r.hypothesis} className="flex items-center gap-2 text-[11px]">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: hypColor(r.hypothesis) }} />
          <span className="truncate text-muted">{title(r.hypothesis)}</span>
          <span className="num ml-auto">{pct(r.probability)}</span>
        </div>))}
    </div>
  );
}

export function RootNode({ data }: NodeProps) {
  const d = data as any;
  return <div className={cn(shell, "border-accent/50 shadow-glow")} data-testid="tree-node-root">{handles}<div className="flex items-center gap-2 text-xs font-semibold"><CircleDot className="h-3.5 w-3.5 text-accent" />{d.title}</div><div className="num mt-1 text-[11px] text-faint">uncertainty {Number(d.entropy).toFixed(2)} bits</div><Top rows={d.top} /></div>;
}
export function ExperimentNode({ data }: NodeProps) {
  const d = data as any;
  return (
    <div className={cn(shell, "border-accent/40")} data-testid="tree-node-experiment">{handles}
      <div className="flex items-center gap-2 text-[13px] font-semibold"><FlaskConical className="h-3.5 w-3.5 shrink-0 text-accent" /><span className="truncate">{d.title}</span></div>
      <div className="num mt-1.5 flex gap-3 text-[11px] text-muted"><span>EIG {Number(d.eig).toFixed(2)}b</span><span>cost {d.cost}</span><span>risk {d.risk}</span></div>
      <div className="mt-1 text-[10px] uppercase tracking-wider text-faint">{d.source === "external" ? "entered manually" : d.chosen_by === "user" ? "chosen by you" : "planner pick"}</div>
    </div>);
}
export function OutcomeNode({ data }: NodeProps) {
  const d = data as any; const taken = d.taken;
  return (
    <div className={cn(shell, taken ? "border-accent bg-accent/10" : "border-dashed border-line/25 opacity-75")} data-testid={taken ? "tree-outcome-taken" : "tree-outcome-hypothetical"}>{handles}
      <div className="flex items-center justify-between gap-2"><span className="text-[13px] font-semibold capitalize">{d.title}</span><span className={cn("rounded-full px-2 py-0.5 text-[10px] font-medium", taken ? "bg-accent text-accent-fg" : "bg-line/15 text-muted")}>{taken ? "observed" : "not observed"}</span></div>
      <div className="num mt-1 text-[11px] text-faint">P(outcome) {pct(d.probability, 0)} · {Number(d.entropy).toFixed(2)} bits left</div>
      <Top rows={d.top} />
      {!taken && d.hypothetical_next && <div className="mt-2 border-t border-line/10 pt-1.5 text-[11px] text-muted">would then run: <span className="text-fg">{d.hypothetical_next}</span></div>}
    </div>);
}
export function StopNode({ data }: NodeProps) {
  const d = data as any;
  return <div className={cn(shell, "border-good/50 bg-good/8")} data-testid="tree-node-stop">{handles}<div className="flex items-center gap-2 text-[13px] font-semibold"><Flag className="h-3.5 w-3.5 text-good" />{d.title}</div><p className="mt-1 text-[11px] leading-relaxed text-muted">{d.message}</p><div className="num mt-1 text-[11px] text-faint">cost spent {Number(d.spent).toFixed(1)}</div></div>;
}
export function PendingNode({ data }: NodeProps) {
  const d = data as any;
  return <div className={cn(shell, "border-dashed border-warn/50")} data-testid="tree-node-pending">{handles}<div className="flex items-center gap-2 text-[13px] font-semibold"><Hourglass className="h-3.5 w-3.5 text-warn" />Next: {d.title}</div><p className="mt-1 text-[11px] text-muted">{d.message}</p>{d.eig != null && <div className="num mt-1 text-[11px] text-faint">EIG {Number(d.eig).toFixed(2)}b · cost {d.cost}</div>}</div>;
}

export const nodeTypes = { root: RootNode, experiment: ExperimentNode, outcome: OutcomeNode, stop: StopNode, pending: PendingNode };
