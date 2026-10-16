import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import type { HistoryPoint, Marginal } from "@/types/api";
import { PLANNER_COLORS, hypColor, humanize, pct, title } from "@/lib/utils";

const axis = { stroke: "rgb(var(--faint))", fontSize: 11, tickLine: false, axisLine: false } as const;
const grid = <CartesianGrid stroke="rgb(var(--line) / 0.08)" vertical={false} />;

function Tip({ active, payload, label, fmt = (v: number) => v.toFixed(3) }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card bg-surface px-3 py-2 text-xs shadow-xl">
      {label != null && <div className="mb-1 font-medium">{label}</div>}
      {payload.filter((p: any) => p.value != null).map((p: any) => (
        <div key={p.dataKey ?? p.name} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-muted">{p.name}</span><span className="num ml-auto pl-3">{typeof p.value === "number" ? fmt(p.value) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

/** Posterior marginal per root cause with prior markers. */
export function BeliefBars({ marginals, priors, height = 300 }: { marginals: Marginal[]; priors?: Record<string, number>; height?: number }) {
  const data = marginals.map((m) => ({ id: m.hypothesis, name: `${m.code} ${m.name}`.slice(0, 40), p: m.probability, prior: priors?.[m.hypothesis] }));
  return (
    <div role="img" aria-label="Root-cause probabilities" style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 36, top: 4, bottom: 4 }} barCategoryGap={10}>
          <CartesianGrid stroke="rgb(var(--line) / 0.08)" horizontal={false} />
          <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} />
          <YAxis type="category" dataKey="name" width={188} {...axis} tick={{ fill: "rgb(var(--muted))", fontSize: 12 }} />
          <Tooltip cursor={{ fill: "rgb(var(--line) / 0.06)" }} content={<Tip fmt={(v: number) => pct(v, 1)} />} />
          <Bar dataKey="p" name="Posterior" radius={[0, 8, 8, 0]} isAnimationActive animationDuration={700} label={{ position: "right", fill: "rgb(var(--fg))", fontSize: 12, formatter: (v: number) => pct(v, 0) }}>
            {data.map((d) => <Cell key={d.id} fill={hypColor(d.id)} />)}
          </Bar>
          {priors && <Bar dataKey="prior" name="Prior" fill="rgb(var(--faint))" fillOpacity={0.5} radius={[0, 4, 4, 0]} barSize={3} />}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BeliefEvolution({ history, hypotheses, height = 280 }: { history: HistoryPoint[]; hypotheses: string[]; height?: number }) {
  const data = history.map((h) => ({ step: h.step, label: h.label, ...h.marginals }));
  return (
    <div role="img" aria-label="Belief evolution over diagnostic steps" style={{ height }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ left: -8, right: 12, top: 8 }}>
          {grid}
          <XAxis dataKey="step" {...axis} label={{ value: "experiment step", position: "insideBottom", offset: -2, fill: "rgb(var(--faint))", fontSize: 11 }} height={36} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} />
          <Tooltip content={<Tip fmt={(v: number) => pct(v, 1)} />} />
          {hypotheses.map((h) => <Line key={h} type="monotone" dataKey={h} name={title(h)} stroke={hypColor(h)} strokeWidth={2.2} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive animationDuration={600} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function EntropyArea({ history, height = 160 }: { history: HistoryPoint[]; height?: number }) {
  return (
    <div style={{ height }} role="img" aria-label="Uncertainty in bits over steps">
      <ResponsiveContainer>
        <AreaChart data={history} margin={{ left: -12, right: 8, top: 8 }}>
          <defs><linearGradient id="ent" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="rgb(79,140,255)" stopOpacity={0.45} /><stop offset="1" stopColor="rgb(79,140,255)" stopOpacity={0} /></linearGradient></defs>
          {grid}<XAxis dataKey="step" {...axis} /><YAxis {...axis} />
          <Tooltip content={<Tip fmt={(v: number) => `${v.toFixed(2)} bits`} />} />
          <Area type="monotone" dataKey="entropy" name="Entropy" stroke="rgb(79,140,255)" strokeWidth={2} fill="url(#ent)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ evidence charts */
export function HBars({ data, x, y, color = "#4f8cff", height = 260, fmt = (v: number) => v.toFixed(2), refLines = [], name }:
  { data: Record<string, any>[]; x: string; y: string; color?: string; height?: number; fmt?: (v: number) => string; refLines?: { value: number; label: string }[]; name?: string }) {
  return (
    <div style={{ height }} role="img" aria-label={name ?? `${y} by ${x}`}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 28, top: 4 }}>
          <CartesianGrid stroke="rgb(var(--line) / 0.08)" horizontal={false} />
          <XAxis type="number" {...axis} tickFormatter={(v) => fmt(v)} />
          <YAxis type="category" dataKey={x} width={130} {...axis} tick={{ fill: "rgb(var(--muted))", fontSize: 11 }} />
          <Tooltip cursor={{ fill: "rgb(var(--line) / 0.06)" }} content={<Tip fmt={fmt} />} />
          {refLines.map((r) => <ReferenceLine key={r.label} x={r.value} stroke="rgb(var(--warn))" strokeDasharray="4 4" label={{ value: r.label, fill: "rgb(var(--warn))", fontSize: 10, position: "top" }} />)}
          <Bar dataKey={y} name={name ?? y} fill={color} radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function GroupedBars({ data, x, series, height = 240, fmt = (v: number) => v.toFixed(2) }:
  { data: Record<string, any>[]; x: string; series: { key: string; name: string; color: string }[]; height?: number; fmt?: (v: number) => string }) {
  return (
    <div style={{ height }} role="img" aria-label="Grouped comparison">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: -8, right: 8, top: 8 }}>
          {grid}<XAxis dataKey={x} {...axis} interval={0} tick={{ fill: "rgb(var(--muted))", fontSize: 11 }} /><YAxis {...axis} tickFormatter={fmt} />
          <Tooltip cursor={{ fill: "rgb(var(--line) / 0.06)" }} content={<Tip fmt={fmt} />} /><Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s) => <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color} radius={[6, 6, 0, 0]} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ benchmark charts */
export function BudgetCurves({ rows, planners, labels, height = 340, metric = "top1" }:
  { rows: { planner: string; budget: number; top1: number; top1_lo: number; top1_hi: number; f1: number }[]; planners: string[]; labels: Record<string, string>; height?: number; metric?: "top1" | "f1" }) {
  const budgets = [...new Set(rows.map((r) => r.budget))].sort((a, b) => a - b);
  const data = budgets.map((b) => {
    const o: Record<string, number> = { budget: b };
    planners.forEach((p) => { const r = rows.find((x) => x.planner === p && x.budget === b); if (r) o[p] = r[metric]; });
    return o;
  });
  return (
    <div style={{ height }} role="img" aria-label="Accuracy at fixed diagnostic budget">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ left: -4, right: 16, top: 8 }}>
          {grid}<XAxis dataKey="budget" type="number" domain={["dataMin", "dataMax"]} {...axis} label={{ value: "budget (cost units)", position: "insideBottom", offset: -2, fill: "rgb(var(--faint))", fontSize: 11 }} height={36} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} />
          <Tooltip content={<Tip fmt={(v: number) => pct(v, 1)} />} /><Legend wrapperStyle={{ fontSize: 12 }} />
          {planners.map((p) => <Line key={p} type="monotone" dataKey={p} name={labels[p] ?? p} stroke={PLANNER_COLORS[p] ?? "#888"} strokeWidth={p === "adaptive" ? 3.2 : 1.8} dot={{ r: p === "adaptive" ? 4 : 2.5 }} isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CostBars({ rows, height = 300 }: { rows: { label: string; planner: string; cost: number; cost_lo: number; cost_hi: number; top1: number }[]; height?: number }) {
  const data = rows.map((r) => ({ ...r, err: [r.cost - r.cost_lo, r.cost_hi - r.cost] }));
  return (
    <div style={{ height }} role="img" aria-label="Mean diagnostic cost by planner">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: -8, right: 8, top: 8, bottom: 30 }}>
          {grid}<XAxis dataKey="label" {...axis} interval={0} angle={-20} textAnchor="end" tick={{ fill: "rgb(var(--muted))", fontSize: 11 }} height={60} /><YAxis {...axis} />
          <Tooltip cursor={{ fill: "rgb(var(--line) / 0.06)" }} content={<Tip fmt={(v: number) => v.toFixed(2)} />} />
          <Bar dataKey="cost" name="Mean cost" radius={[8, 8, 0, 0]}>
            {data.map((d) => <Cell key={d.planner} fill={PLANNER_COLORS[d.planner] ?? "#888"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Reliability({ points, height = 280 }: { points: { confidence: number; accuracy: number; count: number }[]; height?: number }) {
  return (
    <div style={{ height }} role="img" aria-label="Reliability diagram">
      <ResponsiveContainer>
        <ComposedChart data={points} margin={{ left: -8, right: 12, top: 8 }}>
          {grid}<XAxis dataKey="confidence" type="number" domain={[0, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} /><YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="rgb(var(--faint))" strokeDasharray="4 4" />
          <Tooltip content={<Tip fmt={(v: number) => v.toFixed(3)} />} />
          <Line dataKey="accuracy" name="Observed frequency" stroke="rgb(79,140,255)" strokeWidth={2.4} dot={{ r: 4 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RobustnessChart({ rows, labels, height = 260 }: { rows: { sigma: number; planner: string; f1: number }[]; labels: Record<string, string>; height?: number }) {
  const sig = [...new Set(rows.map((r) => r.sigma))].sort((a, b) => a - b);
  const planners = [...new Set(rows.map((r) => r.planner))];
  const data = sig.map((s) => { const o: Record<string, number> = { sigma: s }; planners.forEach((p) => { const r = rows.find((x) => x.sigma === s && x.planner === p); if (r) o[p] = r.f1; }); return o; });
  return (
    <div style={{ height }} role="img" aria-label="Robustness to misspecified likelihoods">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ left: -8, right: 12, top: 8 }}>
          {grid}<XAxis dataKey="sigma" {...axis} /><YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip content={<Tip fmt={(v: number) => v.toFixed(3)} />} /><Legend wrapperStyle={{ fontSize: 12 }} />
          {planners.map((p) => <Line key={p} dataKey={p} name={labels[p] ?? p} stroke={PLANNER_COLORS[p] ?? "#888"} strokeWidth={2.2} dot={{ r: 3 }} isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
