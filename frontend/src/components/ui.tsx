import { AlertTriangle, CheckCircle2, Info, Loader2, X, XCircle } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useId, useRef, useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ Button */
type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger" | "outline";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
};
export function Button({ variant = "secondary", size = "md", loading, icon, className, children, disabled, ...rest }: BtnProps) {
  const base = "inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-150 select-none disabled:opacity-50 disabled:pointer-events-none active:scale-[.98] whitespace-nowrap";
  const sizes = { sm: "h-8 px-3 text-[13px]", md: "h-10 px-4 text-sm", lg: "h-12 px-6 text-[15px]" }[size];
  const variants = {
    primary: "bg-accent text-accent-fg shadow-[0_6px_20px_-6px_rgb(var(--accent)/.7)] hover:brightness-110",
    secondary: "bg-raised text-fg border border-line/15 hover:border-line/30 hover:bg-raised/70",
    outline: "border border-accent/50 text-accent hover:bg-accent/10",
    ghost: "text-muted hover:text-fg hover:bg-line/10",
    danger: "bg-bad/15 text-bad border border-bad/30 hover:bg-bad/25",
  }[variant];
  return (
    <button className={cn(base, sizes, variants, className)} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
      {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ Card & layout */
export function Card({ className, children, hover, ...rest }: { className?: string; children: ReactNode; hover?: boolean } & React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card", hover && "card-hover", className)} {...rest}>{children}</div>;
}
export function CardHeader({ title, subtitle, action, icon }: { title: ReactNode; subtitle?: ReactNode; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 px-5 pt-5 sm:px-6 sm:pt-6">
      <div className="flex items-start gap-3 min-w-0">
        {icon && <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-accent/12 text-accent">{icon}</div>}
        <div className="min-w-0">
          <h3 className="text-[15px] leading-tight">{title}</h3>
          {subtitle && <p className="mt-1 text-[13px] text-muted">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  );
}
export function PageHeader({ title, subtitle, actions, eyebrow }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
        <h1 className="h-page">{title}</h1>
        {subtitle && <p className="mt-1.5 max-w-3xl text-sm text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ Badge */
export function Badge({ tone = "muted", children, dot, className }: { tone?: "good" | "warn" | "bad" | "accent" | "muted" | "cyan"; children: ReactNode; dot?: boolean; className?: string }) {
  const tones = {
    good: "bg-good/12 text-good border-good/25", warn: "bg-warn/12 text-warn border-warn/25", bad: "bg-bad/12 text-bad border-bad/25",
    accent: "bg-accent/12 text-accent border-accent/25", muted: "bg-line/10 text-muted border-line/15", cyan: "bg-cyan/12 text-cyan border-cyan/25",
  }[tone];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium", tones, className)}>
      {dot && <span className="relative flex h-1.5 w-1.5"><span className="absolute inline-flex h-full w-full rounded-full bg-current opacity-60" /><span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" /></span>}
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------ Stat */
export function Stat({ label, value, hint, icon, tone }: { label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode; tone?: "good" | "warn" | "bad" | "accent" }) {
  const color = tone ? { good: "text-good", warn: "text-warn", bad: "text-bad", accent: "text-accent" }[tone] : "text-fg";
  return (
    <Card className="p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <div className="eyebrow">{label}</div>
        {icon && <div className="text-faint">{icon}</div>}
      </div>
      <div className={cn("num mt-2 text-2xl font-medium sm:text-[28px] leading-none", color)}>{value}</div>
      {hint && <div className="mt-2 text-xs text-muted">{hint}</div>}
    </Card>
  );
}

/* ------------------------------------------------------------------ Form controls */
export function Field({ label, hint, error, children, htmlFor }: { label: ReactNode; hint?: ReactNode; error?: string | null; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-[13px] font-medium text-fg/90">{label}</label>
      {children}
      {error ? <p role="alert" className="text-xs text-bad">{error}</p> : hint ? <p className="text-xs text-muted">{hint}</p> : null}
    </div>
  );
}
const inputCls = "w-full rounded-xl border border-line/15 bg-sunken/70 px-3.5 text-sm text-fg placeholder:text-faint transition focus:border-accent/60 focus:outline-none focus:ring-2 focus:ring-accent/25 disabled:opacity-50";
export function Input({ className, invalid, ...p }: InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }) {
  return <input className={cn(inputCls, "h-10", invalid && "border-bad/60", className)} aria-invalid={invalid || undefined} {...p} />;
}
export function Textarea({ className, ...p }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cn(inputCls, "min-h-[88px] py-2.5", className)} {...p} />;
}
export function Select({ className, children, ...p }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cn(inputCls, "h-10 appearance-none bg-[length:16px] pr-8", className)} {...p}>{children}</select>;
}
export function Slider({ label, value, onChange, min, max, step, format, hint, id }: { label: ReactNode; value: number; onChange: (v: number) => void; min: number; max: number; step: number; format?: (v: number) => string; hint?: ReactNode; id?: string }) {
  const gen = useId(); const sid = id ?? gen;
  const p = ((value - min) / (max - min)) * 100;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label htmlFor={sid} className="text-[13px] font-medium">{label}</label>
        <span className="num rounded-md bg-sunken px-2 py-0.5 text-xs text-fg">{format ? format(value) : value}</span>
      </div>
      <input id={sid} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full accent-accent"
        style={{ background: `linear-gradient(90deg, rgb(var(--accent)) ${p}%, rgb(var(--line)/.18) ${p}%)` }} />
      {hint && <p className="text-xs text-muted">{hint}</p>}
    </div>
  );
}
export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} disabled={disabled} onClick={() => onChange(!checked)}
      className={cn("relative h-6 w-11 shrink-0 rounded-full transition-colors disabled:opacity-40", checked ? "bg-accent" : "bg-line/25")}>
      <span className={cn("absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all", checked ? "left-[22px]" : "left-0.5")} />
    </button>
  );
}

/* ------------------------------------------------------------------ Progress / skeleton / empty */
export function ProgressBar({ value, tone = "accent", indeterminate, className, label }: { value?: number; tone?: "accent" | "good" | "warn" | "bad"; indeterminate?: boolean; className?: string; label?: string }) {
  const color = { accent: "bg-accent", good: "bg-good", warn: "bg-warn", bad: "bg-bad" }[tone];
  return (
    <div role="progressbar" aria-label={label} aria-valuenow={indeterminate ? undefined : Math.round((value ?? 0) * 100)} aria-valuemin={0} aria-valuemax={100}
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-line/15", className)}>
      {indeterminate ? <div className={cn("absolute h-full w-2/5 rounded-full animate-indeterminate", color)} /> :
        <div className={cn("h-full rounded-full transition-[width] duration-700 ease-out", color)} style={{ width: `${Math.min(100, Math.max(0, (value ?? 0) * 100))}%` }} />}
    </div>
  );
}
export function Skeleton({ className }: { className?: string }) { return <div className={cn("skeleton", className)} aria-hidden />; }
export function EmptyState({ icon, title, body, action }: { icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-line/20 px-6 py-14 text-center">
      {icon && <div className="grid h-12 w-12 place-items-center rounded-2xl bg-accent/10 text-accent">{icon}</div>}
      <h3 className="text-base">{title}</h3>
      {body && <p className="max-w-md text-sm text-muted">{body}</p>}
      {action}
    </div>
  );
}
export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const msg = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div role="alert" className="flex items-start gap-3 rounded-2xl border border-bad/30 bg-bad/8 p-4">
      <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-bad" />
      <div className="flex-1"><p className="text-sm font-medium text-bad">Request failed</p><p className="mt-0.5 text-sm text-muted">{msg}</p></div>
      {onRetry && <Button size="sm" variant="secondary" onClick={onRetry}>Retry</Button>}
    </div>
  );
}
export function Callout({ tone = "info", title, children }: { tone?: "info" | "warn" | "bad" | "good"; title?: ReactNode; children?: ReactNode }) {
  const t = { info: ["border-accent/30 bg-accent/8", <Info key="i" className="h-4 w-4 text-accent" />], warn: ["border-warn/30 bg-warn/8", <AlertTriangle key="w" className="h-4 w-4 text-warn" />],
    bad: ["border-bad/30 bg-bad/8", <XCircle key="b" className="h-4 w-4 text-bad" />], good: ["border-good/30 bg-good/8", <CheckCircle2 key="g" className="h-4 w-4 text-good" />] }[tone];
  return (
    <div className={cn("flex gap-3 rounded-xl border p-3.5 text-sm", t[0] as string)}>
      <div className="mt-0.5 shrink-0">{t[1] as ReactNode}</div>
      <div className="min-w-0">{title && <div className="font-medium">{title}</div>}{children && <div className={cn("text-muted", title && "mt-0.5")}>{children}</div>}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ Tabs */
export function Tabs<T extends string>({ tabs, value, onChange, className }: { tabs: { id: T; label: ReactNode; icon?: ReactNode }[]; value: T; onChange: (v: T) => void; className?: string }) {
  return (
    <div role="tablist" className={cn("inline-flex max-w-full gap-1 overflow-x-auto rounded-xl border border-line/10 bg-sunken/70 p-1", className)}>
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          className={cn("relative flex items-center gap-2 whitespace-nowrap rounded-lg px-3.5 py-1.5 text-[13px] font-medium transition", value === t.id ? "text-fg" : "text-muted hover:text-fg")}>
          {value === t.id && <motion.span layoutId={`tab-${tabs.map((x) => x.id).join("-")}`} className="absolute inset-0 rounded-lg bg-raised shadow-sm ring-1 ring-line/15" transition={{ type: "spring", duration: 0.4, bounce: 0.15 }} />}
          <span className="relative flex items-center gap-2">{t.icon}{t.label}</span>
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ Modal */
export function Modal({ open, onClose, title, children, footer, wide }: { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; footer?: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    ref.current?.focus();
    const prevOverflow = document.body.style.overflow; document.body.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = prevOverflow; prev?.focus?.(); };
  }, [open, onClose]);
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-50 flex items-end justify-center p-0 sm:items-center sm:p-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} aria-hidden />
          <motion.div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={typeof title === "string" ? title : undefined}
            initial={{ y: 24, opacity: 0, scale: 0.98 }} animate={{ y: 0, opacity: 1, scale: 1 }} exit={{ y: 16, opacity: 0 }} transition={{ type: "spring", duration: 0.35, bounce: 0.1 }}
            className={cn("card relative z-10 flex max-h-[92vh] w-full flex-col overflow-hidden rounded-b-none bg-surface sm:rounded-2xl", wide ? "max-w-3xl" : "max-w-lg")}>
            <div className="flex items-center justify-between border-b border-line/10 px-5 py-4">
              <h2 className="text-base">{title}</h2>
              <button onClick={onClose} aria-label="Close dialog" className="rounded-lg p-1.5 text-muted hover:bg-line/10 hover:text-fg"><X className="h-4 w-4" /></button>
            </div>
            <div className="overflow-y-auto px-5 py-5">{children}</div>
            {footer && <div className="flex flex-wrap justify-end gap-2 border-t border-line/10 bg-sunken/40 px-5 py-3.5">{footer}</div>}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ------------------------------------------------------------------ Animated number */
export function AnimatedNumber({ value, format, className }: { value: number; format: (v: number) => string; className?: string }) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  useEffect(() => {
    if (reduce) { setShown(value); from.current = value; return; }
    const start = performance.now(); const a = from.current; const dur = 700; let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - start) / dur); const e = 1 - Math.pow(1 - k, 3);
      setShown(a + (value - a) * e);
      if (k < 1) raf = requestAnimationFrame(tick); else from.current = value;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, reduce]);
  return <span className={className}>{format(shown)}</span>;
}

/* ------------------------------------------------------------------ Ring gauge */
export function Ring({ value, size = 96, stroke = 9, label, sub, tone = "accent" }: { value: number; size?: number; stroke?: number; label: ReactNode; sub?: ReactNode; tone?: "accent" | "good" | "warn" | "bad" }) {
  const r = (size - stroke) / 2; const c = 2 * Math.PI * r; const v = Math.min(1, Math.max(0, value));
  const color = { accent: "rgb(var(--accent))", good: "rgb(var(--good))", warn: "rgb(var(--warn))", bad: "rgb(var(--bad))" }[tone];
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }} role="img" aria-label={`${Math.round(v * 100)} percent`}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(var(--line)/.14)" strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - v)}
          style={{ transition: "stroke-dashoffset .9s cubic-bezier(.2,.8,.2,1)", filter: `drop-shadow(0 0 6px ${color})` }} />
      </svg>
      <div className="absolute text-center leading-tight"><div className="num text-lg font-medium">{label}</div>{sub && <div className="text-[10px] uppercase tracking-wider text-faint">{sub}</div>}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ Tooltip-ish hint */
export function Hint({ children, text }: { children: ReactNode; text: string }) {
  return <span title={text} className="cursor-help underline decoration-dotted decoration-faint underline-offset-4">{children}</span>;
}
export const Kbd = ({ children }: { children: ReactNode }) => <kbd className="rounded-md border border-line/20 bg-sunken px-1.5 py-0.5 font-mono text-[11px] text-muted">{children}</kbd>;
