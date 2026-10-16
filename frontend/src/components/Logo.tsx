import { useId } from "react";
import { cn } from "@/lib/utils";

export function LogoMark({ className = "h-8 w-8" }: { className?: string }) {
  const gid = `lm${useId().replace(/:/g, "")}`;
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#7db1ff" /><stop offset="1" stopColor="#2563eb" /></linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill="rgb(var(--sunken))" stroke="rgb(var(--line) / .2)" />
      <circle cx="16" cy="16" r="8.5" fill="none" stroke={`url(#${gid})`} strokeWidth="2" />
      <circle cx="16" cy="16" r="2.8" fill={`url(#${gid})`} />
      <path d="M16 4v5M16 23v5M4 16h5M23 16h5" stroke={`url(#${gid})`} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function Logo({ className, compact }: { className?: string; compact?: boolean }) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <LogoMark />
      {!compact && <span className="text-[15px] font-semibold tracking-tight">Experiment<span className="text-accent">Pilot</span> <span className="text-faint font-medium">AI</span></span>}
    </span>
  );
}
