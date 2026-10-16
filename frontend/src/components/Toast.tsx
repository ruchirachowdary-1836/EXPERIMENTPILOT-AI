import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

type Tone = "success" | "error" | "info";
interface Toast { id: number; tone: Tone; title: string; body?: string }
interface Ctx { push: (t: Omit<Toast, "id">) => void; error: (e: unknown, title?: string) => void; success: (title: string, body?: string) => void }

const ToastCtx = createContext<Ctx | null>(null);
let seq = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const dismiss = useCallback((id: number) => setItems((l) => l.filter((t) => t.id !== id)), []);
  const push = useCallback((t: Omit<Toast, "id">) => {
    const id = ++seq;
    setItems((l) => [...l.slice(-3), { ...t, id }]);
    setTimeout(() => dismiss(id), t.tone === "error" ? 8000 : 4500);
  }, [dismiss]);
  const value = useMemo<Ctx>(() => ({
    push,
    success: (title, body) => push({ tone: "success", title, body }),
    error: (e, title = "Something went wrong") => push({ tone: "error", title, body: e instanceof Error ? e.message : String(e) }),
  }), [push]);
  const icon = { success: <CheckCircle2 className="h-4 w-4 text-good" />, error: <AlertTriangle className="h-4 w-4 text-bad" />, info: <Info className="h-4 w-4 text-accent" /> };
  return (
    <ToastCtx.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[min(92vw,380px)] flex-col gap-2" role="region" aria-live="polite" aria-label="Notifications">
        <AnimatePresence>
          {items.map((t) => (
            <motion.div key={t.id} layout initial={{ opacity: 0, y: 16, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, x: 40 }}
              className={cn("card pointer-events-auto flex items-start gap-3 bg-surface p-3.5 shadow-xl", t.tone === "error" && "border-bad/30")}>
              <div className="mt-0.5">{icon[t.tone]}</div>
              <div className="min-w-0 flex-1"><p className="text-sm font-medium">{t.title}</p>{t.body && <p className="mt-0.5 break-words text-[13px] text-muted">{t.body}</p>}</div>
              <button onClick={() => dismiss(t.id)} aria-label="Dismiss" className="rounded p-1 text-faint hover:text-fg"><X className="h-3.5 w-3.5" /></button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  );
}

export function useToast(): Ctx {
  const c = useContext(ToastCtx);
  if (!c) throw new Error("useToast must be used inside ToastProvider");
  return c;
}
