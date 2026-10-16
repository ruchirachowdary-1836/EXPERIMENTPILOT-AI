import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";
const read = (): Theme => (document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark");

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("ep-theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  return { theme, setTheme, toggle };
}
