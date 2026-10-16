/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        raised: "rgb(var(--raised) / <alpha-value>)",
        sunken: "rgb(var(--sunken) / <alpha-value>)",
        line: "rgb(var(--line) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        faint: "rgb(var(--faint) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-fg": "rgb(var(--accent-fg) / <alpha-value>)",
        cyan: "rgb(var(--cyan) / <alpha-value>)",
        good: "rgb(var(--good) / <alpha-value>)",
        warn: "rgb(var(--warn) / <alpha-value>)",
        bad: "rgb(var(--bad) / <alpha-value>)",
      },
      fontFamily: {
        sans: ['"Inter Variable"', "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgb(255 255 255 / 0.04) inset, 0 8px 24px -12px rgb(0 0 0 / 0.5)",
        glow: "0 0 0 1px rgb(var(--accent) / 0.35), 0 8px 32px -8px rgb(var(--accent) / 0.45)",
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        floaty: { "0%,100%": { transform: "translateY(0)" }, "50%": { transform: "translateY(-6px)" } },
        pulseRing: { "0%": { transform: "scale(.8)", opacity: ".7" }, "100%": { transform: "scale(2.2)", opacity: "0" } },
        indeterminate: { "0%": { left: "-40%" }, "100%": { left: "100%" } },
        bar: { from: { transform: "scaleX(0)" }, to: { transform: "scaleX(1)" } },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
        floaty: "floaty 6s ease-in-out infinite",
        pulseRing: "pulseRing 1.8s ease-out infinite",
        indeterminate: "indeterminate 1.4s ease-in-out infinite",
        bar: "bar .9s cubic-bezier(.2,.8,.2,1) both",
      },
    },
  },
  plugins: [],
};
