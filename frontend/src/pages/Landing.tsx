import { motion } from "framer-motion";
import { ArrowRight, Binary, Compass, FlaskConical, GitBranch, Layers, Scale, ShieldCheck, Sigma, Target, Wallet } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Logo } from "@/components/Logo";
import { ThemeToggle } from "@/components/AppShell";
import { Badge, Button, Card } from "@/components/ui";
import { hypColor } from "@/lib/utils";

const fade = { initial: { opacity: 0, y: 18 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: "-60px" }, transition: { duration: 0.55, ease: [0.2, 0.8, 0.2, 1] } } as const;

/** Illustrative only: cycles through three made-up belief states to show how evidence moves probability mass. */
const FRAMES = [
  { label: "Prior", note: "before any experiment", p: [0.3, 0.2, 0.15, 0.2, 0.2, 0.2] },
  { label: "After feature-distribution test: severe shift", note: "drift, pipeline change and data quality gain mass", p: [0.46, 0.08, 0.05, 0.07, 0.4, 0.3] },
  { label: "After pipeline replay: outputs differ", note: "pipeline change separates from drift", p: [0.14, 0.05, 0.04, 0.05, 0.88, 0.1] },
];
const HYP = [["covariate_drift", "Covariate drift"], ["label_noise", "Label noise"], ["feature_leakage", "Feature leakage"], ["class_prior_shift", "Class-prior shift"], ["pipeline_change", "Pipeline change"], ["data_quality", "Data quality"]];

function HeroViz() {
  const [i, setI] = useState(0);
  useEffect(() => { const t = setInterval(() => setI((x) => (x + 1) % FRAMES.length), 3200); return () => clearInterval(t); }, []);
  const f = FRAMES[i];
  return (
    <div className="relative">
      <div className="glow-blob absolute -inset-10 -z-10 opacity-60" />
      <Card className="overflow-hidden p-5 sm:p-6">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-bad/70" /><span className="h-2.5 w-2.5 rounded-full bg-warn/70" /><span className="h-2.5 w-2.5 rounded-full bg-good/70" /></div>
          <Badge tone="muted">Illustrative animation</Badge>
        </div>
        <div className="eyebrow">Step {i}</div>
        <p className="mt-1 min-h-[44px] text-sm font-medium">{f.label}</p>
        <div className="mt-4 space-y-3">
          {HYP.map(([id, name], k) => (
            <div key={id} className="grid grid-cols-[110px_1fr_40px] items-center gap-3 text-xs sm:grid-cols-[128px_1fr_44px]">
              <span className="truncate text-muted">{name}</span>
              <div className="h-2.5 overflow-hidden rounded-full bg-line/12">
                <motion.div className="h-full rounded-full" style={{ background: hypColor(id) }} animate={{ width: `${f.p[k] * 100}%` }} transition={{ duration: 0.9, ease: [0.2, 0.8, 0.2, 1] }} />
              </div>
              <span className="num text-right">{Math.round(f.p[k] * 100)}%</span>
            </div>
          ))}
        </div>
        <p className="mt-5 border-t border-line/10 pt-3 text-xs text-muted">{f.note}. Values are a hand-made illustration, not a result.</p>
      </Card>
    </div>
  );
}

const STEPS = [
  { icon: Layers, t: "Represent", d: "Competing root causes become a joint probability distribution over which causes are active, so several can be true at once." },
  { icon: Compass, t: "Recommend", d: "Every feasible experiment is scored by expected information gain per unit of cost and risk, under budget and risk-cap constraints." },
  { icon: FlaskConical, t: "Execute", d: "The experiment runs on a copy of your data in a time-limited sandbox. Or you run it yourself and enter the result." },
  { icon: Sigma, t: "Update", d: "Bayes’ rule moves probability using a noisy likelihood model. No outcome is ever treated as certain." },
  { icon: GitBranch, t: "Explain", d: "Every step records why it was chosen, what it ruled out, and which hypotheses the evidence supports or contradicts." },
  { icon: Target, t: "Conclude", d: "Stop at the confidence threshold, when the budget runs out, or when more tests are not worth their cost. Residual uncertainty is reported." },
];

const CAPS = [
  { icon: Binary, t: "Multi-cause Bayesian engine", d: "Exact enumeration of joint binary states with noisy-OR likelihoods and a documented sparse approximation for larger hypothesis sets." },
  { icon: Wallet, t: "Cost- and risk-aware planning", d: "Budget, risk cap and risk weight are configurable. Seven planner variants are implemented and benchmarked against each other." },
  { icon: FlaskConical, t: "Ten real experiments", d: "Drift tests, data-quality audits, confident-learning label audit, leakage scan, pipeline replay, importance weighting, slices, retraining and metric recomputation." },
  { icon: ShieldCheck, t: "Safe by construction", d: "Uploads are validated and never executed or unpickled. Experiments run on copies with time limits. Risky actions need explicit approval." },
  { icon: GitBranch, t: "Interactive decision tree", d: "See the path actually taken, the alternative outcomes that did not happen, and what would have been recommended next." },
  { icon: Scale, t: "Honest evaluation", d: "A reproducible fault-injection benchmark with separate estimation and evaluation seeds, confidence intervals and paired tests." },
];

export default function Landing() {
  return (
    <div className="relative overflow-x-hidden">
      <div className="grid-bg pointer-events-none absolute inset-x-0 top-0 h-[640px]" aria-hidden />
      <header className="relative z-10 mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Logo />
        <nav className="hidden items-center gap-7 text-sm text-muted md:flex" aria-label="Sections">
          <a href="#how" className="hover:text-fg">How it works</a><a href="#capabilities" className="hover:text-fg">Capabilities</a>
          <a href="#math" className="hover:text-fg">The math</a><a href="#transparency" className="hover:text-fg">Transparency</a>
        </nav>
        <div className="flex items-center gap-2"><ThemeToggle /><Link to="/app/projects"><Button variant="primary" size="sm">Open app</Button></Link></div>
      </header>

      <section className="relative z-10 mx-auto grid max-w-6xl items-center gap-12 px-4 pb-20 pt-12 sm:px-6 lg:grid-cols-[1.05fr_1fr] lg:pt-20">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
          <Badge tone="accent" dot>Active diagnosis for ML failures</Badge>
          <h1 className="mt-5 text-[40px] font-semibold leading-[1.05] tracking-tight sm:text-6xl">
            Your model broke.<br /><span className="bg-gradient-to-r from-accent to-cyan bg-clip-text text-transparent">Which test should you run next?</span>
          </h1>
          <p className="mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
            ExperimentPilot AI treats model-failure diagnosis as a sequential decision problem. It keeps a probability distribution over root causes and
            picks the diagnostic experiment with the best information per cost and risk, then updates its beliefs from real measurements on your data.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/app/projects"><Button variant="primary" size="lg" icon={<ArrowRight className="h-4 w-4" />}>Start diagnosing</Button></Link>
            <Link to="/app/benchmark"><Button size="lg">View the benchmark</Button></Link>
          </div>
          <dl className="mt-10 grid max-w-md grid-cols-3 gap-6 border-t border-line/10 pt-6">
            {[["8", "root-cause hypotheses"], ["10", "executable experiments"], ["7", "planner variants"]].map(([n, l]) => (
              <div key={l}><dt className="num text-3xl font-medium">{n}</dt><dd className="mt-1 text-xs text-muted">{l}</dd></div>
            ))}
          </dl>
        </motion.div>
        <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8, delay: 0.15 }}><HeroViz /></motion.div>
      </section>

      <section className="relative z-10 mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <motion.div {...fade} className="grid gap-8 lg:grid-cols-[1fr_1.2fr] lg:items-center">
          <div>
            <div className="eyebrow">The problem</div>
            <h2 className="mt-3 text-3xl sm:text-4xl">Debugging a degraded model is done by habit.</h2>
            <p className="mt-4 text-muted leading-relaxed">Accuracy falls from 92% to 71%. Drift, label errors, leakage, a class-prior change, a preprocessing release, upstream nulls or a metric bug could each be responsible, and each check costs time, compute or labelling effort. Some checks touch production resources.</p>
          </div>
          <Card className="card-pad">
            <ul className="space-y-4 text-sm">
              {[["Monitoring tools", "detect and describe problems."], ["Explanation tools", "attribute individual predictions."], ["Neither", "chooses the next experiment under cost and risk constraints. That is the gap this project explores."]].map(([a, b]) => (
                <li key={a} className="flex gap-3"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" /><span><b className="font-medium">{a}</b> <span className="text-muted">{b}</span></span></li>
              ))}
            </ul>
          </Card>
        </motion.div>
      </section>

      <section id="how" className="relative z-10 mx-auto max-w-6xl scroll-mt-20 px-4 py-16 sm:px-6">
        <motion.div {...fade}><div className="eyebrow">How adaptive diagnosis works</div><h2 className="mt-3 max-w-2xl text-3xl sm:text-4xl">A closed loop of belief, experiment and evidence.</h2></motion.div>
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {STEPS.map((s, i) => (
            <motion.div key={s.t} {...fade} transition={{ ...fade.transition, delay: i * 0.05 }}>
              <Card hover className="h-full card-pad">
                <div className="flex items-center justify-between"><div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/12 text-accent"><s.icon className="h-5 w-5" /></div><span className="num text-xs text-faint">0{i + 1}</span></div>
                <h3 className="mt-4 text-base">{s.t}</h3><p className="mt-1.5 text-sm leading-relaxed text-muted">{s.d}</p>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      <section id="capabilities" className="relative z-10 mx-auto max-w-6xl scroll-mt-20 px-4 py-16 sm:px-6">
        <motion.div {...fade}><div className="eyebrow">Key capabilities</div><h2 className="mt-3 max-w-2xl text-3xl sm:text-4xl">Built as research software, not a dashboard.</h2></motion.div>
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {CAPS.map((c, i) => (
            <motion.div key={c.t} {...fade} transition={{ ...fade.transition, delay: i * 0.05 }}>
              <Card className="h-full card-pad"><c.icon className="h-5 w-5 text-cyan" /><h3 className="mt-4 text-base">{c.t}</h3><p className="mt-1.5 text-sm leading-relaxed text-muted">{c.d}</p></Card>
            </motion.div>
          ))}
        </div>
      </section>

      <section id="math" className="relative z-10 mx-auto max-w-6xl scroll-mt-20 px-4 py-16 sm:px-6">
        <motion.div {...fade}><div className="eyebrow">Mathematical overview</div><h2 className="mt-3 max-w-2xl text-3xl sm:text-4xl">Expected information gain, priced in cost and risk.</h2></motion.div>
        <motion.div {...fade} className="mt-10 grid gap-4 lg:grid-cols-3">
          {[
            ["State", "z ∈ {0,1}ᴷ", "Which of K causes are active. Belief b(z) is exact for small K and truncated to ≤ m simultaneous causes for large K."],
            ["Information", "EIG(e) = H(b) − Σₒ P(o|b,e) · H(b|o,e)", "The expected entropy reduction (bits) of running experiment e, from the predictive outcome distribution."],
            ["Selection", "score(e) = EIG(e) / (c(e) + λ·r(e))", "Subject to c(e) ≤ budget left, r(e) ≤ risk cap, prerequisites available. Denominator is floored so zero-cost tests are safe."],
          ].map(([a, f, d]) => (
            <Card key={a} className="card-pad"><div className="eyebrow">{a}</div><code className="mt-3 block rounded-xl bg-sunken px-3.5 py-3 font-mono text-[13px] leading-relaxed text-accent">{f}</code><p className="mt-3 text-sm leading-relaxed text-muted">{d}</p></Card>
          ))}
        </motion.div>
        <motion.div {...fade} className="mt-4 grid gap-4 lg:grid-cols-2">
          <Card className="card-pad"><div className="eyebrow">Update</div><code className="mt-3 block rounded-xl bg-sunken px-3.5 py-3 font-mono text-[13px] text-accent">b′(z) ∝ b(z) · P(o | e, z)</code><p className="mt-3 text-sm text-muted">Likelihoods are cumulative noisy-OR: two simultaneous causes raise an alarm probability without joint tables for every pair.</p></Card>
          <Card className="card-pad"><div className="eyebrow">Separation</div><code className="mt-3 block rounded-xl bg-sunken px-3.5 py-3 font-mono text-[13px] text-accent">TV( P(o|e,a), P(o|e,b) )</code><p className="mt-3 text-sm text-muted">For the two leading hypotheses a and b: how differently the outcome is distributed. Shown next to every recommendation.</p></Card>
        </motion.div>
      </section>

      <section id="transparency" className="relative z-10 mx-auto max-w-6xl scroll-mt-20 px-4 py-16 sm:px-6">
        <motion.div {...fade}>
          <Card className="card-pad overflow-hidden">
            <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr] lg:items-center">
              <div>
                <div className="eyebrow">Benchmark & research transparency</div>
                <h2 className="mt-3 text-2xl sm:text-3xl">Claims are limited to what was measured.</h2>
                <ul className="mt-5 space-y-3 text-sm text-muted">
                  {["Likelihood tables are estimated from simulated fault injection on built-in datasets and are labelled as simulated everywhere.",
                    "Estimation and evaluation use disjoint random seeds; the app refuses benchmark seeds that overlap the estimation seeds.",
                    "Adaptive diagnosis is established research territory. This project does not claim to be the first, nor that any publication is guaranteed.",
                    "Outputs are statistical indicators, not proof of causation."].map((t) => <li key={t} className="flex gap-3"><ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-good" />{t}</li>)}
                </ul>
                <div className="mt-6 flex flex-wrap gap-3"><Link to="/app/benchmark"><Button variant="primary">Open benchmark lab</Button></Link><Link to="/app/docs"><Button>Read the methodology</Button></Link></div>
              </div>
              <div className="rounded-2xl border border-line/10 bg-sunken/60 p-5 text-sm text-muted">
                <div className="eyebrow mb-2">Limitations, up front</div>
                Conditional independence of experiment outcomes is assumed. Real systems may not match simulated fault signatures. Online refinement of the tables from confirmed diagnoses is future work.
              </div>
            </div>
          </Card>
        </motion.div>
      </section>

      <footer className="relative z-10 border-t border-line/10 py-10">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 text-sm text-muted sm:flex-row sm:px-6">
          <Logo /><p>Local-first research tool. No customer logos, testimonials or performance claims.</p>
        </div>
      </footer>
    </div>
  );
}
