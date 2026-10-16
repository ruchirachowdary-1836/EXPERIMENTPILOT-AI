"""PDF / CSV rendering of a diagnosis report dictionary."""
from __future__ import annotations

import csv
import io
from xml.sax.saxutils import escape

from ..security.uploads import csv_safe

NAVY, BLUE, SLATE = "#0b1220", "#2563eb", "#475569"


def _charts(report: dict) -> list[bytes]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    rk = report["ranking"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = [r["name"][:34] for r in rk][::-1]
    ax.barh(names, [r["posterior"] for r in rk][::-1], color=BLUE)
    ax.barh(names, [r["prior"] for r in rk][::-1], height=0.25, color="#94a3b8")
    ax.set(xlim=(0, 1), xlabel="Posterior probability (blue) vs prior (grey)")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160); plt.close(fig); out.append(buf.getvalue())
    hist = report["posterior_history"]
    if len(hist) > 1:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for r in rk[:6]:
            ax.plot([h["step"] for h in hist], [h["marginals"].get(r["hypothesis"], 0) for h in hist], marker="o", ms=3,
                    label=r["name"][:28])
        ax.set(ylim=(0, 1), xlabel="Experiment step", ylabel="Marginal probability")
        ax.legend(fontsize=6); ax.grid(alpha=0.3); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160); plt.close(fig); out.append(buf.getvalue())
    return out


def render_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor(NAVY), fontSize=22, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor(BLUE), spaceBefore=12)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor(SLATE))
    P = lambda t, st=body: Paragraph(escape(str(t)), st)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"ExperimentPilot diagnosis - {report['session']['name']}")
    st = []
    st += [P("ExperimentPilot AI - Diagnosis Report", h1), P(report["session"]["name"], ss["Heading3"]),
           P(f"Session {report['session']['id']} | status: {report['session']['status']} | planner: {report['session']['planner']} | "
             f"generated {report['generated_at'][:19]} UTC", small), Spacer(1, 6), P(report["verdict"], ParagraphStyle(
               "v", parent=body, fontSize=11, leading=15, textColor=colors.HexColor(NAVY)))]
    c = report["confidence"]
    st.append(P(f"Diagnosis confidence (probability of the single most likely exact combination of causes): {c['map_probability']:.1%} "
                f"(threshold {c['threshold']:.0%}, {'reached' if c['reached'] else 'not reached'}). Residual uncertainty: "
                f"{c['entropy_bits']:.2f} bits of {c['initial_entropy_bits']:.2f} initially ({c['uncertainty_reduction']:.0%} reduction).",
                body))
    if report["session"].get("stop_message"):
        st.append(P("Stopping reason: " + report["session"]["stop_message"], body))
    st.append(P("Ranked root-cause hypotheses", h2))
    rows = [["#", "Hypothesis", "Posterior", "Prior", "Status"]] + [
        [i + 1, f"{r['code']} {r['name']}", f"{r['posterior']:.1%}", f"{r['prior']:.1%}", r["status"]] for i, r in enumerate(report["ranking"])]
    t = Table(rows, colWidths=[8 * mm, 85 * mm, 24 * mm, 24 * mm, 25 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]))
    st.append(t)
    for png in _charts(report):
        st += [Spacer(1, 6), Image(io.BytesIO(png), width=165 * mm, height=75 * mm)]
    k, rk = report["cost"], report["risk"]
    st += [P("Cost, budget and risk", h2),
           P(f"Total diagnostic cost {k['spent']:.2f} of budget {k['budget']:.2f} ({k['utilization']:.0%} utilised); "
             f"{k['experiments_run']} experiments. Total risk {rk['total']:.2f}, highest single-experiment risk {rk['max']:.2f} "
             f"(cap {rk['cap']:.2f}), cap violations: {rk['violations']}.", body)]
    st.append(P("Experiments performed", h2))
    for r in report["experiments"]:
        if r["status"] not in ("completed", "external"):
            continue
        st.append(P(f"Step {r['step']}: {r['name']} -> {str(r['outcome']).replace('_', ' ')} (cost {r['cost']:.2f}, {r['source']})",
                    ParagraphStyle("e", parent=body, fontName="Helvetica-Bold")))
        st.append(P(r["interpretation"], body))
        for f in r["findings"][:4]:
            st.append(P(f"- [{f['level']}] {f['text']}", small))
    st.append(P("Supporting and contradicting evidence", h2))
    for r in report["ranking"][:4]:
        sup = ", ".join(f"{e['experiment']} ({e['outcome'].replace('_', ' ')}, LR {e['likelihood_ratio']:.1f})" for e in r["supporting"]) or "none"
        con = ", ".join(f"{e['experiment']} ({e['outcome'].replace('_', ' ')}, LR {e['likelihood_ratio']:.2f})" for e in r["contradicting"]) or "none"
        st += [P(f"{r['name']} ({r['posterior']:.1%})", ParagraphStyle("hh", parent=body, fontName="Helvetica-Bold")),
               P("Supports: " + sup, small), P("Contradicts: " + con, small)]
    if report["alternatives"]:
        st.append(P("Alternative explanations still plausible", h2))
        st.append(P("; ".join(f"{a['name']} ({a['posterior']:.1%})" for a in report["alternatives"]), body))
    st.append(P("Suggested remediation and validation", h2))
    shown = False
    for r in report["ranking"]:
        if r["remediation"] and r["posterior"] >= 0.2:
            shown = True
            st.append(P(f"{r['name']} ({r['posterior']:.1%})", ParagraphStyle("hr", parent=body, fontName="Helvetica-Bold")))
            for step in r["remediation"]:
                st.append(P("- " + step, body))
    if not shown:
        st.append(P("No hypothesis reached a level that justifies specific remediation; gather more evidence first.", body))
    st.append(P("Limitations and missing evidence", h2))
    for l in report["limitations"]:
        st.append(P("- " + l, small))
    st += [Spacer(1, 8), P(report["disclaimer"], small)]
    doc.build(st)
    return buf.getvalue()


def _csv(rows: list[list], header: list[str]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(header)
    for r in rows:
        w.writerow([csv_safe(v) for v in r])
    return out.getvalue()


def hypotheses_csv(report: dict) -> str:
    return _csv([[i + 1, r["code"], r["name"], f"{r['posterior']:.6f}", f"{r['prior']:.6f}", r["status"],
                  len(r["supporting"]), len(r["contradicting"])] for i, r in enumerate(report["ranking"])],
                ["rank", "code", "hypothesis", "posterior", "prior", "status", "supporting_evidence", "contradicting_evidence"])


def experiments_csv(report: dict) -> str:
    return _csv([[r["step"], r["experiment_id"], r["status"], r["source"], r["outcome"], r["cost"], r["risk"],
                  r["started_at"], r["interpretation"]] for r in report["experiments"]],
                ["step", "experiment", "status", "source", "outcome", "cost", "risk", "started_at", "interpretation"])


def history_csv(report: dict) -> str:
    hyps = [r["hypothesis"] for r in report["ranking"]]
    return _csv([[h["step"], h["label"], h["spent"], h["entropy"], h["map_probability"]] + [h["marginals"].get(x, 0) for x in hyps]
                 for h in report["posterior_history"]], ["step", "label", "spent", "entropy_bits", "map_probability"] + hyps)
