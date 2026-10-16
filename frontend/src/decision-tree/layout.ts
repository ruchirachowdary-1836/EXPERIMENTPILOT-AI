import type { Edge, Node } from "@xyflow/react";
import type { Tree } from "@/types/api";

const X = 320;
const Y = 128;

/** Left-to-right layout: the path actually taken runs along y = 0; unobserved outcomes fan out above and below. */
export function layoutTree(tree: Tree): { nodes: Node[]; edges: Edge[] } {
  const byParent = new Map<string, string[]>();
  for (const e of tree.edges) byParent.set(e.source, [...(byParent.get(e.source) ?? []), e.target]);
  const nodes: Node[] = tree.nodes.map((n) => {
    let y = 0;
    if (n.type === "outcome" && !n.taken) {
      const siblings = tree.nodes.filter((m) => m.type === "outcome" && m.depth === n.depth && !m.taken && m.id.split(".")[0] === n.id.split(".")[0]);
      const idx = siblings.findIndex((m) => m.id === n.id);
      const k = Math.floor(idx / 2) + 1;
      y = (idx % 2 === 0 ? -1 : 1) * k * Y;
    }
    return { id: n.id, type: n.type, position: { x: n.depth * X, y }, data: { ...n.data, taken: n.taken }, draggable: true };
  });
  const edges: Edge[] = tree.edges.map((e) => ({
    id: e.id, source: e.source, target: e.target, label: e.label || undefined, animated: e.taken,
    className: e.taken ? "taken" : "hypothetical", type: "smoothstep",
    style: e.taken ? { stroke: "rgb(var(--accent))", strokeWidth: 2.5 } : { stroke: "rgb(var(--faint))", strokeDasharray: "6 5", strokeWidth: 1.4 },
    labelStyle: { fill: "rgb(var(--muted))", fontSize: 11 }, labelBgStyle: { fill: "rgb(var(--surface))" },
  }));
  return { nodes, edges };
}
