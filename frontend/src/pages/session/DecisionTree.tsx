import { useQuery } from "@tanstack/react-query";
import { Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import { api } from "@/api/client";
import { layoutTree } from "@/decision-tree/layout";
import { nodeTypes } from "@/decision-tree/nodes";
import { Callout, Card, EmptyState, ErrorState, Skeleton } from "@/components/ui";
import { GitBranch } from "lucide-react";
import { useSessionCtx } from "./SessionLayout";

export default function DecisionTree() {
  const { session } = useSessionCtx();
  const q = useQuery({ queryKey: ["tree", session.id, session.version], queryFn: () => api.tree(session.id) });
  const laid = useMemo(() => (q.data ? layoutTree(q.data) : null), [q.data]);
  if (q.isLoading) return <Skeleton className="h-[560px]" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  if (!laid || laid.nodes.length < 2) return <EmptyState icon={<GitBranch className="h-6 w-6" />} title="No path yet" body="Run an experiment to grow the tree." />;
  const observed = laid.nodes.filter((n) => n.type === "outcome" && (n.data as any).taken).length;
  return (
    <div className="space-y-4">
      <Callout tone="info" title="How to read this">
        Solid blue edges are the path actually taken ({observed} observed outcome{observed === 1 ? "" : "s"}). Dashed grey branches are outcomes that did <b>not</b> occur: they show the posterior you would have had and the experiment the planner would have chosen next. Drag nodes, scroll to zoom.
      </Callout>
      <Card className="overflow-hidden">
        <div className="h-[680px] w-full" data-testid="decision-tree">
          <ReactFlowProvider>
            <ReactFlow nodes={laid.nodes} edges={laid.edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.15, maxZoom: 1, nodes: laid.nodes.filter((n) => n.position.x <= 4 * 320).map((n) => ({ id: n.id })) }} minZoom={0.3} maxZoom={1.6} nodesConnectable={false} proOptions={{ hideAttribution: true }}>
              <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="rgb(148 163 184 / .18)" />
              <Controls showInteractive={false} />
              <MiniMap pannable zoomable nodeColor={(n) => ((n.data as any).taken ? "rgb(79 140 255)" : "rgb(100 116 139)")} maskColor="rgb(0 0 0 / .35)" />
            </ReactFlow>
          </ReactFlowProvider>
        </div>
      </Card>
    </div>
  );
}
