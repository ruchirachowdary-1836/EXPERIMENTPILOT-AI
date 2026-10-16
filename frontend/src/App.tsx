import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Skeleton } from "./components/ui";
import Landing from "./pages/Landing";
import Projects from "./pages/Projects";
import ProjectDetail from "./pages/ProjectDetail";
import DiagnosticSetup from "./pages/DiagnosticSetup";
import SessionLayout from "./pages/session/SessionLayout";
import LiveDashboard from "./pages/session/LiveDashboard";
import EvidenceExplorer from "./pages/session/EvidenceExplorer";
import ReportPage from "./pages/session/ReportPage";
import Settings from "./pages/Settings";

const DecisionTree = lazy(() => import("./pages/session/DecisionTree"));
const Benchmark = lazy(() => import("./pages/Benchmark"));
const Docs = lazy(() => import("./pages/Docs"));

const Loading = () => <div className="space-y-4"><Skeleton className="h-10 w-64" /><Skeleton className="h-72 w-full" /></div>;

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<AppShell />}>
          <Route index element={<Navigate to="projects" replace />} />
          <Route path="projects" element={<Projects />} />
          <Route path="projects/:id" element={<ProjectDetail />} />
          <Route path="projects/:id/setup" element={<DiagnosticSetup />} />
          <Route path="sessions/:id" element={<SessionLayout />}>
            <Route index element={<LiveDashboard />} />
            <Route path="tree" element={<DecisionTree />} />
            <Route path="evidence" element={<EvidenceExplorer />} />
            <Route path="report" element={<ReportPage />} />
          </Route>
          <Route path="benchmark" element={<Benchmark />} />
          <Route path="docs" element={<Docs />} />
          <Route path="settings" element={<Settings />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
