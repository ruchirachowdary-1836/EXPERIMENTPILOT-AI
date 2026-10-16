import { defineConfig } from "@playwright/test";

// Requires the backend on :8000 and the frontend dev/preview server on :5173 (see README).
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  use: { baseURL: process.env.E2E_BASE_URL ?? "http://localhost:5173", trace: "retain-on-failure" },
  reporter: [["list"]],
});
