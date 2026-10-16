import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

/** Full journey: create project → upload data → map columns → start diagnosis → run experiments → observe updates → stop → download report. */
test("diagnosis end to end", async ({ page }) => {
  test.setTimeout(240_000);
  // Build small CSVs on the fly
  const rows = (n: number, shift: number) => {
    let s = "x1,x2,target,prediction\n";
    for (let i = 0; i < n; i++) {
      const x1 = Math.sin(i * 12.9898) * 2 + shift, x2 = Math.cos(i * 78.233) * 2;
      const y = x1 - shift + 0.5 * x2 > 0 ? 1 : 0;
      s += `${x1.toFixed(4)},${x2.toFixed(4)},${y},${x1 + 0.5 * x2 > 0 ? 1 : 0}\n`;
    }
    return s;
  };
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ep-e2e-"));
  fs.writeFileSync(path.join(dir, "reference.csv"), rows(400, 0));
  fs.writeFileSync(path.join(dir, "recent.csv"), rows(400, 1.5));

  await page.goto("/app/projects");
  await page.getByRole("button", { name: /new project/i }).click();
  await page.getByLabel("Project name").fill("E2E project");
  await page.getByRole("button", { name: /create project/i }).click();
  await expect(page.getByRole("heading", { name: "E2E project" })).toBeVisible();

  await page.getByLabel("Upload Reference data").setInputFiles(path.join(dir, "reference.csv"));
  await expect(page.getByText("reference.csv")).toBeVisible();
  await page.getByLabel("Upload Recent data").setInputFiles(path.join(dir, "recent.csv"));
  await expect(page.getByText("recent.csv")).toBeVisible();

  await page.getByRole("tab", { name: /column mapping/i }).click();
  await page.getByLabel("Target (label)").selectOption("target");
  await page.getByLabel("Prediction").selectOption("prediction");
  await page.getByRole("button", { name: /save mapping/i }).click();
  await expect(page.getByText(/column mapping saved/i)).toBeVisible();

  await page.getByRole("button", { name: /configure diagnosis/i }).click();
  await page.getByRole("button", { name: /start diagnosis/i }).click();
  await expect(page.getByTestId("next-name")).toBeVisible();

  const before = await page.getByText(/bits/).first().textContent();
  for (let i = 0; i < 2; i++) {
    await page.getByRole("button", { name: /run recommended/i }).click();
    await expect(page.getByText(/completed|Experiment log/i).first()).toBeVisible();
    await page.waitForTimeout(2500);
  }
  await expect(page.getByText(/^1\. /)).toBeVisible();
  expect(await page.getByText(/bits/).first().textContent()).not.toBeNull();
  expect(before).not.toBeNull();

  page.once("dialog", (d) => d.accept());
  await page.getByRole("button", { name: /^stop$/i }).click();
  await page.getByRole("link", { name: /report/i }).first().click();
  await expect(page.getByTestId("verdict")).toBeVisible();
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: /pdf report/i }).click()]);
  expect(download.suggestedFilename()).toMatch(/\.pdf$/);
});
