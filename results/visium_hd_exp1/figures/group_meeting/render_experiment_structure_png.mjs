import path from "node:path";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import fs from "node:fs";

const PLAYWRIGHT = path.join(
  process.env.HOME,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/.pnpm/playwright@1.59.1/node_modules/playwright/index.js",
);

const require = createRequire(import.meta.url);
const { chromium } = require(PLAYWRIGHT);

const repo = process.cwd();
const input = process.argv[2] || "experiment_structure_diagram.svg";
const output = process.argv[3] || input.replace(/\.svg$/i, "_full.png");
const svg = path.join(repo, "results/visium_hd_exp1/figures/group_meeting", input);
const out = path.join(repo, "results/visium_hd_exp1/figures/group_meeting", output);
const svgText = fs.readFileSync(svg, "utf8");
const width = Number(svgText.match(/<svg[^>]*width="([0-9.]+)"/)?.[1] || 1800);
const height = Number(svgText.match(/<svg[^>]*height="([0-9.]+)"/)?.[1] || 1180);

const browser = await chromium.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
});
const page = await browser.newPage({
  viewport: { width, height },
  deviceScaleFactor: 1,
});
await page.goto(pathToFileURL(svg).href);
await page.locator("svg").screenshot({ path: out });
await browser.close();
console.log(out);
