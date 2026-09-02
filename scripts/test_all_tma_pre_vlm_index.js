#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { pathToFileURL, fileURLToPath } = require("url");
const { chromium } = require("playwright");

async function main() {
  const indexPath = path.resolve(process.argv[2]);
  const screenshotDir = process.argv[3] ? path.resolve(process.argv[3]) : null;
  const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const browser = await chromium.launch({
    headless: true,
    executablePath: fs.existsSync(chromePath) ? chromePath : undefined,
  });
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));

  try {
    await page.goto(pathToFileURL(indexPath).href, { waitUntil: "load" });
    const result = await page.evaluate(() => ({
      title: document.title,
      h1: document.querySelector("h1")?.textContent || "",
      cards: document.querySelectorAll("a.card").length,
      rows: document.querySelectorAll("tbody tr").length,
      links: [...document.querySelectorAll("a.card")].map((node) => node.href),
      overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    }));
    if (result.cards !== 10) errors.push(`expected 10 TMA cards, found ${result.cards}`);
    if (result.rows !== 10) errors.push(`expected 10 summary rows, found ${result.rows}`);
    if (result.overflow) errors.push("desktop index has horizontal overflow");
    for (const href of result.links) {
      const target = fileURLToPath(href);
      if (!fs.existsSync(target)) errors.push(`missing report target: ${target}`);
    }

    if (screenshotDir) {
      fs.mkdirSync(screenshotDir, { recursive: true });
      await page.screenshot({ path: path.join(screenshotDir, "index_desktop.png"), fullPage: false });
    }
    await page.locator("a.card").first().click();
    await page.waitForLoadState("load");
    const linkedHeading = await page.locator("h1").textContent();
    if (!linkedHeading?.startsWith("TMA07:")) errors.push(`first report link opened unexpected page: ${linkedHeading}`);

    await page.goto(pathToFileURL(indexPath).href, { waitUntil: "load" });
    await page.setViewportSize({ width: 390, height: 844 });
    const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
    if (mobileOverflow) errors.push("mobile index has horizontal overflow");
    if (screenshotDir) {
      await page.screenshot({ path: path.join(screenshotDir, "index_mobile.png"), fullPage: false });
    }

    const output = { index: indexPath, ...result, linkedHeading, errors, status: errors.length ? "failed" : "passed" };
    process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
    if (errors.length) process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
