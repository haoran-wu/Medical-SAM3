#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");

async function testReport(browser, reportPath, screenshotDir) {
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));

  await page.goto(pathToFileURL(reportPath).href, { waitUntil: "load" });
  await page.waitForFunction(() => document.images.length > 0 && [...document.images].every((image) => image.complete));

  const result = await page.evaluate(() => {
    const brokenImages = [...document.images]
      .filter((image) => image.getAttribute("src"))
      .filter((image) => image.naturalWidth === 0 || image.naturalHeight === 0)
      .map((image) => image.alt || image.id || "unnamed image");
    const candidateButtons = [...document.querySelectorAll(".candidate-button")];
    const sourceFilter = document.getElementById("source-filter");
    const moduleFilter = document.getElementById("module-filter");
    return {
      title: document.title,
      h1: document.querySelector("h1")?.textContent || "",
      imageCount: document.images.length,
      brokenImages,
      moduleCards: document.querySelectorAll(".module-card").length,
      candidateButtons: candidateButtons.length,
      sourceOptions: sourceFilter ? [...sourceFilter.options].map((option) => option.value) : [],
      moduleOptions: moduleFilter ? [...moduleFilter.options].map((option) => option.value) : [],
      initialCandidate: document.getElementById("candidate-title")?.textContent || "",
      bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    };
  });

  if (result.brokenImages.length) errors.push(`broken images: ${result.brokenImages.join(", ")}`);
  if (result.moduleCards !== 9) errors.push(`expected 9 module cards, found ${result.moduleCards}`);
  if (!result.candidateButtons) errors.push("candidate inventory is empty");
  if (!result.initialCandidate) errors.push("candidate viewer did not initialize");
  if (result.bodyOverflow) errors.push("desktop page has horizontal overflow");

  for (const source of result.sourceOptions.filter((value) => value !== "all")) {
    await page.selectOption("#source-filter", source);
    const count = await page.locator(".candidate-button").count();
    if (!count) errors.push(`source filter ${source} returned no candidates`);
  }
  await page.selectOption("#source-filter", "all");

  for (const moduleValue of result.moduleOptions.filter((value) => !["all", "none"].includes(value))) {
    await page.selectOption("#module-filter", moduleValue);
    const count = await page.locator(".candidate-button").count();
    if (!count) errors.push(`module filter ${moduleValue} returned no candidates`);
  }
  await page.selectOption("#module-filter", "all");

  await page.locator(".module-card .image-button").first().click();
  if (!(await page.locator("#modal").evaluate((node) => node.classList.contains("open")))) {
    errors.push("GeneMap modal did not open");
  }
  await page.locator("#modal-close").click();

  if (screenshotDir) {
    fs.mkdirSync(screenshotDir, { recursive: true });
    const stem = path.basename(reportPath, path.extname(reportPath));
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(screenshotDir, `${stem}_desktop.png`), fullPage: false });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: "load" });
    await page.evaluate(() => window.scrollTo(0, 0));
    const mobileLayout = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      width: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      offenders: [...document.querySelectorAll("body *")]
        .filter((node) => node.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
        .slice(0, 12)
        .map((node) => ({
          tag: node.tagName,
          id: node.id,
          className: typeof node.className === "string" ? node.className : "",
          right: Math.round(node.getBoundingClientRect().right),
          text: (node.textContent || "").trim().slice(0, 80),
        })),
    }));
    if (mobileLayout.overflow) errors.push(`mobile page has horizontal overflow: ${JSON.stringify(mobileLayout)}`);
    await page.screenshot({ path: path.join(screenshotDir, `${stem}_mobile.png`), fullPage: false });
  }

  await page.close();
  return { report: reportPath, ...result, errors, status: errors.length ? "failed" : "passed" };
}

async function main() {
  const reportPaths = process.argv.slice(2).filter((value) => !value.startsWith("--"));
  const screenshotArg = process.argv.find((value) => value.startsWith("--screenshots="));
  const screenshotDir = screenshotArg ? path.resolve(screenshotArg.split("=", 2)[1]) : null;
  if (!reportPaths.length) throw new Error("Usage: test_all_tma_pre_vlm_report.js [--screenshots=DIR] REPORT.html [...]");

  const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const browser = await chromium.launch({
    headless: true,
    executablePath: fs.existsSync(chromePath) ? chromePath : undefined,
  });
  const results = [];
  try {
    for (const reportPath of reportPaths) {
      results.push(await testReport(browser, path.resolve(reportPath), screenshotDir));
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(`${JSON.stringify(results, null, 2)}\n`);
  if (results.some((result) => result.status !== "passed")) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
