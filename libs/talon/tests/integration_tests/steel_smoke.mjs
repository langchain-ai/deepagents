import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const [source, origin, login, phase] = process.argv.slice(2);
const require = createRequire(path.join(source, "package.json"));
const puppeteer = require("puppeteer-core");
const WebSocket = require("ws");
const browser = await puppeteer.connect({ browserWSEndpoint: origin.replace("http", "ws") + "/" });
try {
  const page = (await browser.pages())[0];
  if (phase === "login") {
    await page.goto("https://example.com", { timeout: 20000 });
    assert.equal(await page.title(), "Example Domain");
    await page.goto(login + "/login");
  }
  await page.goto(login + "/private");
  assert.equal(await page.title(), "Synthetic account");
  await new Promise((resolve, reject) => {
    const ws = new WebSocket(origin.replace("http", "ws") + "/v1/sessions/cast?pageIndex=0");
    const timer = setTimeout(() => {
      ws.terminate();
      reject(new Error("Steel viewer frame timeout"));
    }, 10000);
    ws.on("error", (error) => { clearTimeout(timer); reject(error); });
    ws.on("message", (raw) => {
      if (JSON.parse(raw).data) {
        clearTimeout(timer);
        ws.close();
        resolve();
      }
    });
  });
} finally {
  await browser.disconnect();
}
