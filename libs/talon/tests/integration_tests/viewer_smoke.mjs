import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(path.join(process.argv[2], 'package.json'));
const puppeteer = require('puppeteer-core');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const { origin, token, chrome } = JSON.parse(input);
const browser = await puppeteer.launch({ executablePath: chrome, headless: true });
process.on('SIGTERM', async () => { await browser.close(); process.exit(1); });
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(15000);
  let leaked = false;
  page.on('request', (request) => { leaked ||= request.url().includes('token='); });
  await page.goto(`${origin}/#token=${token}`);
  await page.waitForSelector('#automation');
  assert.equal(page.url(), `${origin}/`);
  assert.equal(leaked, false);
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Watching the shared browser.');
  const iframe = await page.waitForSelector('iframe[src="/viewer"]');
  const frame = await iframe.contentFrame();
  await frame.waitForFunction(() => document.querySelector('canvas')?.getContext('2d').getImageData(0, 0, 1, 1).data[3] > 0);
  await page.click('#automation');
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Automation paused. You can interact.');
  const canvas = await frame.$('canvas');
  const box = await canvas.boundingBox();
  assert.ok(box);
  // The synthetic input occupies x=20..420, y=20..100 in Steel's 1920x1080 viewport.
  await page.mouse.click(box.x + box.width * 100 / 1920, box.y + box.height * 50 / 1080, { clickCount: 3 });
  await page.keyboard.type('human works', { delay: 30 });
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel({ deltaY: 400 });
  // Round-trip through the shared input queue before resuming automation.
  await new Promise(resolve => setTimeout(resolve, 500));
  await page.click('#automation');
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Watching the shared browser.');
  await page.click('#logout');
  await page.waitForSelector('input[name=token]');
  process.stdout.write('resumed\n');
} catch {
  process.stdout.write('viewer_failed\n');
  process.exitCode = 1;
} finally {
  await browser.close();
}
