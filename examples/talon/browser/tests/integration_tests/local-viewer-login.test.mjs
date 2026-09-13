import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import test from 'node:test';
import { Coordinator } from '../../coordinator.mjs';
import { createLocalViewer } from '../../local-viewer.mjs';

test('Chromium form and fragment login preserve strict origin checks', {
  skip: process.env.TALON_TEST_VIEWER_LOGIN !== '1', timeout: 30000,
}, async (t) => {
  const require = createRequire('/app/api/package.json');
  const { WebSocket, WebSocketServer } = require('ws');
  const puppeteer = require('puppeteer-core');
  const origin = 'http://127.0.0.1:8765';
  const token = randomBytes(32).toString('base64url');
  const coordinator = new Coordinator({ operator: 'test', identities: { test: 'sender' } });
  const viewer = createLocalViewer({ coordinator, WebSocket, WebSocketServer, origin, token });
  const server = createServer(viewer.handler);
  let browser;
  t.after(async () => {
    await browser?.close();
    await viewer.close();
    await new Promise((resolve) => server.close(resolve));
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(8765, '127.0.0.1', resolve);
  });
  browser = await puppeteer.launch({ executablePath: '/usr/lib/chromium/chromium',
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const page = await browser.newPage();
  page.setDefaultTimeout(5000);
  let tokenInRequestURL = false;
  page.on('request', (request) => { tokenInRequestURL ||= request.url().includes('token='); });

  await page.goto(origin);
  await page.type('input[name=token]', token);
  await Promise.all([page.waitForNavigation(), page.click('button')]);
  assert.equal(await page.title(), 'Local browser control');
  await page.click('#logout');
  await page.waitForSelector('input[name=token]');

  for (const fragment of ['token=invalid', `token=${'x'.repeat(43)}`, `token=${token}&extra=1`]) {
    await page.goto(`${origin}/#${fragment}`);
    await page.waitForFunction(() => document.querySelector('#login-message')?.textContent.includes('Invalid'));
    assert.ok(page.url() === `${origin}/`, 'invalid link fragment must be removed');
    assert.equal(await page.title(), 'Local browser login');
  }
  await page.goto(`${origin}/#token=${token}`);
  await page.waitForSelector('#take');
  assert.ok(page.url() === `${origin}/`, 'login fragment must be removed');
  assert.equal(tokenInRequestURL, false);
  await page.goto(`${origin}/#token=${token}`);
  await page.waitForFunction(() => location.hash === '');
  await page.click('#logout');
  await page.waitForSelector('input[name=token]');
});
