import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { randomBytes } from 'node:crypto';
import { mkdtempSync, writeFileSync, chmodSync, rmSync } from 'node:fs';
import net from 'node:net';
import http from 'node:http';
import { once } from 'node:events';
import { tmpdir } from 'node:os';
import { Coordinator } from '../../../deepagents_talon/steel_runtime/coordinator.mjs';
import { createBridge, discover, Transport, readToken, MAX_BYTES } from '../../../deepagents_talon/steel_runtime/bridge.mjs';

const make = () => new Coordinator();
class Socket extends EventEmitter {
  constructor() { super(); this.readyState = 1; this.sent = []; queueMicrotask(() => this.emit('open')); }
  send(value) { this.sent.push(JSON.parse(value)); }
  terminate() { this.readyState = 3; this.emit('close'); }
  close() { this.terminate(); }
}

test('runtime token requires owner-only permissions', () => {
  const directory = mkdtempSync(`${tmpdir()}/jkb90-`);
  try {
    const path = `${directory}/token`;
    const token = randomBytes(32).toString('base64url');
    writeFileSync(path, token, { mode: 0o400 });
    assert.equal(readToken(path), token);
    chmodSync(path, 0o600);
    assert.throws(() => readToken(path), /invalid_token_file/);
  } finally { rmSync(directory, { recursive: true }); }
});

test('remapped IDs, session routing, sanitized CDP error and command timeout fencing', async () => {
  const c = make();
  c.acquire('run');
  const timers = [];
  const t = new Transport({ WebSocket: Socket, allowed: () => c.allowed(c.run), onFailure: () => c.fail(), discoverURL: async () => 'fixed', timer: (fn, ms) => { timers.push({ fn, ms }); return timers.length; }, clear() {} });
  c.run.transport = t;
  const pending = t.command('Runtime.evaluate', { expression: '1+1' }, 'session');
  await new Promise(setImmediate);
  assert.deepEqual(t.socket.sent[0], { id: 1, method: 'Runtime.evaluate', params: { expression: '1+1' }, sessionId: 'session' });
  t.message(Buffer.from('{"id":1,"result":{"value":2}}'), false);
  assert.deepEqual(await pending, { value: 2 });
  const bad = t.command('Anything.allowed', {});
  await new Promise(setImmediate);
  t.message(Buffer.from('{"id":2,"error":{"message":"secret upstream"}}'), false);
  await assert.rejects(bad, /^Error: cdp_error$/);
  const nav = t.command('Page.navigate', { url: 'data:text/html,test' });
  await new Promise(setImmediate);
  assert.equal(timers.at(-1).ms, 30000);
  timers.at(-1).fn();
  await assert.rejects(nav, /browser_unavailable/);
  assert.equal(c.failed, true);
});

test('HTTP route isolation, authentication and command execution', async () => {
  const coordinator = make();
  const token = randomBytes(32).toString('base64url');
  const bridge = createBridge({ token, coordinator, WebSocket: Socket, discoverURL: async () => 'fixed', healthy: async () => true, controlHost: '127.0.0.1', viewerHost: '127.0.0.1', controlPort: 0, viewerPort: 0 });
  await bridge.start();
  try {
    const control = `http://127.0.0.1:${bridge.control.address().port}`;
    const viewer = `http://127.0.0.1:${bridge.viewer.address().port}`;
    assert.equal((await fetch(`${viewer}/health`)).status, 200);
    assert.equal((await fetch(`${viewer}/internal/browser/status`)).status, 404);
    assert.equal((await fetch(`${control}/internal/browser/status`)).status, 401);
    const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
    assert.equal((await fetch(`${control}/internal/browser/status`, { headers: { ...headers, Origin: 'https://example.com' } })).status, 403);
    const rebound = await new Promise((resolve, reject) => {
      http.get(`${control}/internal/browser/status`, { headers: { ...headers, Host: 'attacker.example' } }, (response) => { response.resume(); resolve(response.statusCode); }).on('error', reject);
    });
    assert.equal(rebound, 403);
    const envelope = { run_id: 'run', method: 'Runtime.evaluate', params: { expression: '2' } };
    const response = bridge.command(envelope);
    await new Promise(setImmediate);
    coordinator.run.transport.message(Buffer.from('{"id":1,"result":{"value":2}}'), false);
    assert.deepEqual(await response, { result: { value: 2 } });
    assert.throws(() => bridge.command({ ...envelope, run_id: 'other' }), /browser_busy/);
    assert.throws(() => bridge.command({ ...envelope, run_id: null }), /invalid_run/);
    const released = await fetch(`${control}/internal/browser/release`, {
      method: 'POST', headers, body: JSON.stringify({ run_id: 'run' }),
    });
    assert.equal(released.status, 200);
    assert.equal(coordinator.run, null);
  } finally { await bridge.close(); }
});

test('transport oversized incoming message fails closed', async () => {
  const c = make();
  c.acquire('run');
  const t = new Transport({ WebSocket: Socket, allowed: () => c.allowed(c.run), onFailure: () => c.fail() });
  c.run.transport = t;
  t.message(Buffer.alloc(MAX_BYTES + 1), false);
  assert.equal(c.failed, true);
});


test('production discovery validates listing and creates only empty default session, ignoring supplied endpoints', async (t) => {
  const calls = [];
  let listing = { sessions: [] };
  let created = { id: 'synthetic', status: 'live', websocketUrl: 'ws://attacker/' };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify(options.method === 'POST' ? created : listing));
  });
  assert.equal(await discover(), 'ws://127.0.0.1:3000/');
  assert.deepEqual(calls.map((call) => call.url), Array(2).fill('http://127.0.0.1:3000/v1/sessions'));
  assert.equal(calls[1].options.body, '{}');
  listing = { sessions: [{ id: 'synthetic', status: 'live' }] };
  calls.length = 0;
  assert.equal(await discover(), 'ws://127.0.0.1:3000/');
  assert.equal(calls.length, 1);
  listing = { sessions: [null] };
  await assert.rejects(discover(), /upstream_unavailable/);
  listing = { sessions: [] };
  created = {};
  await assert.rejects(discover(), /upstream_unavailable/);
});


test('connection cap is shared by control and viewer and enforced before HTTP headers', async () => {
  const bridge = createBridge({ token: randomBytes(32).toString('base64url'), coordinator: make(), WebSocket: Socket,
    controlHost: '127.0.0.1', viewerHost: '127.0.0.1', controlPort: 0, viewerPort: 0 });
  const sockets = [];
  await bridge.start();
  try {
    for (let i = 0; i < 64; i++) {
      const socket = net.connect((i % 2 ? bridge.control : bridge.viewer).address().port, '127.0.0.1');
      sockets.push(socket);
      await once(socket, 'connect');
    }
    const rejected = net.connect(bridge.control.address().port, '127.0.0.1');
    sockets.push(rejected);
    await once(rejected, 'close');
    assert.equal(sockets.slice(0, 64).every((socket) => !socket.destroyed), true);
  } finally {
    sockets.forEach((socket) => socket.destroy());
    await bridge.close();
  }
});
