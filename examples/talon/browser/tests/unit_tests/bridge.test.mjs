import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { randomBytes } from 'node:crypto';
import { mkdtempSync, writeFileSync, chmodSync, rmSync } from 'node:fs';
import net from 'node:net';
import { once } from 'node:events';
import { tmpdir } from 'node:os';
import { Coordinator } from '../../coordinator.mjs';
import { createBridge, discover, Transport, validateSocketURL, readToken, MAX_BYTES } from '../../bridge.mjs';

const owner = { operator_id: 'op', provider: 'telegram', sender_id: 'sender', conversation_id: 'chat', run_id: 'run', background: false };
const make = () => new Coordinator({ operator: 'op', identities: { telegram: 'sender' } });
class WSS { close() {} }
class Socket extends EventEmitter {
  constructor() { super(); this.readyState = 1; this.sent = []; queueMicrotask(() => this.emit('open')); }
  send(value) { this.sent.push(JSON.parse(value)); }
  terminate() { this.readyState = 3; this.emit('close'); }
  close() { this.terminate(); }
}

test('runtime token and exact fixed browser endpoint', () => {
  const directory = mkdtempSync(`${tmpdir()}/jkb90-`);
  try {
    const path = `${directory}/token`;
    const token = randomBytes(32).toString('base64url');
    writeFileSync(path, token, { mode: 0o400 });
    assert.equal(readToken(path), token);
    chmodSync(path, 0o600);
    assert.throws(() => readToken(path), /invalid_token_file/);
  } finally { rmSync(directory, { recursive: true }); }
  assert.equal(validateSocketURL('ws://172.30.14.2:3000/'), 'ws://172.30.14.2:3000/');
  for (const url of ['ws://localhost:9222/devtools/browser/a', 'ws://172.30.14.2:3000/devtools/browser/a', 'ws://172.30.14.2:9222/devtools/browser/a?secret=x', 'ws://user@172.30.14.2:9222/devtools/browser/a']) assert.throws(() => validateSocketURL(url));
});

test('remapped IDs, session routing, sanitized CDP error and command timeout fencing', async () => {
  const c = make();
  await c.action({ action: 'acquire', owner, request_id: 'a' });
  const timers = [];
  const t = new Transport({ WebSocket: Socket, coordinator: c, lease: c.lease, discoverURL: async () => 'fixed', timer: (fn, ms) => { timers.push({ fn, ms }); return timers.length; }, clear() {} });
  c.lease.transport = t;
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
  await assert.rejects(nav, /lease_failed/);
  assert.equal(c.status().mode, 'FAILED');
});

test('HTTP route isolation, auth, dedup and late response fencing', async () => {
  const coordinator = make();
  const token = randomBytes(32).toString('base64url');
  const bridge = createBridge({ token, coordinator, WebSocket: Socket, WebSocketServer: WSS, discoverURL: async () => 'fixed', healthy: async () => true, controlHost: '127.0.0.1', viewerHost: '127.0.0.1', controlPort: 0, viewerPort: 0 });
  await bridge.start();
  try {
    const control = `http://127.0.0.1:${bridge.control.address().port}`;
    const viewer = `http://127.0.0.1:${bridge.viewer.address().port}`;
    assert.equal((await fetch(`${viewer}/health`)).status, 200);
    assert.equal((await fetch(`${viewer}/internal/browser/status`)).status, 404);
    assert.equal((await fetch(`${control}/internal/browser/status`)).status, 401);
    const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
    const lease = await (await fetch(`${control}/internal/browser/actions`, { method: 'POST', headers, body: JSON.stringify({ action: 'acquire', owner, request_id: 'a' }) })).json();
      for (const [payload, status] of [
        [{ action: 'nonsense', owner, request_id: 'bad' }, 400],
        [{ action: 'acquire', owner: { ...owner, sender_id: 'other' }, request_id: 'bad' }, 403],
        [{ action: 'acquire', owner: { ...owner, run_id: 'other' }, request_id: 'bad' }, 409],
      ]) {
        const response = await fetch(`${control}/internal/browser/actions`, { method: 'POST', headers, body: JSON.stringify(payload) });
        assert.equal(response.status, status);
      }
    const envelope = { ...lease, owner, request_id: 'eval', method: 'Runtime.evaluate', params: { expression: '2' } };
    const response = bridge.command(envelope);
    await new Promise(setImmediate);
    coordinator.lease.transport.message(Buffer.from('{"id":1,"result":{"value":2}}'), false);
    assert.deepEqual(await response, { result: { value: 2 } });
    assert.throws(() => bridge.command(envelope), /request_replayed/);
    assert.equal(coordinator.lease.transport.socket.sent.length, 1);
    const late = bridge.command({ ...envelope, request_id: 'late' });
    await new Promise(setImmediate);
    const handoff = coordinator.action({ ...lease, owner, action: 'handoff', request_id: 'h' });
    coordinator.lease.transport.message(Buffer.from('{"id":2,"result":{"secret":"not delivered"}}'), false);
    await assert.rejects(late, /stale_version|lease_fenced/);
    await handoff;
    await assert.rejects(async () => bridge.command(envelope), /stale_version|lease_fenced/);
  } finally { await bridge.close(); }
});

test('transport oversized incoming message fails closed', async () => {
  const c = make();
  await c.action({ action: 'acquire', owner, request_id: 'a' });
  const t = new Transport({ WebSocket: Socket, coordinator: c, lease: c.lease });
  c.lease.transport = t;
  t.message(Buffer.alloc(MAX_BYTES + 1), false);
  assert.equal(c.status().mode, 'FAILED');
});


test('production discovery validates listing and creates only empty default session, ignoring supplied endpoints', async (t) => {
  const calls = [];
  let listing = { sessions: [] };
  let created = { id: 'synthetic', status: 'live', websocketUrl: 'ws://attacker/' };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify(options.method === 'POST' ? created : listing));
  });
  assert.equal(await discover(), 'ws://172.30.14.2:3000/');
  assert.deepEqual(calls.map((call) => call.url), Array(2).fill('http://172.30.14.2:3000/v1/sessions'));
  assert.equal(calls[1].options.body, '{}');
  listing = { sessions: [{ id: 'synthetic', status: 'live' }] };
  calls.length = 0;
  assert.equal(await discover(), 'ws://172.30.14.2:3000/');
  assert.equal(calls.length, 1);
  listing = { sessions: [null] };
  await assert.rejects(discover(), /upstream_unavailable/);
  listing = { sessions: [] };
  created = {};
  await assert.rejects(discover(), /upstream_unavailable/);
});


test('connection cap is shared by control and viewer and enforced before HTTP headers', async () => {
  const bridge = createBridge({ token: randomBytes(32).toString('base64url'), coordinator: make(), WebSocket: Socket, WebSocketServer: WSS,
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
