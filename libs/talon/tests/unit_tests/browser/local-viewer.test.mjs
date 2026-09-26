import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createServer, request } from 'node:http';
import { test } from 'node:test';
import { Coordinator } from '../../../deepagents_talon/steel_runtime/coordinator.mjs';
import { createLocalViewer } from '../../../deepagents_talon/steel_runtime/local-viewer.mjs';

const origin = 'http://127.0.0.1:8765';
const token = 'x'.repeat(43);
const template = "<!doctype html><title>Steel</title><script>const singlePageMode = true; const baseWsUrl = 'ws://127.0.0.1:3000/v1/sessions/cast?pageIndex=0';</script>";

class Socket extends EventEmitter {
  constructor() {
    super();
    this.readyState = 1;
    this.bufferedAmount = 0;
    this.sent = [];
  }
  send(data, options, callback) { this.sent.push({ data: data.toString(), options }); callback?.(); }
  close() { this.readyState = 3; this.emit('close'); }
  terminate() { this.close(); }
}

async function fixture(t, options = {}) {
  const inputs = [];
  const createInput = (allowed) => {
    const input = { allowed, calls: [], closed: false, aborted: false,
      async command(method, params, sessionId) { this.calls.push({ method, params, sessionId }); return { sessionId: 'session' }; },
      async close() { this.closed = true; },
      abort() { this.aborted = true; },
    };
    inputs.push(input);
    return input;
  };
  const upstreams = [];
  const clients = [];
  class WebSocket extends Socket {
    constructor(url, config) { super(); this.url = url; this.config = config; upstreams.push(this); }
  }
  class WebSocketServer {
    constructor(config) { this.config = config; }
    handleUpgrade(req, socket, head, callback) { const client = new Socket(); clients.push(client); callback(client); }
    close(callback) { callback(); }
  }
  const coordinator = new Coordinator();
  const viewer = createLocalViewer({ coordinator, WebSocket, WebSocketServer, origin, token, fetchHTML: async () => template, createInput, ...options });
  const server = createServer(viewer.handler);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { await viewer.close(); await new Promise((resolve) => server.close(resolve)); });
  async function http(path, { method = 'GET', headers = {}, body = '' } = {}) {
    return new Promise((resolve, reject) => {
      const req = request({ hostname: '127.0.0.1', port: server.address().port, path, method, headers: { Host: '127.0.0.1:8765', ...headers } }, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: data }));
      });
      req.on('error', reject);
      req.end(body);
    });
  }
  async function login() {
    const result = await http('/auth/login', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/x-www-form-urlencoded' }, body: `token=${token}` });
    assert.equal(result.status, 303);
    const cookie = result.headers['set-cookie'][0].split(';')[0];
    const page = await http('/', { headers: { Cookie: cookie } });
    const csrf = /'X-CSRF-Token':'([^']+)'/.exec(page.body)[1];
    const headers = { Cookie: cookie, Origin: origin, 'X-CSRF-Token': csrf };
    return { headers, result, mutate: (path) => http(path, { method: 'POST', headers }), get: (path) => http(path, { headers }) };
  }
  function upgrade(auth, path = '/v1/sessions/cast?pageId=ABC', headers = {}) {
    const socket = { response: '', end(data) { this.response = data; } };
    viewer.upgrade({ method: 'GET', url: path, headers: { host: '127.0.0.1:8765', origin, cookie: auth.headers.Cookie, ...headers } }, socket, Buffer.alloc(0));
    return socket;
  }
  return { viewer, coordinator, http, login, upgrade, clients, upstreams, inputs };
}

const tick = () => new Promise((resolve) => setImmediate(resolve));
const mouse = { type: 'mouseEvent', pageId: 'ABC', event: { type: 'mouseMoved', x: 20, y: 30, button: 'none', modifiers: 0 } };

test('viewer authentication, exact routes and pause permissions', async (t) => {
  const f = await fixture(t);
  assert.equal((await f.http('/viewer')).status, 401);
  assert.equal((await f.http('/', { headers: { Host: 'evil.test' } })).status, 403);
  for (const path of ['/json', '/state?x=1', '/viewer?url=http://evil']) assert.equal((await f.http(path)).status, 404);
  const auth = await f.login();
  assert.match(auth.result.headers['set-cookie'][0], /HttpOnly; SameSite=Strict/);
  assert.equal((await auth.get('/viewer')).status, 200);
  for (const headers of [{ Cookie: auth.headers.Cookie, Origin: origin }, { ...auth.headers, Origin: 'http://evil.test' }]) {
    assert.equal((await f.http('/pause', { method: 'POST', headers })).status, 403);
  }
  assert.equal((await auth.mutate('/pause')).status, 200);
  const other = await f.login();
  assert.equal((await other.get('/viewer')).status, 200);
  assert.equal((await other.mutate('/pause')).status, 409);
  assert.equal((await other.mutate('/resume')).status, 403);
  assert.equal((await auth.mutate('/resume')).status, 200);
});

test('viewing during agent work sends no input; pause waits only for the active command', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  let finish;
  const command = f.coordinator.command('agent', () => new Promise((resolve) => { finish = resolve; }));
  await tick();
  assert.equal((await auth.get('/viewer')).status, 200);
  assert.equal(f.upgrade(auth).response, '');
  const frame = Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' }));
  f.upstreams[0].emit('message', frame, false);
  assert.equal(f.clients[0].sent.length, 1);
  f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
  assert.equal(f.inputs.length, 0);
  let paused = false;
  const pause = auth.mutate('/pause').then((result) => { paused = true; return result; });
  while (!f.coordinator.paused) await tick();
  assert.equal(paused, false);
  assert.throws(() => f.coordinator.command('agent', () => {}), /browser_paused/);
  finish({});
  await command;
  assert.equal((await pause).status, 200);
  f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
  await tick();
  assert.deepEqual(f.inputs[0].calls.map((call) => call.method), ['Target.attachToTarget', 'Input.dispatchMouseEvent']);
  assert.equal(f.upstreams[0].sent.length, 0);
});

test('resume waits for acknowledged input and drops queued input without closing the viewer', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/pause');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  const input = f.inputs[0];
  let ack;
  const command = input.command.bind(input);
  input.command = (method, ...args) => {
    const result = command(method, ...args);
    return method.startsWith('Input.') ? new Promise((resolve) => { ack = resolve; }) : result;
  };
  const emit = () => f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
  emit();
  while (!ack) await tick();
  emit();
  const resume = auth.mutate('/resume');
  while (input.allowed()) await tick();
  assert.equal(f.coordinator.paused, true);
  assert.equal(input.closed, false);
  emit();
  ack({});
  assert.equal((await resume).status, 200);
  assert.equal(input.calls.length, 2);
  assert.equal(input.closed, true);
  assert.equal(f.clients[0].readyState, 1);
  assert.equal(f.coordinator.paused, false);
  await f.coordinator.command('agent', () => {});
  await f.coordinator.release('agent');
});

test('only authenticated fixed cast routes accept sockets', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  for (const headers of [{ cookie: '' }, { origin: 'http://evil' }, { host: 'evil' }]) assert.match(f.upgrade(auth, undefined, headers).response, /403/);
  for (const path of ['/json', '/v1/sessions/cast?pageIndex=1', '/v1/sessions/cast?url=http://evil']) assert.match(f.upgrade(auth, path).response, /403/);
  assert.equal(f.upgrade(auth).response, '');
  assert.equal(f.upgrade(auth, '/v1/sessions/cast?tabInfo=true').response, '');
  assert.match(f.upgrade(auth).response, /403/);
});

test('viewer input validates schema and the current frame target', async (t) => {
  for (const message of [{ type: 'cdp', method: 'Runtime.evaluate' }, { ...mouse, pageId: 'other' }, { ...mouse, event: { ...mouse.event, x: -1 } }]) {
    const f = await fixture(t);
    const auth = await f.login();
    await auth.mutate('/pause');
    f.upgrade(auth);
    f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
    f.clients[0].emit('message', Buffer.from(JSON.stringify(message)), false);
    await tick();
    assert.equal(f.inputs[0].calls.length, 0);
    assert.equal(f.upstreams[0].sent.length, 0);
    assert.equal(f.clients[0].readyState, 3);
    assert.equal(f.coordinator.paused, true);
  }
});

test('cast limits disconnect oversized frames and slow clients', async (t) => {
  for (const pressure of [false, true]) {
    const f = await fixture(t);
    const auth = await f.login();
    f.upgrade(auth);
    if (pressure) f.clients[0].bufferedAmount = 2 * 1024 * 1024 + 1;
    f.upstreams[0].emit('message', pressure ? Buffer.from('{}') : Buffer.alloc(2 * 1024 * 1024 + 1), false);
    assert.equal(f.clients[0].readyState, 3);
    assert.equal(f.coordinator.paused, false);
  }
});

test('uncertain input prevents resuming automation', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/pause');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  f.inputs[0].command = async () => { throw new Error('unknown completion'); };
  f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
  await tick();
  assert.equal((await auth.mutate('/resume')).status, 409);
  assert.equal(f.coordinator.failed, true);
  assert.throws(() => f.coordinator.command('agent', () => {}), /browser_unavailable/);
});

test('disconnect stops interaction but leaves explicit resume available', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/pause');
  f.upgrade(auth);
  f.clients[0].close();
  await tick();
  assert.equal(f.inputs[0].allowed(), false);
  assert.equal(f.coordinator.paused, true);
  assert.equal((await auth.mutate('/resume')).status, 200);
  assert.equal(f.coordinator.paused, false);
});

test('logout during pause invalidates the session and resumes after drain', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  let finish;
  const command = f.coordinator.command('agent', () => new Promise((resolve) => { finish = resolve; }));
  await tick();
  const pause = auth.mutate('/pause');
  while (!f.coordinator.paused) await tick();
  const logout = auth.mutate('/auth/logout');
  while ((await auth.get('/state')).status !== 401) await tick();
  finish({});
  await command;
  assert.equal((await pause).status, 409);
  assert.equal((await logout).status, 200);
  assert.equal(f.coordinator.paused, false);
  assert.equal(f.inputs.length, 0);
});

test('session expiry closes viewing sockets and ends a healthy pause', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/pause');
  f.upgrade(auth);
  const now = Date.now;
  Date.now = () => now() + 1800001;
  try {
    assert.equal((await auth.get('/state')).status, 401);
    await tick();
    assert.equal(f.clients[0].readyState, 3);
    assert.equal(f.coordinator.paused, false);
  } finally { Date.now = now; }
});

test('unexpected viewer templates fail closed', async (t) => {
  const f = await fixture(t, { fetchHTML: async () => 'unexpected' });
  const auth = await f.login();
  assert.equal((await auth.get('/viewer')).status, 409);
});
