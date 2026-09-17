import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createServer, request } from 'node:http';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
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
    this.autoClose = true;
  }
  send(data, options, callback) { this.sent.push({ data: data.toString(), options }); callback?.(); }
  finish() { this.readyState = 3; this.emit('close'); }
  close() { this.readyState = 2; if (this.autoClose) queueMicrotask(() => this.finish()); }
  terminate() { this.finish(); }
}

async function fixture(t, options = {}) {
  const inputs = [];
  const createInput = (lease, allowed) => {
    const input = { lease, allowed, calls: [], closed: false, aborted: false,
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
  const coordinator = new Coordinator({ operator: 'operator', identities: { slack: 'person' } });
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

test('configuration rejects non-loopback origins and weak launch secrets', () => {
  for (const config of [{ origin: 'http://localhost:8765', token }, { origin, token: 'short' }, { origin: `${origin}/`, token }, { origin: 'http://127.0.0.1:65536', token }, { origin: 'http://127.0.0.1:0', token }]) {
    assert.throws(() => createLocalViewer(config), /invalid_local_viewer_config/);
  }
});

test('login, exact HTTP allowlist, host/origin/CSRF and private state', async (t) => {
  const f = await fixture(t);
  assert.match((await f.http('/')).body, /type="password"/);
  assert.equal((await f.http('/state')).status, 401);
  for (const path of ['/json', '/v1/sessions/debug', '/viewer?url=http://evil', '/state?x=1', '//state', '/%73tate', `/?token=${token}`]) assert.equal((await f.http(path)).status, 404);
  assert.equal((await f.http('/', { headers: { Host: 'evil.test' } })).status, 403);
  assert.equal((await f.http('/auth/login', { method: 'POST', body: `token=${token}` })).status, 403);
  assert.equal((await f.http('/auth/login', { method: 'POST', headers: { Origin: 'null', 'Content-Type': 'application/x-www-form-urlencoded' }, body: `token=${token}` })).status, 403);
  const auth = await f.login();
  assert.match(auth.result.headers['set-cookie'][0], /HttpOnly; SameSite=Strict/);
  assert.equal(auth.result.headers['cache-control'], 'no-store');
  assert.match(auth.result.headers['content-security-policy'], /img-src data:/);
  assert.deepEqual(JSON.parse((await auth.get('/state')).body), { available: true, owned: false, controlling: false });
  assert.equal((await f.http('/take', { method: 'POST', headers: { Cookie: auth.headers.Cookie, Origin: origin } })).status, 403);
  assert.equal((await f.http('/take', { method: 'POST', headers: { ...auth.headers, Origin: 'http://evil.test' } })).status, 403);
  assert.equal((await auth.get('/viewer')).status, 403);
  assert.equal((await auth.mutate('/take')).status, 200);
  assert.equal(f.coordinator.lease.mode, 'HUMAN');
  assert.equal((await auth.get('/viewer')).body, template.replace('ws://127.0.0.1:3000', 'ws://127.0.0.1:8765'));
  const other = await f.login();
  assert.deepEqual(JSON.parse((await other.get('/state')).body), { available: false, owned: false, controlling: false });
  assert.equal((await other.get('/viewer')).status, 403);
  assert.equal((await other.mutate('/take')).status, 409);
  assert.equal((await other.mutate('/release')).status, 403);
  assert.equal((await auth.mutate('/release')).status, 200);
  assert.equal(f.coordinator.lease, null);
});

test('Take never steals an agent lease', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  const owner = { operator_id: 'operator', provider: 'slack', sender_id: 'person', conversation_id: 'agent-conversation', run_id: 'agent-run', background: false };
  await f.coordinator.action({ action: 'acquire', owner, request_id: 'acquire' });
  const lease = f.coordinator.lease;
  assert.equal((await auth.mutate('/take')).status, 409);
  assert.equal(f.coordinator.lease, lease);
  assert.equal(lease.mode, 'AGENT');
  assert.doesNotMatch((await auth.get('/state')).body, /agent-|lease_id|generation/);
});

test('template rewrite fails closed on unexpected stock source', async (t) => {
  const f = await fixture(t, { fetchHTML: async () => template.replace('127.0.0.1', 'localhost') });
  const auth = await f.login();
  await auth.mutate('/take');
  assert.equal((await auth.get('/viewer')).status, 409);
});

test('upgrades allow only fixed cast targets and two socket roles', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  assert.match(f.upgrade(auth).response, /403/);
  await auth.mutate('/take');
  for (const path of ['/json', '/v1/sessions/cast', '/v1/sessions/cast?pageIndex=1', '/v1/sessions/cast?url=http://evil', '/v1/sessions/cast?pageId=ABC&tabInfo=true', '/v1/sessions/cast?pageId=../json']) assert.match(f.upgrade(auth, path).response, /403/);
  assert.match(f.upgrade(auth, undefined, { origin: 'http://evil' }).response, /403/);
  assert.match(f.upgrade(auth, undefined, { host: 'evil' }).response, /403/);
  assert.equal(f.upgrade(auth, '/v1/sessions/cast?tabInfo=true').response, '');
  assert.equal(f.upgrade(auth).response, '');
  assert.equal(f.upstreams[0].url, 'ws://127.0.0.1:3000/v1/sessions/cast?tabInfo=true');
  assert.equal(f.upstreams[1].url, 'ws://127.0.0.1:3000/v1/sessions/cast?pageId=ABC');
  assert.match(f.upgrade(auth, '/v1/sessions/cast?pageIndex=0').response, /403/);
  assert.match(f.upgrade(auth, '/v1/sessions/cast?tabInfo=true').response, /403/);
  assert.equal(f.upstreams[1].config.maxPayload, 2 * 1024 * 1024);
});

test('Release waits for BOTH close events before coordinator stop and never resumes agent', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  const client = f.clients[0], upstream = f.upstreams[0];
  client.autoClose = upstream.autoClose = false;
  const lease = f.coordinator.lease;
  let stops = 0;
  const original = f.coordinator.stop.bind(f.coordinator);
  f.coordinator.stop = (...args) => { stops += 1; return original(...args); };
  const release = auth.mutate('/release');
  while (client.readyState !== 2) await tick();
  assert.equal(stops, 0);
  assert.equal(lease.mode, 'PAUSED');
  client.emit('message', Buffer.from(JSON.stringify(mouse)), false);
  assert.equal(upstream.sent.length, 0);
  client.finish();
  await tick();
  assert.equal(stops, 0);
  upstream.finish();
  assert.equal((await release).status, 200);
  assert.equal(stops, 1);
  assert.equal(f.coordinator.lease, null);
});

test('disconnect pauses retained ownership until explicit clean Release', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  f.clients[0].finish();
  await tick();
  assert.equal(f.coordinator.lease.mode, 'PAUSED');
  assert.equal(f.coordinator.failedLatch, false);
  assert.deepEqual(JSON.parse((await auth.get('/state')).body), { available: false, owned: true, controlling: false });
  assert.match(f.upgrade(auth).response, /403/);
  assert.equal((await auth.mutate('/release')).status, 200);
});

test('strict input schemas dispatch ACKed CDP actions only', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  const messages = [mouse, { ...mouse, event: { type: 'mouseWheel', x: 20, y: 30, button: 'none', modifiers: 0, deltaX: 0, deltaY: 100 } }, { type: 'keyEvent', pageId: 'ABC', event: { type: 'char', key: 'a', code: 'KeyA', keyCode: 65, text: 'a' } }, { type: 'keyEvent', pageId: 'ABC', event: { type: 'keyDown', key: 'a', code: 'KeyA', keyCode: 65, text: 'a' } }];
  for (const message of messages) f.clients[0].emit('message', Buffer.from(JSON.stringify(message)), false);
  await tick();
  assert.equal(f.upstreams[0].sent.length, 0);
  assert.deepEqual(f.inputs[0].calls[0], { method: 'Target.attachToTarget', params: { targetId: 'ABC', flatten: true }, sessionId: undefined });
  assert.deepEqual(f.inputs[0].calls.slice(1).map((call) => call.method), ['Input.dispatchMouseEvent', 'Input.dispatchMouseEvent', 'Input.dispatchKeyEvent', 'Input.dispatchKeyEvent']);
  assert.equal(f.inputs[0].calls[1].params.buttons, 0);
  assert.equal(f.inputs[0].calls[3].params.windowsVirtualKeyCode, 65);
  assert.equal(f.inputs[0].calls[3].sessionId, 'session');
});

for (const [name, message] of Object.entries({
  CDP: { id: 1, method: 'Runtime.evaluate', params: {} },
  closeTab: { type: 'closeTab', pageId: 'ABC' },
  selection: { type: 'getSelectedText', pageId: 'ABC' },
  clipboard: { type: 'clipboardWrite', pageId: 'ABC', event: { text: 'secret' } },
  navigation: { type: 'navigation', pageId: 'ABC', event: { action: 'back' } },
  url: { type: 'navigation', pageId: 'ABC', event: { url: 'https://example.com' } },
  wrongTarget: { ...mouse, pageId: 'DEF' },
  extraField: { ...mouse, command: 'bad' },
  extraEvent: { ...mouse, event: { ...mouse.event, command: 'bad' } },
  coordinates: { ...mouse, event: { ...mouse.event, x: -1 } },
  modifiers: { ...mouse, event: { ...mouse.event, modifiers: 16 } },
})) {
  test(`denies ${name} and fences input`, async (t) => {
    const f = await fixture(t);
    const auth = await f.login();
    await auth.mutate('/take');
    f.upgrade(auth);
    f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
    f.clients[0].emit('message', Buffer.from(JSON.stringify(message)), false);
    assert.equal(f.upstreams[0].sent.length, 0);
    assert.equal(f.coordinator.lease.mode, 'PAUSED');
  });
}

test('rate burst, frame cap, payload caps and backpressure', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  const frame = Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'aGVsbG8=' }));
  f.upstreams[0].emit('message', frame, false);
  f.upstreams[0].emit('message', frame, false);
  assert.equal(f.clients[0].sent.length, 1);
  for (let index = 0; index < 241; index += 1) f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
  assert.equal(f.upstreams[0].sent.length, 0);
  assert.equal(f.inputs[0].calls.length, 0);
  assert.equal(f.coordinator.lease.mode, 'FAILED');
});

for (const condition of ['binary', 'oversizedInput', 'oversizedFrame', 'clientPressure', 'upstreamPressure', 'discoveryInput', 'malformed']) {
  test(`fences ${condition}`, async (t) => {
    const f = await fixture(t);
    const auth = await f.login();
    await auth.mutate('/take');
    f.upgrade(auth, condition === 'discoveryInput' ? '/v1/sessions/cast?tabInfo=true' : undefined);
    if (condition === 'clientPressure') f.clients[0].bufferedAmount = 2 * 1024 * 1024 + 1;
    if (condition === 'upstreamPressure') f.upstreams[0].bufferedAmount = 16 * 1024 + 1;
    if (['oversizedFrame', 'clientPressure'].includes(condition)) f.upstreams[0].emit('message', condition === 'oversizedFrame' ? Buffer.alloc(2 * 1024 * 1024 + 1) : Buffer.from('{}'), false);
    else f.clients[0].emit('message', condition === 'oversizedInput' ? Buffer.alloc(16 * 1024 + 1) : condition === 'malformed' ? Buffer.from('{') : Buffer.from(JSON.stringify(mouse)), condition === 'binary');
    assert.equal(f.coordinator.lease.mode, 'PAUSED');
    assert.equal(f.upstreams[0].sent.length, 0);
  });
}

test('idle session expiry releases healthy ownership', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  const now = Date.now;
  Date.now = () => now() + 600001;
  try {
    assert.equal((await auth.get('/state')).status, 401);
    await tick();
    assert.equal(f.coordinator.status().mode, 'IDLE');
    assert.equal(f.coordinator.failedLatch, false);
  } finally { Date.now = now; }
});

test('unconfirmed socket closure fails closed rather than releasing lease', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  f.upstreams[0].autoClose = false;
  assert.equal((await auth.mutate('/release')).status, 409);
  assert.equal(f.coordinator.lease.mode, 'FAILED');
  assert.equal(f.coordinator.failedLatch, true);
});

test('absolute session expiry fences sockets even with HTTP activity', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  const now = Date.now;
  Date.now = () => now() + 1800001;
  try {
    assert.equal((await auth.get('/state')).status, 401);
    await tick();
    assert.equal(f.coordinator.status().mode, 'IDLE');
    assert.equal(f.coordinator.failedLatch, false);
    assert.equal(f.clients[0].readyState, 3);
    assert.equal(f.upstreams[0].readyState, 3);
    f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
    assert.equal(f.upstreams[0].sent.length, 0);
  } finally { Date.now = now; }
});


test('Release waits for input ACK, drops queued input, closes transport and permits reuse', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
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
  let released = false;
  const release = auth.mutate('/release').then((result) => { released = true; return result; });
  while (f.coordinator.lease.mode === 'HUMAN') await tick();
  assert.equal(input.allowed(), false);
  emit();
  await tick();
  assert.equal(released, false);
  assert.equal(input.closed, false);
  assert.equal(f.upstreams[0].readyState, 1);
  ack({});
  assert.equal((await release).status, 200);
  assert.equal(input.calls.length, 2);
  assert.equal(input.closed, true);
  assert.equal(f.coordinator.failedLatch, false);
  assert.equal((await auth.mutate('/take')).status, 200);
  assert.notEqual(f.inputs[1], input);
  assert.equal(input.allowed(), false);
});

for (const failure of ['missing', 'error', 'timeout']) {
  test(`input ${failure} fails closed`, async (t) => {
    const f = await fixture(t, failure === 'missing' ? { createInput: null } : {});
    const auth = await f.login();
    await auth.mutate('/take');
    f.upgrade(auth);
    f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
    if (failure !== 'missing') f.inputs[0].command = () => failure === 'timeout' ? new Promise(() => {}) : Promise.reject(new Error('CDP failure'));
    f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
    await tick();
    assert.equal((await auth.mutate('/release')).status, 409);
    assert.equal(f.coordinator.failedLatch, true);
    if (failure !== 'missing') assert.equal(f.inputs[0].aborted, true);
  });
}

test('logout requires CSRF and permanently invalidates session and closes both sockets', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  assert.equal((await f.http('/auth/logout', { method: 'POST', headers: { Cookie: auth.headers.Cookie, Origin: origin } })).status, 403);
  const result = await auth.mutate('/auth/logout');
  assert.equal(result.status, 200);
  assert.match(result.headers['set-cookie'][0], /Max-Age=0/);
  assert.equal((await auth.get('/state')).status, 401);
  assert.equal(f.clients[0].readyState, 3);
  assert.equal(f.upstreams[0].readyState, 3);
  assert.equal(f.coordinator.failedLatch, false);
  assert.equal(f.coordinator.status().mode, 'IDLE');
  const next = await f.login();
  assert.equal((await next.mutate('/take')).status, 200);
});

test('login limited to five attempts per minute and fifty per process', async (t) => {
  const f = await fixture(t);
  const now = Date.now;
  let time = now();
  Date.now = () => time;
  const attempt = () => f.http('/auth/login', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'token=wrong' });
  try {
    for (let minute = 0; minute < 10; minute += 1) {
      for (let i = 0; i < 5; i += 1) assert.equal((await attempt()).status, 403);
      assert.equal((await attempt()).status, 429);
      time += 60001;
    }
    assert.equal((await attempt()).status, 429);
  } finally { Date.now = now; }
});


test('pinned Steel template renders a single interactive page without denied chrome controls', { skip: !process.env.STEEL_VIEWER_TEMPLATE }, async (t) => {
  const source = readFileSync(process.env.STEEL_VIEWER_TEMPLATE, 'utf8');
  assert.equal(createHash('sha256').update(source).digest('hex'), '2de2e9328a568d00df9dd8d11b36fde58b3d1da97e71f9cacb863530a215064d');
  const values = {
    theme: 'dark',
    singlePageMode: 'true',
    interactive: 'true',
    wsUrl: 'ws://127.0.0.1:3000/v1/sessions/cast?pageIndex=0',
    "singlePageMode ? 'data-single-page-mode=\"true\"' : ''": 'data-single-page-mode="true"',
    'interactive ? "pointer" : "default"': 'pointer',
    'interactive ? "var(--tab-hover-bg)" : "transparent"': 'var(--tab-hover-bg)',
    'dimensions?.width || 1920': '1920',
    'dimensions?.width || 1080': '1080',
  };
  const html = source.replace(/<% if \(showControls\) { %>[\s\S]*?<% } %>[\s\S]*?<% } %>/, '').replace(/<%= (.*?) %>/g, (_, expression) => {
    assert.ok(Object.hasOwn(values, expression), expression);
    return values[expression];
  });
  assert.doesNotMatch(html, /<%/);
  const f = await fixture(t, { fetchHTML: async () => html });
  const auth = await f.login();
  await auth.mutate('/take');
  const result = await auth.get('/viewer');
  assert.equal(result.status, 200);
  assert.match(result.body, /const baseWsUrl = 'ws:\/\/127\.0\.0\.1:8765\/v1\/sessions\/cast\?pageIndex=0';/);
  assert.doesNotMatch(result.body, /id="(?:url-text|tab-bar|back-button|forward-button|refresh-button)"/);
  assert.match(result.body, /const interactive = true;/);
  assert.equal(f.upgrade(auth, '/v1/sessions/cast?pageIndex=0').response, '');
  f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
  f.clients[0].emit('message', Buffer.from(JSON.stringify({ ...mouse, pageId: 'default' })), false);
  await tick();
  assert.equal(f.inputs[0].calls.length, 2);
  assert.deepEqual(f.inputs[0].calls[0], { method: 'Target.attachToTarget', params: { targetId: 'ABC', flatten: true }, sessionId: undefined });
  assert.equal(f.upstreams[0].sent.length, 0);
});

for (const target of [null, 'DEF']) {
  test(`input requires latest frame target (${target})`, async (t) => {
    const f = await fixture(t);
    const auth = await f.login();
    await auth.mutate('/take');
    f.upgrade(auth, '/v1/sessions/cast?pageIndex=0');
    if (target) f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: target, data: 'frame' })), false);
    f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
    await tick();
    assert.equal(f.inputs[0].calls.length, 0);
    assert.equal(f.coordinator.lease.mode, 'PAUSED');
  });
}


test('single-page default alias attaches the latest real frame target', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth, '/v1/sessions/cast?pageIndex=0');
  for (const target of ['ABC', 'DEF']) {
    f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: target, data: 'frame' })), false);
    f.clients[0].emit('message', Buffer.from(JSON.stringify({ ...mouse, pageId: 'default' })), false);
    await tick();
    assert.deepEqual(f.inputs[0].calls.at(-2), { method: 'Target.attachToTarget', params: { targetId: target, flatten: true }, sessionId: undefined });
    assert.equal(f.inputs[0].calls.at(-1).method, 'Input.dispatchMouseEvent');
  }
});

for (const [path, target, message] of [
  ['pageId=ABC', 'ABC', { ...mouse, pageId: 'default' }],
  ['pageIndex=0', null, { ...mouse, pageId: 'default' }],
  ['pageIndex=0', 'default', { ...mouse, pageId: 'default' }],
  ['pageIndex=0', 'ABC', { ...mouse, pageId: 'attacker' }],
  ['pageIndex=0', 'ABC', { ...mouse, pageId: 'default', event: { ...mouse.event, x: -1 } }],
]) {
  test(`alias validation rejects ${path}/${target}/${JSON.stringify(message)}`, async (t) => {
    const f = await fixture(t);
    const auth = await f.login();
    await auth.mutate('/take');
    f.upgrade(auth, `/v1/sessions/cast?${path}`);
    if (target) f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: target, data: 'frame' })), false);
    f.clients[0].emit('message', Buffer.from(JSON.stringify(message)), false);
    await tick();
    assert.equal(f.inputs[0].calls.length, 0);
    assert.equal(f.coordinator.lease.mode, 'PAUSED');
  });
}

for (const stage of ['acquire', 'handoff']) {
  test(`logout during ${stage} releases captured lease without failing`, async (t) => {
    const f = await fixture(t);
    const auth = await f.login();
    const action = f.coordinator.action.bind(f.coordinator);
    let resume;
    f.coordinator.action = async (request) => {
      const result = await action(request);
      if (request.action === stage) await new Promise((resolve) => { resume = resolve; });
      return result;
    };
    const take = auth.mutate('/take');
    while (!resume) await tick();
    const logout = auth.mutate('/auth/logout');
    while ((await auth.get('/state')).status !== 401) await tick();
    resume();
    assert.equal((await logout).status, 200);
    assert.equal((await take).status, 409);
    assert.equal(f.inputs.length, 0);
    assert.equal(f.coordinator.status().mode, 'IDLE');
    assert.equal(f.coordinator.failedLatch, false);
  });
}

for (const failure of ['ack', 'close']) {
  test(`logout ${failure} timeout invalidates session but retains FAILED lease`, async (t) => {
    const f = await fixture(t);
    const auth = await f.login();
    await auth.mutate('/take');
    f.upgrade(auth);
    f.upstreams[0].emit('message', Buffer.from(JSON.stringify({ pageId: 'ABC', data: 'frame' })), false);
    if (failure === 'ack') {
      f.inputs[0].command = () => new Promise(() => {});
      f.clients[0].emit('message', Buffer.from(JSON.stringify(mouse)), false);
      await tick();
    } else f.upstreams[0].autoClose = false;
    assert.equal((await auth.mutate('/auth/logout')).status, 409);
    assert.equal((await auth.get('/state')).status, 401);
    assert.equal(f.coordinator.failedLatch, true);
    assert.equal(f.coordinator.lease.mode, 'FAILED');
  });
}

test('logout waits for both sockets before releasing lease', async (t) => {
  const f = await fixture(t);
  const auth = await f.login();
  await auth.mutate('/take');
  f.upgrade(auth);
  const client = f.clients[0], upstream = f.upstreams[0];
  client.autoClose = upstream.autoClose = false;
  const logout = auth.mutate('/auth/logout');
  while (client.readyState !== 2) await tick();
  assert.equal((await auth.get('/state')).status, 401);
  assert.equal(f.coordinator.lease.mode, 'PAUSED');
  client.finish();
  await tick();
  assert.notEqual(f.coordinator.lease, null);
  upstream.finish();
  assert.equal((await logout).status, 200);
  assert.equal(f.coordinator.status().mode, 'IDLE');
  assert.equal(f.coordinator.failedLatch, false);
});


test('configured viewer port keeps exact Host and Origin checks', async (t) => {
  const configured = 'http://127.0.0.1:9843';
  const f = await fixture(t, { origin: configured });
  assert.equal((await f.http('/')).status, 403);
  const headers = { Host: '127.0.0.1:9843', Origin: configured, 'Content-Type': 'application/x-www-form-urlencoded' };
  assert.equal((await f.http('/', { headers })).status, 200);
  assert.equal((await f.http('/auth/login', { method: 'POST', headers: { ...headers, Origin: origin }, body: `token=${token}` })).status, 403);
  assert.equal((await f.http('/auth/login', { method: 'POST', headers, body: `token=${token}` })).status, 303);
});
