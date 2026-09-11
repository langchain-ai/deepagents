import http from 'node:http';
import { constants, openSync, fstatSync, readFileSync, closeSync } from 'node:fs';
import { timingSafeEqual } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { Coordinator, BridgeError, fail, object, text } from './coordinator.mjs';

export const MAX_BYTES = 4 * 1024 * 1024;
const STEEL = 'http://172.30.14.2:3000';
const SOCKET = 'ws://172.30.14.2:3000/';
const errorStatus = (error) => {
  const code = errorCode(error);
  if (['invalid_owner', 'wrong_owner'].includes(code)) return 403;
  if (code.startsWith('invalid_') || error instanceof SyntaxError) return 400;
  if (['request_limit', 'response_limit'].includes(code)) return 413;
  return code === 'upstream_unavailable' ? 503 : 409;
};
const errorCode = (error) => error instanceof BridgeError ? error.message : error instanceof SyntaxError ? 'invalid_request' : 'upstream_unavailable';

export function readToken(path) {
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || (stat.mode & 0o777) !== 0o400 || stat.size !== 43) fail('invalid_token_file');
    const token = readFileSync(fd, 'utf8');
    if (!/^[A-Za-z0-9_-]{43}$/.test(token)) fail('invalid_token_file');
    return token;
  } finally { closeSync(fd); }
}

export function validateSocketURL(value) {
  if (value !== SOCKET) fail('invalid_upstream');
  return SOCKET;
}

async function fixedJSON(url, method = 'GET') {
  const response = await fetch(url, { method, redirect: 'error', signal: AbortSignal.timeout(2000),
    ...(method === 'POST' ? { body: '{}', headers: { 'Content-Type': 'application/json' } } : {}) });
  if (!response.ok) { await response.body?.cancel(); fail('upstream_unavailable'); }
  let size = 0;
  const chunks = [];
  for await (const chunk of response.body) {
    size += chunk.length;
    if (size > MAX_BYTES) fail('upstream_unavailable');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString());
}

export async function discover() {
  const listing = await fixedJSON(`${STEEL}/v1/sessions`);
  if (!object(listing) || !Array.isArray(listing.sessions) || !listing.sessions.every((session) => object(session) && text(session.id) && text(session.status))) fail('upstream_unavailable');
  if (!listing.sessions.some((session) => session.status === 'live')) {
    const session = await fixedJSON(`${STEEL}/v1/sessions`, 'POST');
    if (!object(session) || !text(session.id) || session.status !== 'live') fail('upstream_unavailable');
  }
  return validateSocketURL(SOCKET);
}

export class Transport {
  constructor({ WebSocket, coordinator, lease, discoverURL = discover, events = () => {}, timer = setTimeout, clear = clearTimeout }) {
    Object.assign(this, { WebSocket, coordinator, lease, discoverURL, events, timer, clear });
    this.pending = new Map();
    this.nextId = 0;
    this.closed = false;
  }

  connect() {
    if (!this.opening) this.opening = this.open();
    return this.opening;
  }

  async open() {
    const url = await this.discoverURL();
    if (this.closed) fail('lease_fenced');
    const socket = this.socket = new this.WebSocket(url, { maxPayload: MAX_BYTES, handshakeTimeout: 10000, followRedirects: false, perMessageDeflate: false });
    socket.on('message', (data, binary) => this.message(data, binary));
    socket.on('error', () => this.break());
    socket.on('close', () => { if (!this.closed) this.break(); });
    await new Promise((resolve, reject) => {
      socket.once('open', resolve);
      socket.once('error', () => reject(new BridgeError('upstream_unavailable')));
      socket.once('close', () => reject(new BridgeError('upstream_unavailable')));
    });
    if (this.closed) fail('lease_fenced');
  }

  message(data, binary) {
    try {
      if (binary || data.length > MAX_BYTES) return this.break();
      const message = JSON.parse(data.toString());
      if (!object(message)) return this.break();
      if (Object.hasOwn(message, 'id')) {
        const entry = this.pending.get(message.id);
        if (!entry) return this.break();
        this.pending.delete(message.id);
        this.clear(entry.timer);
        if (message.error) entry.reject(new BridgeError('cdp_error'));
        else entry.resolve(message.result ?? {});
      } else if (this.lease.mode === 'AGENT') this.events(data.toString());
    } catch { this.break(); }
  }

  async command(method, params, sessionId) {
    if (this.closed || this.lease.mode !== 'AGENT') fail('lease_fenced');
    try { await this.connect(); }
    catch { this.break(); fail('upstream_unavailable'); }
    if (this.closed || this.lease.mode !== 'AGENT') fail('lease_fenced');
    if (this.pending.size >= 32) fail('pending_limit');
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = this.timer(() => this.break(), method === 'Page.navigate' ? 30000 : 10000);
      this.pending.set(id, { resolve, reject, timer });
      try {
        const payload = JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) });
        if (Buffer.byteLength(payload) + (this.socket.bufferedAmount ?? 0) > MAX_BYTES) {
          this.pending.delete(id);
          this.clear(timer);
          return reject(new BridgeError('request_limit'));
        }
        this.socket.send(payload);
      } catch { this.break(); }
    });
  }

  break() {
    this.coordinator.failed(this.lease);
  }

  abort() {
    this.closed = true;
    for (const entry of this.pending.values()) {
      this.clear(entry.timer);
      entry.reject(new BridgeError('lease_failed'));
    }
    this.pending.clear();
    this.socket?.terminate();
    this.client?.terminate();
  }

  async close() {
    this.closed = true;
    const sockets = [this.socket, this.client].filter(Boolean);
    await Promise.all(sockets.map((socket) => new Promise((resolve, reject) => {
      if (socket.readyState === 3) return resolve();
      const timer = this.timer(() => { socket.terminate(); reject(new BridgeError('close_timeout')); }, 1000);
      socket.once('close', () => { this.clear(timer); resolve(); });
      socket.close();
    })));
  }
}

function authorized(request, token) {
  if (request.rawHeaders.filter((_, index) => index % 2 === 0 && request.rawHeaders[index].toLowerCase() === 'authorization').length !== 1) return false;
  const value = Buffer.from(request.headers.authorization ?? '');
  const expected = Buffer.from(`Bearer ${token}`);
  return value.length === expected.length && timingSafeEqual(value, expected);
}

function reply(response, status, value) {
  let body = JSON.stringify(value);
  if (Buffer.byteLength(body) > MAX_BYTES) { status = 413; body = '{"error":"response_limit"}'; }
  response.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Content-Length': Buffer.byteLength(body), Connection: 'close' });
  response.end(body);
}

async function body(request, limit) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > limit) fail('request_limit');
    chunks.push(chunk);
  }
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString());
    if (!object(value)) fail('invalid_request');
    return value;
  } catch { fail('invalid_request'); }
}

function commandFields(envelope) {
  if (!text(envelope.method) || !object(envelope.params) ||
      (envelope.session_id !== undefined && !text(envelope.session_id))) fail('invalid_request');
}

export function createBridge({ token, coordinator, WebSocket, WebSocketServer, discoverURL = discover,
  healthy = async () => { await fixedJSON(`${STEEL}/v1/sessions`); return true; },
  controlHost = '172.30.12.3', controlPort = 8081, viewerHost = '172.30.13.3', viewerPort = 8080 }) {
  if (!/^[A-Za-z0-9_-]{43}$/.test(token)) fail('invalid_token_file');
  const wss = new WebSocketServer({ noServer: true, maxPayload: MAX_BYTES, perMessageDeflate: false });
  const transport = (lease) => {
    if (!lease.transport) lease.transport = new Transport({ WebSocket, coordinator, lease, discoverURL });
    return lease.transport;
  };
  const command = (envelope) => {
    commandFields(envelope);
    const lease = coordinator.check(envelope);
    if (lease.external) fail('transport_busy');
    return coordinator.command(lease, envelope, async () => {
      const result = await coordinator.track(lease, () => transport(lease).command(envelope.method, envelope.params, envelope.session_id));
      coordinator.check(envelope);
      if (Buffer.byteLength(JSON.stringify({ result })) > MAX_BYTES) fail('response_limit');
      return { result };
    }).then((result) => { coordinator.check(envelope); return result; });
  };
  const handler = (control) => async (request, response) => {
    try {
      if (request.method === 'GET' && request.url === '/health') {
        const ready = await healthy().catch(() => false);
        return reply(response, ready ? 200 : 503, { status: ready ? 'ready' : 'unavailable' });
      }
      const status = request.method === 'GET' && request.url === '/internal/browser/status';
      const action = request.method === 'POST' && request.url === '/internal/browser/actions';
      const invoke = request.method === 'POST' && request.url === '/internal/browser/command';
      if (!control || !(status || action || invoke)) return reply(response, 404, { error: 'not_found' });
      if (!authorized(request, token)) return reply(response, 401, { error: 'unauthorized' });
      if (status) return reply(response, 200, coordinator.status());
      const envelope = await body(request, invoke ? MAX_BYTES : 16384);
      const result = await (action ? coordinator.action(envelope) : command(envelope));
      if (invoke) coordinator.check(envelope);
      return reply(response, 200, result);
    } catch (error) { if (!response.destroyed) reply(response, errorStatus(error), { error: errorCode(error) }); }
  };
  const control = http.createServer({ requestTimeout: 10000, headersTimeout: 10000, maxHeaderSize: 16384 }, handler(true));
  const viewer = http.createServer({ requestTimeout: 10000, headersTimeout: 10000, maxHeaderSize: 16384 }, handler(false));
  const reject = (socket, status, code) => {
    const payload = JSON.stringify({ error: code });
    socket.end(`HTTP/1.1 ${status} Rejected\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: ${Buffer.byteLength(payload)}\r\n\r\n${payload}`);
  };
  let connections = 0;
  for (const server of [control, viewer]) {
    server.on('connection', (socket) => {
      if (connections >= 64) { socket.destroy(); return; }
      connections++;
      socket.once('close', () => { connections--; });
      socket.setTimeout(10000, () => socket.destroy());
    });
  }
  viewer.on('upgrade', (_, socket) => reject(socket, 404, 'not_found'));
  control.on('upgrade', (request, socket, head) => {
    try {
      if (request.method !== 'GET' || request.url !== '/internal/browser/cdp') return reject(socket, 404, 'not_found');
      if (!authorized(request, token)) return reject(socket, 401, 'unauthorized');
      const encoded = request.headers['x-browser-owner'];
      if (typeof encoded !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) fail('invalid_owner');
      const envelope = { owner: JSON.parse(Buffer.from(encoded, 'base64').toString()), lease_id: request.headers['x-browser-lease'], generation: Number(request.headers['x-browser-generation']) };
      const lease = coordinator.check(envelope, true, false);
      if (lease.transport || lease.external) fail('transport_busy');
      lease.external = true;
      const upstream = transport(lease);
      wss.handleUpgrade(request, socket, head, (client) => attach(client, upstream, envelope, lease));
      socket.once('close', () => { if (!upstream.client && lease.mode === 'AGENT') coordinator.failed(lease); });
    } catch (error) { reject(socket, errorStatus(error), errorCode(error)); }
  });
  function attach(client, upstream, envelope, lease) {
    upstream.client = client;
    const ids = new Set();
    const send = (value) => {
      const data = typeof value === 'string' ? value : JSON.stringify(value);
      if (Buffer.byteLength(data) + client.bufferedAmount > MAX_BYTES) return coordinator.failed(lease);
      if (client.readyState === 1) client.send(data);
    };
    upstream.events = send;
    client.on('error', () => coordinator.failed(lease));
    client.on('close', () => { if (lease.mode === 'AGENT') coordinator.failed(lease); });
    client.on('message', async (data, binary) => {
      let message;
      let accepted = false;
      try {
        if (binary || data.length > MAX_BYTES) fail('invalid_request');
        message = JSON.parse(data.toString());
        if (!object(message) || !Number.isSafeInteger(message.id) || ids.has(message.id)) fail('invalid_request');
        commandFields({ ...message, params: message.params ?? {}, session_id: message.sessionId });
        coordinator.check(envelope, true, false);
        ids.add(message.id);
        accepted = true;
        const result = await coordinator.track(lease, () => upstream.command(message.method, message.params ?? {}, message.sessionId));
        coordinator.check(envelope, true, false);
        send({ id: message.id, result, ...(message.sessionId ? { sessionId: message.sessionId } : {}) });
      } catch (error) {
        if (lease.mode === 'AGENT') send({ ...(Number.isSafeInteger(message?.id) ? { id: message.id } : {}), error: errorCode(error) });
      } finally { if (accepted) ids.delete(message.id); }
    });
  }
  const listen = (server, host, port) => new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, host, resolve); });
  return { control, viewer, coordinator, command,
    async start() {
      try { await listen(control, controlHost, controlPort); await listen(viewer, viewerHost, viewerPort); }
      catch (error) { control.close(); viewer.close(); throw error; }
    },
    async close() {
      if (coordinator.lease) { coordinator.clear(coordinator.lease.timer); coordinator.failed(coordinator.lease); }
      wss.close();
      await Promise.all([control, viewer].map((server) => new Promise((resolve) => { server.close(resolve); server.closeAllConnections(); })));
    },
  };
}

export async function main(env = process.env) {
  if (env.TALON_BROWSER_STEEL_URL && env.TALON_BROWSER_STEEL_URL !== STEEL) fail('invalid_config');
  const addresses = {};
  for (const [name, host, port] of [['CONTROL', '172.30.12.3', 8081], ['VIEWER', '172.30.13.3', 8080]]) {
    if (env[`TALON_BROWSER_${name}_HOST`] && env[`TALON_BROWSER_${name}_HOST`] !== host) fail('invalid_config');
    const configuredPort = Number(env[`TALON_BROWSER_${name}_PORT`] ?? port);
    if (!Number.isInteger(configuredPort) || configuredPort < 1 || configuredPort > 65535) fail('invalid_config');
    addresses[`${name.toLowerCase()}Host`] = host;
    addresses[`${name.toLowerCase()}Port`] = configuredPort;
  }
  const coordinator = new Coordinator({ operator: env.TALON_BROWSER_OPERATOR_ID,
    identities: JSON.parse(env.TALON_BROWSER_IDENTITIES), ttl: Number(env.TALON_BROWSER_LEASE_TTL_SECONDS ?? 1800) * 1000 });
  const { WebSocket, WebSocketServer } = createRequire('/app/api/package.json')('ws');
  const bridge = createBridge({ token: readToken(env.TALON_BROWSER_TOKEN_FILE), coordinator, WebSocket, WebSocketServer, ...addresses });
  await bridge.start();
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => { void bridge.close(); });
  return bridge;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(() => { process.stderr.write('browser_bridge_start_failed\n'); process.exitCode = 1; });
}
