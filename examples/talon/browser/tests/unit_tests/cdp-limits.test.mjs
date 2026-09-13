import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Coordinator } from '../../coordinator.mjs';
import { Transport } from '../../bridge.mjs';

const owner = { operator_id: 'op', provider: 'telegram', sender_id: 'sender', conversation_id: 'chat', run_id: 'run', background: false };

async function setup() {
  const c = new Coordinator({ operator: 'op', identities: { telegram: 'sender' } });
  await c.action({ action: 'acquire', owner, request_id: 'acquire' });
  return c;
}

test('32 pending limit rejects before operation dispatch', async () => {
  const c = await setup();
  let dispatched = 0;
  const resolvers = [];
  for (let i = 0; i < 32; i++) c.track(c.lease, () => { dispatched++; return new Promise((resolve) => resolvers.push(resolve)); });
  assert.throws(() => c.track(c.lease, () => { dispatched++; }), /pending_limit/);
  await Promise.resolve();
  assert.equal(dispatched, 32);
  const pending = [...c.lease.pending];
  resolvers.forEach((resolve) => resolve());
  await Promise.all(pending);
});

test('handoff while discovery awaits never dispatches a CDP command', async () => {
  const c = await setup();
  let discover;
  let sends = 0;
  class Socket extends EventEmitter {
    constructor() { super(); this.readyState = 1; queueMicrotask(() => this.emit('open')); }
    send() { sends++; }
    close() { this.readyState = 3; this.emit('close'); }
    terminate() { this.close(); }
  }
  const t = new Transport({ WebSocket: Socket, coordinator: c, lease: c.lease, discoverURL: () => new Promise((resolve) => { discover = resolve; }) });
  c.lease.transport = t;
  const command = c.track(c.lease, () => t.command('Runtime.evaluate', { expression: 'sensitive()' }));
  await Promise.resolve();
  const handoff = c.action({ ...c.status(), owner, action: 'handoff', request_id: 'h' });
  discover('fixed');
  await assert.rejects(command, /lease_fenced/);
  await handoff;
  assert.equal(sends, 0);
  assert.equal(c.status().mode, 'PAUSED');
});

test('failed socket never leaks events or grants replay', async () => {
  const c = await setup();
  let events = 0;
  const t = new Transport({ coordinator: c, lease: c.lease, events: () => events++ });
  c.lease.transport = t;
  c.failed(c.lease);
  t.message(Buffer.from('{"method":"Runtime.consoleAPICalled","params":{"secret":"login"}}'), false);
  assert.equal(events, 0);
  assert.throws(() => c.action({ action: 'acquire', owner, request_id: 'acquire' }), /lease_fenced/);
});
