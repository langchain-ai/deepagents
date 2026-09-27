import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Coordinator } from '../../../deepagents_talon/steel_runtime/coordinator.mjs';
import { Transport } from '../../../deepagents_talon/steel_runtime/bridge.mjs';

test('32 pending limit rejects before operation dispatch', async () => {
  const c = new Coordinator();
  let dispatched = 0;
  const resolvers = [];
  for (let i = 0; i < 32; i++) c.command('run', () => { dispatched++; return new Promise((resolve) => resolvers.push(resolve)); });
  assert.throws(() => c.command('run', () => { dispatched++; }), /pending_limit/);
  await Promise.resolve();
  assert.equal(dispatched, 32);
  const pending = [...c.run.pending];
  resolvers.forEach((resolve) => resolve());
  await Promise.all(pending);
  await c.release('run');
});

test('pause while discovery awaits never dispatches a CDP command', async () => {
  const c = new Coordinator();
  let discover;
  let sends = 0;
  class Socket extends EventEmitter {
    constructor() { super(); this.readyState = 1; queueMicrotask(() => this.emit('open')); }
    send() { sends++; }
    close() { this.readyState = 3; this.emit('close'); }
    terminate() { this.close(); }
  }
  c.acquire('run');
  const t = new Transport({ WebSocket: Socket, allowed: () => c.allowed(c.run), onFailure: () => c.fail(), discoverURL: () => new Promise((resolve) => { discover = resolve; }) });
  c.run.transport = t;
  const command = c.command('run', () => t.command('Runtime.evaluate', { expression: 'sensitive()' }));
  await Promise.resolve();
  const pause = c.pause();
  discover('fixed');
  await assert.rejects(command, /browser_paused/);
  await pause;
  assert.equal(sends, 0);
  assert.equal(c.paused, true);
  await c.release('run');
});
