import test from 'node:test';
import assert from 'node:assert/strict';
import { Coordinator } from '../../../deepagents_talon/steel_runtime/coordinator.mjs';
const tick = () => new Promise(setImmediate);

test('exclusive runs release only their own access', async () => {
  const c = new Coordinator();
  await c.command('one', async () => 1);
  assert.throws(() => c.command('two', async () => 2), /browser_busy/);
  await c.release('two');
  assert.throws(() => c.command('two', async () => 2), /browser_busy/);
  await c.release('one');
  assert.equal(await c.command('two', async () => 2), 2);
  await c.release('two');
});

test('pause blocks new commands, drains active work, and resumes the same run', async () => {
  const c = new Coordinator();
  let finish;
  const command = c.command('agent', () => new Promise((resolve) => { finish = resolve; }));
  await tick();
  let paused = false;
  const pause = c.pause().then(() => { paused = true; });
  assert.throws(() => c.command('agent', () => {}), /browser_paused/);
  assert.throws(() => c.command('another', () => {}), /browser_paused/);
  await tick();
  assert.equal(paused, false);
  finish('done');
  assert.equal(await command, 'done');
  await pause;
  await c.resume();
  assert.equal(await c.command('agent', () => 'resumed'), 'resumed');
  await c.release('agent');
});

test('uncertain completion blocks reuse after pause', async () => {
  const c = new Coordinator({ drainMs: 5 });
  let finish;
  const command = c.command('agent', () => new Promise((resolve) => { finish = resolve; }));
  await tick();
  await assert.rejects(c.pause(), /browser_unavailable/);
  await assert.rejects(c.resume(), /browser_unavailable/);
  assert.throws(() => c.command('other', () => {}), /browser_unavailable/);
  finish();
  await command;
});

test('abandoned runs expire after draining outstanding commands', async () => {
  const c = new Coordinator({ ttl: 5 });
  let finish;
  const command = c.command('old', () => new Promise((resolve) => { finish = resolve; }));
  await new Promise((resolve) => setTimeout(resolve, 15));
  assert.throws(() => c.command('new', () => {}), /browser_busy/);
  finish();
  await command;
  await tick();
  await c.command('new', () => {});
  await c.release('new');
});
