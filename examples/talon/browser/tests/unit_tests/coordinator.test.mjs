import test from 'node:test';
import assert from 'node:assert/strict';
import { Coordinator } from '../../coordinator.mjs';

const owner = { operator_id: 'operator', provider: 'telegram', sender_id: 'sender', conversation_id: 'chat', run_id: 'run', background: false };
const make = (options = {}) => new Coordinator({ operator: 'operator', identities: { telegram: 'sender' }, ...options });
const acquire = (c, id = 'acquire') => c.action({ action: 'acquire', owner, request_id: id });

test('immutable exact identity and deployment-wide ownership', async () => {
  assert.throws(() => make({ operator: '' }), /invalid_config/);
  const c = make();
  const lease = await acquire(c);
  assert.deepEqual(Object.keys(lease), ['lease_id', 'generation', 'version', 'mode']);
  for (const invalid of [{ ...owner, extra: 1 }, { ...owner, sender_id: 'other' }, { ...owner, background: 'false' }, { ...owner, run_id: '' }]) {
    assert.throws(() => c.action({ action: 'acquire', owner: invalid, request_id: 'other' }), /invalid_owner/);
  }
  assert.throws(() => c.action({ action: 'acquire', owner: { ...owner, run_id: 'other' }, request_id: 'other' }), /lease_busy/);
  assert.throws(() => c.check({ ...lease, owner, generation: 0 }), /invalid_lease/);
  assert.throws(() => c.check({ ...lease, owner, version: 0 }), /stale_version/);
});

test('dedup is payload-sensitive and full cache never evicts', async () => {
  const c = make();
  const first = await acquire(c);
  assert.deepEqual(await acquire(c), first);
  assert.throws(() => c.action({ action: 'acquire', owner, request_id: 'acquire', version: 9 }), /request_conflict/);
  for (let i = 1; i < 256; i++) await acquire(c, String(i));
  assert.throws(() => acquire(c, 'overflow'), /request_limit/);
  assert.deepEqual(await acquire(c), first);
});

test('handoff fences immediately, drains, closes before HUMAN and only local cancel frees lease', async () => {
  const c = make();
  const lease = await acquire(c);
  let finish;
  const work = c.track(c.lease, () => new Promise((resolve) => { finish = resolve; }));
  await Promise.resolve();
  let closed = false;
  c.lease.transport = { close: async () => { closed = true; }, abort() {} };
  const pending = c.action({ ...lease, owner, action: 'handoff', request_id: 'handoff' });
  assert.equal(c.status().mode, 'HANDOFF_PENDING');
  assert.throws(() => acquire(c), /lease_fenced/);
  assert.throws(() => c.check({ ...lease, owner }), /stale_version/);
  assert.equal(closed, false);
  finish();
  await work;
  const result = await pending;
  assert.equal(result.status, 'viewer_unavailable');
  assert.equal(closed, true);
  await c.take({ ...result, owner }, owner);
  assert.equal(c.status().mode, 'HUMAN');
  assert.throws(() => c.action({ ...lease, owner, action: 'take', request_id: 'take' }), /invalid_request/);
  await c.cancel({ ...c.status(), handoff_id: result.handoff_id, owner }, owner);
  assert.equal(c.status().mode, 'IDLE');
  assert.ok((await acquire(c)).generation > lease.generation);
});

test('bound inspect reconciles stale handoff without weakening release CAS or human fencing', async () => {
  const c = make();
  const lease = await acquire(c);
  const h = await c.action({ ...lease, owner, action: 'handoff', request_id: 'h' });
  const inspect = { ...lease, version: undefined, owner, action: 'inspect', request_id: 'inspect' };
  for (const action of ['inspect', 'release']) {
    assert.throws(() => c.action({ ...inspect, version: lease.version, action, owner: { ...owner, run_id: 'other' } }), /wrong_owner/);
    assert.throws(() => c.action({ ...inspect, version: lease.version, action, generation: 0 }), /invalid_lease/);
    assert.throws(() => c.action({ ...inspect, version: lease.version, action, lease_id: 'other' }), /invalid_lease/);
  }
  assert.deepEqual(await c.action(inspect), c.status());
  await assert.rejects(c.action({ ...lease, owner, action: 'release', request_id: 'stale' }), /stale_version/);
  const current = await c.action(inspect);
  const human = await c.take({ ...h, owner }, owner);
  await assert.rejects(c.action({ ...current, owner, action: 'release', request_id: 'race' }), /stale_version/);
  await assert.rejects(c.action({ ...human, owner, action: 'release', request_id: 'human' }), /lease_fenced/);
  assert.equal(c.status().mode, 'HUMAN');
  await c.cancel({ ...human, handoff_id: h.handoff_id, owner }, owner);
  await acquire(c, 'next');
  assert.throws(() => c.action(inspect), /invalid_lease/);
});

test('never-human handoff expiry frees ownership only after draining', async () => {
  for (const pending of [false, true]) {
    let now = 0;
    const c = make({ now: () => now, ttl: 100 });
    const lease = await acquire(c);
    let finish;
    if (pending) {
      c.track(c.lease, () => new Promise((resolve) => { finish = resolve; }));
      await Promise.resolve();
    }
    const handoff = c.action({ ...lease, owner, action: 'handoff', request_id: 'h' });
    if (!pending) await handoff;
    now = 101;
    c.expire();
    assert.throws(() => acquire(c, 'blocked'), /lease_fenced/);
    finish?.();
    await handoff;
    await new Promise(setImmediate);
    assert.equal(c.status().mode, 'IDLE');
    const next = await c.action({ owner: { ...owner, run_id: 'next' }, action: 'acquire', request_id: 'next' });
    assert.ok(next.generation > lease.generation);
  }
});

test('background handoff is human_required and duplicate handoff stable', async () => {
  const c = make();
  const bg = { ...owner, background: true };
  const lease = await c.action({ action: 'acquire', owner: bg, request_id: 'acquire' });
  const envelope = { ...lease, owner: bg, action: 'handoff', request_id: 'handoff' };
  const result = await c.action(envelope);
  assert.equal(result.status, 'human_required');
  assert.deepEqual(await c.action(envelope), result);
});

test('expiry keeps active lease fenced until drain completes', async () => {
  let now = 0;
  const c = make({ now: () => now });
  await acquire(c);
  let finish;
  const work = c.track(c.lease, () => new Promise((resolve) => { finish = resolve; }));
  await Promise.resolve();
  now = 1800001;
  assert.equal(c.status().mode, 'PAUSED');
  assert.throws(() => acquire(c), /lease_fenced/);
  finish();
  await work;
  await c.lease.stopping;
  await new Promise(setImmediate);
  assert.equal(c.status().mode, 'IDLE');
});

test('uncertain drain fails permanently rather than granting another owner', async () => {
  const c = make({ drainMs: 5 });
  const lease = await acquire(c);
  c.track(c.lease, () => new Promise(() => {}));
  await assert.rejects(c.action({ ...lease, owner, action: 'release', request_id: 'release' }), /lease_failed/);
  assert.equal(c.status().mode, 'FAILED');
  assert.throws(() => acquire(c), /lease_fenced/);
});


test('private viewer operations require owner and current fences; done resumes same unexpired lease', async () => {
  const c = make();
  const lease = await acquire(c);
  assert.ok(Number.isSafeInteger(lease.generation) && lease.generation > 0);
  assert.notEqual(make().generation, make().generation);
  assert.throws(() => c.action({ ...lease, version: undefined, owner, action: 'release', request_id: 'r' }), /invalid_request/);
  const h = await c.action({ ...lease, owner, action: 'handoff', request_id: 'h' });
  assert.equal(h.lease_id, lease.lease_id);
  assert.equal(h.mode, 'PAUSED');
  assert.ok(h.version > lease.version);
  await assert.rejects(c.take({ ...h, owner }, { ...owner, run_id: 'other' }), /wrong_owner/);
  await assert.rejects(c.take({ ...h, owner, generation: 0 }, owner), /invalid_lease/);
  const human = await c.take({ ...h, owner }, owner);
  await assert.rejects(c.done({ ...h, owner }, owner), /stale_version/);
  const resumed = await c.done({ ...human, owner, handoff_id: h.handoff_id }, owner);
  assert.equal(resumed.mode, 'AGENT');
  assert.equal(resumed.lease_id, lease.lease_id);
  assert.equal(c.lease.external, false);
  assert.equal(c.lease.stopping, null);
  await c.action({ ...resumed, owner, action: 'release', request_id: 'r' });
});

test('human expiry pauses without reassignment or resume; local extensions bounded and foreground only', async () => {
  let now = 0;
  const c = make({ now: () => now, ttl: 100 });
  let lease = await acquire(c);
  assert.throws(() => c.extend(owner, { ...lease, owner }, 101), /invalid_request/);
  lease = c.extend(owner, { ...lease, owner }, 100);
  const h = await c.action({ ...lease, owner, action: 'handoff', request_id: 'h' });
  const human = await c.take({ ...h, owner }, owner);
  now = 101;
  assert.equal(c.status().mode, 'PAUSED');
  await c.lease.stopping;
  assert.equal(c.status().lease_id, lease.lease_id);
  await assert.rejects(c.done({ ...human, owner, handoff_id: h.handoff_id }, owner), /stale_version|lease_fenced/);
  assert.throws(() => acquire(c), /lease_fenced/);
  const bg = make();
  const background = { ...owner, background: true };
  const b = await bg.action({ owner: background, action: 'acquire', request_id: 'b' });
  assert.throws(() => bg.extend(background, { ...b, owner: background }, 10), /lease_fenced/);
});

test('256 HTTP command quota is separate from coordination; no results cached and failure latch survives cleanup', async () => {
  const c = make();
  const lease = await acquire(c);
  const held = c.lease;
  for (let i = 0; i < 256; i++) {
    await c.command(held, { request_id: String(i) }, () => ({ raw: 'private' }));
  }
  assert.throws(() => c.command(held, { request_id: '0' }, () => {}), /request_replayed/);
  assert.throws(() => c.command(held, { request_id: '256' }, () => {}), /request_limit/);
  assert.ok([...held.commands.values()].every((entry) => Object.keys(entry).sort().join() === 'digest,status'));
  await acquire(c, 'still-coordinates');
  c.failed(held);
  await new Promise(setImmediate);
  assert.equal(held.commands.size, 0);
  assert.equal(held.requests.size, 0);
  assert.equal(held.pending.size, 0);
  assert.throws(() => acquire(c), /lease_fenced/);
  assert.equal(c.status().lease_id, lease.lease_id);
});
