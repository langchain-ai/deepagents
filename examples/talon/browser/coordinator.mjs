import { createHash, randomInt, randomUUID } from 'node:crypto';

export const OWNER_FIELDS = ['operator_id', 'provider', 'conversation_id', 'sender_id', 'run_id', 'background'];
export const fail = (code) => { throw new BridgeError(code); };
export class BridgeError extends Error {}
export const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
export const text = (value) => typeof value === 'string' && value.trim().length > 0 && value.length <= 1024;
export function canonical(value) {
  if (Array.isArray(value)) return JSON.stringify(value.map((item) => JSON.parse(canonical(item))));
  if (!object(value)) return JSON.stringify(value);
  return JSON.stringify(Object.fromEntries(Object.keys(value).sort().map((key) => [key, JSON.parse(canonical(value[key]))])));
}

export function configuration(operator, identities) {
  if (!text(operator) || !object(identities) || !Object.keys(identities).length) fail('invalid_config');
  for (const [provider, sender] of Object.entries(identities)) {
    if (!text(provider) || !text(sender)) fail('invalid_config');
  }
  return Object.freeze({ operator, identities: Object.freeze({ ...identities }) });
}

export class Coordinator {
  constructor({ operator, identities, ttl = 1800000, now = Date.now, timer = setTimeout, clear = clearTimeout, drainMs = 10000 }) {
    this.config = configuration(operator, identities);
    if (!Number.isSafeInteger(ttl) || ttl < 1) fail('invalid_config');
    Object.assign(this, { ttl, now, timer, clear, drainMs });
    this.generation = randomInt(1, 2 ** 48 - 1);
    this.failedLatch = false;
    this.lease = null;
  }

  owner(owner) {
    if (!object(owner) || Object.keys(owner).length !== OWNER_FIELDS.length ||
        !OWNER_FIELDS.every((key) => Object.hasOwn(owner, key))) fail('invalid_owner');
    if (!OWNER_FIELDS.filter((key) => key !== 'background').every((key) => text(owner[key])) ||
        typeof owner.background !== 'boolean') fail('invalid_owner');
    if (owner.operator_id !== this.config.operator || !Object.hasOwn(this.config.identities, owner.provider) ||
        owner.sender_id !== this.config.identities[owner.provider]) fail('invalid_owner');
    return canonical(owner);
  }

  status() {
    this.expire();
    const lease = this.lease;
    return { lease_id: lease?.id ?? null, generation: lease?.generation ?? this.generation,
      version: lease?.version ?? 0, mode: lease?.mode ?? 'IDLE' };
  }

  expire() {
    const lease = this.lease;
    if (lease && this.now() >= lease.expires && !lease.expiring) {
      lease.expiring = true;
      void this.stop(lease, lease.mode === 'AGENT' || (!lease.human && ['PAUSED', 'HANDOFF_PENDING'].includes(lease.mode))).catch(() => {});
    }
  }

  check(envelope, agent = true, version = true) {
    const owner = this.owner(envelope.owner);
    this.expire();
    const lease = this.lease;
    if (lease && owner !== lease.owner) fail('wrong_owner');
    if (!lease || envelope.lease_id !== lease.id || envelope.generation !== lease.generation || owner !== lease.owner) fail('invalid_lease');
    if (version && envelope.version !== lease.version) fail('stale_version');
    if (agent && lease.mode !== 'AGENT') fail('lease_fenced');
    return lease;
  }

  memo(lease, envelope, operation) {
    if (!text(envelope.request_id)) fail('invalid_request');
    const digest = createHash('sha256').update(canonical(envelope)).digest('hex');
    const existing = lease.requests.get(envelope.request_id);
    if (existing) {
      if (existing.digest !== digest) fail('request_conflict');
      return existing.promise;
    }
    if (lease.requests.size >= 256) fail('request_limit');
    const entry = { digest };
    lease.requests.set(envelope.request_id, entry);
    try { entry.promise = Promise.resolve(operation()); }
    catch (error) { entry.promise = Promise.reject(error); }
    return entry.promise;
  }

  action(envelope) {
    if (!object(envelope) || !['acquire', 'release', 'handoff', 'inspect'].includes(envelope.action) || !text(envelope.request_id)) fail('invalid_request');
    const owner = this.owner(envelope.owner);
    this.expire();
    if (envelope.action === 'acquire') {
      if (this.failedLatch) fail('lease_fenced');
      if (!this.lease) this.create(owner, envelope.owner.background);
      const lease = this.lease;
      if (lease.owner !== owner) fail('lease_busy');
      if (lease.mode !== 'AGENT') fail('lease_fenced');
      return this.memo(lease, envelope, () => {
        if (lease.mode !== 'AGENT') fail('lease_fenced');
        return this.status();
      });
    }
    if (envelope.action === 'inspect') {
      const lease = this.check(envelope, false, false);
      return { lease_id: lease.id, generation: lease.generation, version: lease.version, mode: lease.mode };
    }
    if (!Number.isSafeInteger(envelope.version)) fail('invalid_request');
    const lease = this.check(envelope, false, false);
    return this.memo(lease, envelope, async () => {
      if (envelope.version !== lease.version) fail('stale_version');
      if (envelope.action === 'release') {
        if (!['AGENT', 'PAUSED', 'HANDOFF_PENDING', 'FAILED'].includes(lease.mode)) fail('lease_fenced');
        await this.stop(lease, true);
        return { status: 'released' };
      }
      if (lease.mode !== 'AGENT') fail('lease_fenced');
      lease.handoff = randomUUID();
      this.fence(lease, 'HANDOFF_PENDING');
      await this.stop(lease, false);
      return { ...this.status(), status: lease.background ? 'human_required' : 'viewer_unavailable', handoff_id: lease.handoff };
    });
  }

  create(owner, background) {
    const lease = { id: randomUUID(), generation: ++this.generation, version: 1, mode: 'AGENT', owner,
      background, human: false, expires: this.now() + this.ttl, requests: new Map(), commands: new Map(), pending: new Set(), transport: null };
    this.lease = lease;
    lease.timer = this.timer(() => this.expire(), this.ttl);
    lease.timer?.unref?.();
  }

  fence(lease, mode) {
    if (lease.mode !== mode) { lease.mode = mode; lease.version += 1; }
  }

  async stop(lease, release) {
    if (lease.mode === 'FAILED') fail('lease_failed');
    if (lease.mode !== 'HANDOFF_PENDING') this.fence(lease, 'PAUSED');
    if (!lease.stopping) lease.stopping = this.drain(lease);
    await lease.stopping;
    if (lease.mode === 'FAILED') fail('lease_failed');
    this.fence(lease, 'PAUSED');
    if (release && this.lease === lease) {
      lease.requests.clear();
      lease.commands.clear();
      this.clear(lease.timer);
      this.lease = null;
    }
  }

  async drain(lease) {
    let timer;
    try {
      await Promise.race([
        Promise.allSettled([...lease.pending]),
        new Promise((_, reject) => { timer = this.timer(() => reject(new BridgeError('drain_timeout')), this.drainMs); }),
      ]);
      await lease.transport?.close();
      lease.transport = null;
    } catch {
      this.failed(lease);
      fail('lease_failed');
    } finally { this.clear(timer); }
  }

  failed(lease) {
    this.failedLatch = true;
    this.fence(lease, 'FAILED');
    this.clear(lease.timer);
    lease.transport?.abort();
    lease.transport = null;
    lease.requests.clear();
    lease.commands.clear();
    void Promise.allSettled([...lease.pending]).then(() => {
      lease.pending.clear();
      lease.requests.clear();
      lease.commands.clear();
      lease.stopping = null;
    });
  }

  track(lease, operation) {
    if (lease.mode !== 'AGENT') fail('lease_fenced');
    if (lease.pending.size >= 32) fail('pending_limit');
    const promise = Promise.resolve().then(operation);
    lease.pending.add(promise);
    promise.finally(() => lease.pending.delete(promise)).catch(() => {});
    return promise;
  }

  viewer(envelope, context) {
    if (!object(context) || this.owner(context) !== this.owner(envelope.owner)) fail('wrong_owner');
    const lease = this.check(envelope, false);
    if (!text(envelope.handoff_id) || lease.handoff !== envelope.handoff_id) fail('invalid_handoff');
    return lease;
  }

  async take(envelope, context) {
    const lease = this.viewer(envelope, context);
    if (lease.mode !== 'PAUSED' || lease.expiring) fail('lease_fenced');
    await lease.stopping;
    this.viewer(envelope, context);
    if (lease.expiring) fail('lease_fenced');
    lease.stopping = null;
    lease.human = true;
    this.fence(lease, 'HUMAN');
    return this.status();
  }

  async done(envelope, context) {
    const lease = this.viewer(envelope, context);
    if (lease.mode !== 'HUMAN' || lease.expiring) fail('lease_fenced');
    await this.stop(lease, false);
    this.expire();
    if (lease.expiring || this.lease !== lease || lease.mode !== 'PAUSED') fail('lease_fenced');
    lease.stopping = null;
    lease.requests.clear();
    lease.external = false;
    lease.handoff = null;
    this.fence(lease, 'AGENT');
    return this.status();
  }

  async cancel(envelope, context) {
    const lease = this.viewer(envelope, context);
    if (!['PAUSED', 'HUMAN'].includes(lease.mode)) fail('invalid_handoff');
    await this.stop(lease, true);
    return this.status();
  }

  extend(owner, envelope, duration) {
    if (this.owner(owner) !== this.owner(envelope.owner)) fail('wrong_owner');
    const lease = this.check(envelope, false);
    if (lease.background || lease.expiring || !['AGENT', 'PAUSED', 'HUMAN'].includes(lease.mode)) fail('lease_fenced');
    if (!Number.isSafeInteger(duration) || duration < 1 || duration > this.ttl) fail('invalid_request');
    this.clear(lease.timer);
    lease.expires = this.now() + duration;
    lease.timer = this.timer(() => this.expire(), duration);
    lease.timer?.unref?.();
    lease.version += 1;
    return this.status();
  }

  command(lease, envelope, operation) {
    if (!text(envelope.request_id)) fail('invalid_request');
    const digest = createHash('sha256').update(canonical(envelope)).digest('hex');
    const existing = lease.commands.get(envelope.request_id);
    if (existing) fail(existing.digest === digest ? 'request_replayed' : 'request_conflict');
    if (lease.commands.size >= 256) fail('request_limit');
    const entry = { digest, status: 'pending' };
    lease.commands.set(envelope.request_id, entry);
    return Promise.resolve().then(operation).then(
      (result) => { entry.status = 'complete'; return result; },
      (error) => { entry.status = 'failed'; throw error; },
    );
  }
}
