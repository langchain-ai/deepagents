export const fail = (code) => { throw new BridgeError(code); };
export class BridgeError extends Error {}
export const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
export const text = (value) => typeof value === 'string' && value.trim().length > 0 && value.length <= 1024;

export class Coordinator {
  constructor({ ttl = 1800000, drainMs = 10000 } = {}) {
    if (!Number.isSafeInteger(ttl) || ttl < 1) fail('invalid_config');
    Object.assign(this, { ttl, drainMs, run: null, paused: false, failed: false, pausing: null });
  }

  status() {
    return { paused: this.paused, busy: this.run !== null, failed: this.failed };
  }

  allowed(run) {
    return !this.failed && !this.paused && this.run === run && !run.closing;
  }

  acquire(id) {
    if (!text(id)) fail('invalid_run');
    if (this.failed) fail('browser_unavailable');
    if (this.paused) fail('browser_paused');
    if (this.run && (this.run.id !== id || this.run.closing)) fail('browser_busy');
    if (!this.run) {
      const run = this.run = { id, pending: new Set(), transport: null, closing: null };
      run.timer = setTimeout(() => { void this.release(id).catch(() => {}); }, this.ttl);
      run.timer.unref?.();
    }
    return this.run;
  }

  command(id, operation) {
    const run = this.acquire(id);
    if (run.pending.size >= 32) fail('pending_limit');
    const pending = Promise.resolve().then(() => {
      if (!this.allowed(run)) fail(this.paused ? 'browser_paused' : 'browser_unavailable');
      return operation(run);
    });
    run.pending.add(pending);
    pending.finally(() => run.pending.delete(pending)).catch(() => {});
    return pending;
  }

  drain(run) {
    if (!run) return Promise.resolve();
    if (!run.closing) run.closing = (async () => {
      let timer;
      try {
        await Promise.race([
          Promise.allSettled([...run.pending]),
          new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('drain_timeout')), this.drainMs); }),
        ]);
        await run.transport?.close();
        run.transport = null;
        if (this.failed) fail('browser_unavailable');
      } catch {
        this.fail();
        fail('browser_unavailable');
      } finally { clearTimeout(timer); }
    })();
    return run.closing;
  }

  async release(id) {
    if (!text(id)) fail('invalid_run');
    const run = this.run;
    if (!run || run.id !== id) return;
    await this.drain(run);
    clearTimeout(run.timer);
    if (this.run === run) this.run = null;
  }

  pause() {
    if (this.failed) fail('browser_unavailable');
    this.paused = true;
    if (!this.pausing) this.pausing = this.drain(this.run);
    return this.pausing;
  }

  async resume() {
    await this.pausing;
    if (this.failed) fail('browser_unavailable');
    if (this.run) this.run.closing = null;
    this.pausing = null;
    this.paused = false;
  }

  fail() {
    this.failed = true;
    this.paused = true;
    if (this.run) {
      clearTimeout(this.run.timer);
      this.run.transport?.abort();
    }
  }
}
