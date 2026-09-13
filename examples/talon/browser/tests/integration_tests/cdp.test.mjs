import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { randomBytes } from 'node:crypto';
import { writeFileSync, rmSync } from 'node:fs';
import { Coordinator } from '../../coordinator.mjs';
import { createBridge } from '../../bridge.mjs';

const enabled = process.env.TALON_TEST_LIVE_CDP === '1';

test('live Chrome HTTP and external CDP: tabs, flattened sessions, evaluation, file input, handoff', { skip: !enabled, timeout: 60000 }, async () => {
  const { WebSocket, WebSocketServer } = createRequire('/app/api/package.json')('ws');
  const coordinator = new Coordinator({ operator: 'test', identities: { telegram: 'synthetic' } });
  const owner = { operator_id: 'test', provider: 'telegram', sender_id: 'synthetic', conversation_id: 'controlled', run_id: 'cdp-test', background: false };
  const token = randomBytes(32).toString('base64url');
  const bridge = createBridge({ token, coordinator, WebSocket, WebSocketServer, controlHost: '127.0.0.1', viewerHost: '127.0.0.1', controlPort: 0, viewerPort: 0,
    ...(process.env.TALON_TEST_CDP_LOOPBACK === '1' ? { discoverURL: async () => 'ws://127.0.0.1:3000/' } : {}) });
  const file = `/files/jkb90-${randomBytes(8).toString('hex')}.txt`;
  await bridge.start();
  let client;
  try {
    const base = `http://127.0.0.1:${bridge.control.address().port}`;
    const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
    const post = async (path, envelope) => {
      const response = await fetch(`${base}/internal/browser/${path}`, { method: 'POST', headers, body: JSON.stringify(envelope) });
      const result = await response.json();
      assert.equal(response.status, 200, JSON.stringify(result));
      return result;
    };
    let lease = await post('actions', { action: 'acquire', owner, request_id: 'acquire' });
    let sequence = 0;
    const command = async (method, params = {}, session_id) => (await post('command', { owner, ...lease, request_id: `cmd-${++sequence}`, method, params, ...(session_id ? { session_id } : {}) })).result;
    const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true });
    await command('Page.navigate', { url: 'data:text/html,<title>Controlled</title><input type=file id=upload>' }, sessionId);
    assert.equal((await command('Runtime.evaluate', { expression: '6 * 7', returnByValue: true }, sessionId)).result.value, 42);
    assert.ok((await command('Target.getTargets')).targetInfos.some((target) => target.targetId === targetId));
    const { root } = await command('DOM.getDocument', {}, sessionId);
    const { nodeId } = await command('DOM.querySelector', { nodeId: root.nodeId, selector: '#upload' }, sessionId);
    writeFileSync(file, 'synthetic upload');
    await command('DOM.setFileInputFiles', { nodeId, files: [file] }, sessionId);
    assert.equal((await command('Runtime.evaluate', { expression: 'document.querySelector("#upload").files[0].size', returnByValue: true }, sessionId)).result.value, 16);
    await command('Target.closeTarget', { targetId });
    await post('actions', { ...lease, owner, action: 'release', request_id: 'release' });
    lease = await post('actions', { action: 'acquire', owner, request_id: 'acquire-ws' });
    client = new WebSocket(base.replace('http:', 'ws:') + '/internal/browser/cdp', { headers: { Authorization: `Bearer ${token}`, 'X-Browser-Lease': lease.lease_id, 'X-Browser-Generation': String(lease.generation), 'X-Browser-Owner': Buffer.from(JSON.stringify(owner)).toString('base64') } });
    await new Promise((resolve, reject) => { client.once('open', resolve); client.once('error', reject); });
    const response = new Promise((resolve) => client.on('message', (data) => { const value = JSON.parse(data); if (value.id === 991) resolve(value); }));
    client.send(JSON.stringify({ id: 991, method: 'Target.getTargets', params: {} }));
    assert.ok(Array.isArray((await response).result.targetInfos));
    const busy = await fetch(`${base}/internal/browser/command`, { method: 'POST', headers, body: JSON.stringify({ ...lease, owner, request_id: 'busy', method: 'Browser.getVersion', params: {} }) });
    assert.deepEqual(await busy.json(), { error: 'transport_busy' });
    const handoff = await post('actions', { ...lease, owner, action: 'handoff', request_id: 'handoff' });
    assert.equal(handoff.status, 'viewer_unavailable');
    assert.equal(client.readyState, WebSocket.CLOSED);
    await coordinator.cancel({ ...handoff, owner }, owner);
  } finally { client?.terminate(); await bridge.close(); rmSync(file, { force: true }); }
});
