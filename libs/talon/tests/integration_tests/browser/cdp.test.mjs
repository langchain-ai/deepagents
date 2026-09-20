import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { randomBytes } from 'node:crypto';
import { writeFileSync, rmSync } from 'node:fs';
import { Coordinator } from '../../../deepagents_talon/steel_runtime/coordinator.mjs';
import { createBridge } from '../../../deepagents_talon/steel_runtime/bridge.mjs';

const enabled = process.env.TALON_TEST_LIVE_CDP === '1';

test('live Chrome HTTP CDP: tabs, flattened sessions, evaluation, file input', { skip: !enabled, timeout: 60000 }, async () => {
  const { WebSocket } = createRequire(`${process.env.TALON_TEST_STEEL_DIR}/package.json`)('ws');
  const coordinator = new Coordinator();
  const token = randomBytes(32).toString('base64url');
  const bridge = createBridge({ token, coordinator, WebSocket, controlHost: '127.0.0.1', viewerHost: '127.0.0.1', controlPort: 0, viewerPort: 0,
    ...(process.env.TALON_TEST_CDP_LOOPBACK === '1' ? { discoverURL: async () => 'ws://127.0.0.1:3000/' } : {}) });
  const file = `/tmp/jkb90-${randomBytes(8).toString('hex')}.txt`;
  await bridge.start();
  try {
    const base = `http://127.0.0.1:${bridge.control.address().port}`;
    const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
    const post = async (path, envelope) => {
      const response = await fetch(`${base}/internal/browser/${path}`, { method: 'POST', headers, body: JSON.stringify(envelope) });
      const result = await response.json();
      assert.equal(response.status, 200, JSON.stringify(result));
      return result;
    };
    const command = async (method, params = {}, session_id) => (await post('command', {
      run_id: 'cdp-test', method, params, ...(session_id ? { session_id } : {}),
    })).result;
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
    await post('release', { run_id: 'cdp-test' });
  } finally { await bridge.close(); rmSync(file, { force: true }); }
});
