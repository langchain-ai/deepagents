const assert = require('node:assert/strict');
const test = require('node:test');

const { classifyTopicLabels: classify, loadTopicLabels } = require('../../labeling/topic-classifier.js');
const { ENDPOINT, MODEL } = require('../../labeling/semif-topic-classifier.js');
const classifyTopicLabels = (text, labels, options) => classify(text, labels, { ...options, provider: 'semif' });

const allowed = ['topic:mcp', 'topic:models'];

function response(scores, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return { answers: Object.fromEntries(Object.entries(scores).map(([label, noul]) => [label, { type: 'noul', noul }])) };
    },
  };
}

test('loads classifier choices from the cached manifest', () => {
  const labels = loadTopicLabels();
  assert.ok(labels.length > 0);
  assert.ok(labels.every(label => label.startsWith('topic:')));
});

test('uses the gateway System One contract and ignores unsolicited labels', async () => {
  let request;
  const labels = await classifyTopicLabels('MCP authentication fails', allowed, {
    apiKey: 'secret',
    fetchImpl: async (url, options) => {
      request = { url, options };
      return response({ 'topic:mcp': 0.95, 'topic:models': 0.3, 'priority:urgent': 1 });
    },
  });
  assert.deepEqual([...labels], ['topic:mcp']);
  assert.equal(request.url, ENDPOINT);
  assert.equal(request.options.headers.Authorization, 'Bearer secret');
  const body = JSON.parse(request.options.body);
  assert.equal(body.model, MODEL);
  assert.equal(body.state, 'MCP authentication fails');
  assert.deepEqual(Object.keys(body.questions), allowed);
  assert.equal(body.questions['topic:mcp'].type, 'noul');
});

test('sends the configured workspace header and omits it when unset or empty', async t => {
  const previous = process.env.LANGSMITH_WORKSPACE_ID;
  t.after(() => {
    if (previous === undefined) delete process.env.LANGSMITH_WORKSPACE_ID;
    else process.env.LANGSMITH_WORKSPACE_ID = previous;
  });
  for (const workspace of ['00000000-0000-4000-8000-000000000001', '', undefined]) {
    if (workspace === undefined) delete process.env.LANGSMITH_WORKSPACE_ID;
    else process.env.LANGSMITH_WORKSPACE_ID = workspace;
    await classifyTopicLabels('text', ['topic:mcp'], {
      apiKey: 'secret',
      fetchImpl: async (_url, options) => {
        assert.equal(options.headers['X-Tenant-ID'], workspace || undefined);
        assert.equal(Object.hasOwn(options.headers, 'X-Tenant-ID'), Boolean(workspace));
        return response({ 'topic:mcp': 0.95 });
      },
    });
  }
});

test('ranks distinct labels by probability and caps them at three', async () => {
  const scores = { a: 0.8, b: 0.99, c: 0.9, d: 0.95, e: 0.79 };
  const labels = await classifyTopicLabels('text', [...Object.keys(scores), 'b'], {
    apiKey: 'secret', fetchImpl: async () => response(scores),
  });
  assert.deepEqual([...labels], ['b', 'd', 'c']);
});

test('abstains below the threshold and accepts the threshold boundary', async () => {
  for (const [score, expected] of [[0.79, []], [0.8, ['topic:mcp']]]) {
    const labels = await classifyTopicLabels('text', ['topic:mcp'], {
      apiKey: 'secret', fetchImpl: async () => response({ 'topic:mcp': score }),
    });
    assert.deepEqual([...labels], expected);
  }
});

test('batches at most 32 questions and ranks across batches', async () => {
  const topics = Array.from({ length: 33 }, (_, i) => `topic:${i}`);
  const sizes = [];
  const labels = await classifyTopicLabels('text', topics, {
    apiKey: 'secret',
    fetchImpl: async (_url, options) => {
      const names = Object.keys(JSON.parse(options.body).questions);
      sizes.push(names.length);
      return response(Object.fromEntries(names.map(name => [name, name === 'topic:32' ? 0.99 : 0.1])));
    },
  });
  assert.deepEqual(sizes, [32, 1]);
  assert.deepEqual([...labels], ['topic:32']);
});

test('keeps the timeout active while reading the response body', async () => {
  const fetchImpl = async (_url, options) => ({
    ok: true,
    async json() {
      await new Promise((resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(options.signal.reason));
      });
    },
  });
  await assert.rejects(
    classifyTopicLabels('text', allowed, { apiKey: 'secret', fetchImpl, timeoutMs: 1 }),
    { name: 'AbortError' },
  );
});

test('empty input or taxonomy does not call the gateway', async () => {
  for (const [text, topics] of [[' ', allowed], ['text', []]]) {
    const labels = await classifyTopicLabels(text, topics, {
      fetchImpl: async () => assert.fail('fetch should not be called'),
    });
    assert.deepEqual([...labels], []);
  }
});

test('rejects failed, missing, and invalid probability responses', async () => {
  await assert.rejects(
    classifyTopicLabels('text', allowed, { apiKey: 'secret', fetchImpl: async () => response({}, 429) }),
    /HTTP 429/,
  );
  for (const answer of [undefined, { type: 'choice', noul: 0.9 }, ...[null, '0.9', -1, 2, NaN].map(noul => ({ type: 'noul', noul }))]) {
    await assert.rejects(
      classifyTopicLabels('text', ['topic:mcp'], {
        apiKey: 'secret',
        fetchImpl: async () => ({ ok: true, json: async () => ({ answers: { 'topic:mcp': answer } }) }),
      }),
      /invalid probabilities/,
    );
  }
});
