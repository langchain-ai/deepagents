const assert = require('node:assert/strict');
const test = require('node:test');

const { classifyTopicLabels, loadTopicLabels, ENDPOINT, MODEL } = require('../../labeling/topic-classifier.js');

const allowed = ['topic:mcp', 'topic:models'];

function response(content, status = 200, finishReason = 'stop') {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return { choices: [{ message: { content }, finish_reason: finishReason }] };
    },
  };
}

test('loads classifier choices from the cached manifest', () => {
  const labels = loadTopicLabels();
  assert.ok(labels.length > 0);
  assert.ok(labels.every(label => label.startsWith('topic:')));
});

test('classifies with the small open model and filters output to the allowlist', async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url, options };
    return response('{"labels":["topic:mcp","priority:urgent","topic:mcp"]}');
  };

  const labels = await classifyTopicLabels('MCP authentication fails', allowed, {
    apiKey: 'secret', fetchImpl,
  });

  assert.deepEqual([...labels], ['topic:mcp']);
  assert.equal(request.url, ENDPOINT);
  const body = JSON.parse(request.options.body);
  assert.equal(body.model, MODEL);
  assert.equal(body.temperature, 0);
  assert.deepEqual(body.response_format, { type: 'json_object' });
  assert.match(body.messages[1].content, /topic:models/);
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

test('allows reasoning to consume tokens before the final JSON', async () => {
  const fetchImpl = async (_url, options) => {
    const budget = JSON.parse(options.body).max_completion_tokens;
    // Simulate a completion that needs 2,000 reasoning tokens plus its answer.
    return budget >= 2100
      ? response('{"labels":["topic:mcp"]}')
      : response('', 200, 'length');
  };
  const labels = await classifyTopicLabels('MCP authentication fails', allowed, {
    apiKey: 'secret', fetchImpl,
  });
  assert.deepEqual([...labels], ['topic:mcp']);
});

for (const content of ['', '{"labels":["topic:mcp"', '{"labels":["topic:mcp"]}']) {
  test(`rejects length-limited output even when it looks valid: ${JSON.stringify(content)}`, async () => {
    await assert.rejects(
      classifyTopicLabels('text', allowed, {
        apiKey: 'secret', fetchImpl: async () => response(content, 200, 'length'),
      }),
      /exhausted its completion token budget/,
    );
  });
}

test('returns no labels for empty input without calling the model', async () => {
  const labels = await classifyTopicLabels(' ', allowed, {
    fetchImpl: async () => assert.fail('fetch should not be called'),
  });
  assert.deepEqual([...labels], []);
});

test('rejects failed and malformed model responses', async () => {
  await assert.rejects(
    classifyTopicLabels('text', allowed, { apiKey: 'secret', fetchImpl: async () => response('{}', 429) }),
    /HTTP 429/,
  );
  await assert.rejects(
    classifyTopicLabels('text', allowed, { apiKey: 'secret', fetchImpl: async () => response('not json') }),
    /JSON/,
  );
  await assert.rejects(
    classifyTopicLabels('text', allowed, { apiKey: 'secret', fetchImpl: async () => response('{}') }),
    /invalid labels/,
  );
});
