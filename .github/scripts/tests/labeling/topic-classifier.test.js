const assert = require('node:assert/strict');
const test = require('node:test');

const { classifyTopicLabels, ENDPOINT, MODEL } = require('../../labeling/topic-classifier.js');

const allowed = ['topic:mcp', 'topic:models'];

function response(content, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return { choices: [{ message: { content } }] };
    },
  };
}

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
