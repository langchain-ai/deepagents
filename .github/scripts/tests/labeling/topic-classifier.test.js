const assert = require('node:assert/strict');
const test = require('node:test');

const { classifyTopicLabels, loadTopicLabels, ENDPOINT, MODEL } = require('../../labeling/topic-classifier.js');

const allowed = ['topic:mcp', 'topic:models'];
const descriptions = {
  'topic:mcp': 'Model Context Protocol support and behavior.',
  'topic:models': 'Model providers, model selection, and model configuration.',
  'priority:urgent': 'Not a topic description',
};

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
    apiKey: 'secret', fetchImpl, descriptions,
  });

  assert.deepEqual([...labels], ['topic:mcp']);
  assert.equal(request.url, ENDPOINT);
  const body = JSON.parse(request.options.body);
  assert.equal(body.model, MODEL);
  assert.equal(body.temperature, 0);
  assert.deepEqual(body.response_format, { type: 'json_object' });
  const taxonomy = body.messages[1].content.split('\n\nGitHub item:')[0].replace('Allowed labels and descriptions: ', '');
  assert.deepEqual(JSON.parse(taxonomy), allowed.map(name => ({ name, description: descriptions[name] })));
});

test('keeps at most three distinct allowed labels in relevance order', async () => {
  const topics = ['topic:prompts', 'topic:memory', 'topic:models', 'topic:middleware'];
  const labels = await classifyTopicLabels('text', topics, {
    apiKey: 'secret',
    fetchImpl: async () => response(JSON.stringify({
      labels: ['priority:urgent', topics[0], topics[0], ...topics.slice(1)],
    })),
  });

  assert.deepEqual([...labels], topics.slice(0, 3));
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

test('environment selects the provider and defaults to Groq', async t => {
  const previous = process.env.TOPIC_CLASSIFIER_PROVIDER;
  t.after(() => {
    if (previous === undefined) delete process.env.TOPIC_CLASSIFIER_PROVIDER;
    else process.env.TOPIC_CLASSIFIER_PROVIDER = previous;
  });
  for (const provider of [undefined, '', 'groq', 'semif', 'invalid']) {
    if (provider === undefined) delete process.env.TOPIC_CLASSIFIER_PROVIDER;
    else process.env.TOPIC_CLASSIFIER_PROVIDER = provider;
    const options = {
      apiKey: 'secret',
      fetchImpl: async (url) => {
        assert.equal(url, provider === 'semif' ? 'https://gateway.smith.langchain.com/v1/systemone' : ENDPOINT);
        return provider === 'semif'
          ? { ok: true, json: async () => ({ answers: { 'topic:mcp': { type: 'noul', noul: 0.95 } } }) }
          : response('{"labels":["topic:mcp"]}');
      },
    };
    if (provider === 'invalid') {
      options.fetchImpl = async () => assert.fail('invalid provider must not make a request');
      await assert.rejects(classifyTopicLabels('text', ['topic:mcp'], options), /must be groq or semif/);
    } else {
      assert.deepEqual([...await classifyTopicLabels('text', ['topic:mcp'], options)], ['topic:mcp']);
    }
  }
});
