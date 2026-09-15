const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function workflow(name) {
  return fs.readFileSync(path.join(__dirname, '../../../workflows', name), 'utf8');
}

// Execute the checked-in github-script body with API doubles, without checking
// out PR code or requiring a YAML dependency for this network-free test suite.
function runStep(name, step, globals) {
  const source = workflow(name).split(`- name: ${step}\n`)[1];
  assert.ok(source, `Missing step: ${step}`);
  const lines = source.split('          script: |\n')[1].split('\n');
  const end = lines.findIndex(line => line.trim() && !line.startsWith('            '));
  const body = (end === -1 ? lines : lines.slice(0, end))
    .map(line => line.slice(12)).join('\n');
  return vm.runInNewContext(`(async () => {\n${body}\n})()`, {
    console: { log() {} }, core: { warning() {} }, ...globals,
  });
}

for (const label of [null, 'ci:keep-open', 'do-not-close']) {
  for (const boundary of [0, 1, 2]) {
    test(`waiting-on-author respects ${label} at fetch ${boundary}`, async () => {
      let fetch = 0;
      const closed = [];
      const item = () => ({
        number: 42, state: 'open', updated_at: '2020-01-01T00:00:00Z',
        user: { login: 'contributor', type: 'User' },
        labels: [
          { name: 'auto:waiting-on-author' },
          ...(label && fetch++ >= boundary ? [{ name: label }] : []),
        ],
      });
      const issues = {
        listForRepo: async () => [item()],
        get: async () => ({ data: item() }),
        listEvents: async () => [{
          event: 'labeled', label: { name: 'auto:waiting-on-author' },
          created_at: '2020-01-01T00:00:00Z',
        }],
        listComments: async () => [],
        createComment: async () => {},
        update: async params => closed.push(params.issue_number),
      };
      await runStep('waiting_on_author.yml', 'Close issues and PRs awaiting an author response', {
        context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } },
        github: { rest: { issues }, paginate: (method, params) => method(params) },
      });
      assert.deepEqual(closed, label ? [] : [42]);
    });
  }
}
