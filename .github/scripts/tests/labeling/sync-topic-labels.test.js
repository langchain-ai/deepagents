const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { syncTopicLabels } = require('../../labeling/sync-topic-labels.js');

test('writes sorted topic labels and excludes other namespaces', async () => {
  const output = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'topics-')), 'labels.json');
  const github = {
    rest: { issues: { listLabelsForRepo: async () => [
      { name: 'topic:zeta' }, { name: 'priority:high' }, { name: 'topic:alpha' },
    ] } },
    paginate: method => method(),
  };

  const topics = await syncTopicLabels(github, 'owner', 'repo', output);

  assert.deepEqual(topics, ['topic:alpha', 'topic:zeta']);
  assert.equal(fs.readFileSync(output, 'utf8'), '[\n  "topic:alpha",\n  "topic:zeta"\n]\n');
});
