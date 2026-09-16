const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');

// Both issue forms apply a `type:*` label instead of a GitHub Issue Type.
// Three separate files have to agree on those names, and each disagreement
// fails silently, so they are asserted here rather than left to review.
function formLabels() {
  const templates = path.join(__dirname, '../../../ISSUE_TEMPLATE');
  return fs.readdirSync(templates)
    .filter(name => name.endsWith('.yml'))
    .flatMap((name) => {
      const match = fs.readFileSync(path.join(templates, name), 'utf8')
        .match(/^labels:\s*\[([^\]]*)\]/m);
      return match
        ? match[1].split(',').map(s => s.trim().replace(/^["']|["']$/g, '')).filter(Boolean)
        : [];
    });
}

// The issue forms dropped GitHub Issue Types in favour of `labels:`, so
// close_unchecked_issues.yml Rule 0 now decides "was this submitted
// programmatically?" from a hardcoded set of type labels. The two literals
// live in different files with nothing binding them. If they ever disagree,
// every web-form issue from an external author is auto-closed with a
// "submitted programmatically" comment.
test('Rule 0 recognizes every label the issue forms apply', () => {
  const applied = formLabels();
  assert.ok(applied.length >= 2, 'expected the bug and feature forms to apply labels');

  const rule0 = fs.readFileSync(
    path.join(__dirname, '../../../workflows/close_unchecked_issues.yml'), 'utf8',
  ).match(/const formLabels = new Set\((\[[^\]]*\])\)/);
  assert.ok(rule0, 'close_unchecked_issues.yml must declare formLabels');
  const recognized = new Set(JSON.parse(rule0[1].replace(/'/g, '"')));

  for (const label of applied) {
    assert.ok(
      recognized.has(label),
      `issue forms apply '${label}', but Rule 0 does not recognize it — ` +
      'every web-form issue carrying it would be closed as programmatic',
    );
  }
});

// A form label that does not exist on the repo is silently skipped by GitHub,
// which lands the issue in exactly the state Rule 0 treats as programmatic.
// So every form label must also be one the labeler can create with a
// description, per labelDescriptions in pr-labeler-config.json.
test('every issue-form label has a description in the labeler config', () => {
  const config = JSON.parse(fs.readFileSync(
    path.join(__dirname, '../../labeling/pr-labeler-config.json'), 'utf8',
  ));
  const applied = formLabels();
  for (const label of applied) {
    assert.ok(
      config.labelDescriptions?.[label],
      `issue forms apply '${label}' but labelDescriptions has no entry for it`,
    );
  }
});
