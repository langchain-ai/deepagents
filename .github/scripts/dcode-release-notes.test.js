'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const releaseNotes = require('./dcode-release-notes.js');

const BOT = { login: 'dcode-release-bot', id: 42 };
const HEAD = 'a'.repeat(40);
const APPLIED_HEAD = 'b'.repeat(40);
const VERSION = '0.1.35';
const OVERRIDE_UPDATED_AT = '2026-07-09T12:00:00Z';
const APPLIED_UPDATED_AT = '2026-07-09T12:05:00Z';
const HEADING = `## [${VERSION}](https://github.com/langchain-ai/deepagents/compare/deepagents-code==0.1.34...deepagents-code==${VERSION}) (2026-07-09)`;
const GENERATED_SECTION = `${HEADING}\n\n### Features\n\n* **code:** add useful feature ([#1](https://github.com/langchain-ai/deepagents/issues/1))\n`;
const CURATED_SECTION = `${HEADING}\n\n### Features\n\n* Add a useful feature ([#1](https://github.com/langchain-ai/deepagents/issues/1))\n`;

function changelog(section = GENERATED_SECTION) {
  return `# Changelog\n\n${section}\n## [0.1.34](https://example.test) (2026-07-01)\n\n* Older\n`;
}

function releasePr(overrides = {}) {
  return {
    number: 123,
    title: `release(deepagents-code): ${VERSION}`,
    state: 'open',
    draft: false,
    body: `Release notes preview\n\n${GENERATED_SECTION}\n_End release notes preview._\n`,
    head: {
      ref: releaseNotes.RELEASE_BRANCH,
      sha: HEAD,
      repo: { full_name: 'langchain-ai/deepagents' },
    },
    base: { ref: 'main', repo: { full_name: 'langchain-ai/deepagents' } },
    labels: [],
    ...overrides,
  };
}

function overrideComment({ id = 10, section = CURATED_SECTION, fingerprint = releaseNotes.sha256(GENERATED_SECTION), head = HEAD, updatedAt = OVERRIDE_UPDATED_AT } = {}) {
  return {
    id,
    updated_at: updatedAt,
    user: BOT,
    body: [
      '<!-- dcode-release-notes-override',
      'package: deepagents-code',
      `version: ${VERSION}`,
      `release-pr-head: ${head}`,
      `release-heading-hash: ${releaseNotes.sha256(HEADING)}`,
      `changelog-fingerprint: ${fingerprint}`,
      'state: draft',
      '-->',
      'instructions',
      releaseNotes.CONTENT_START,
      section.trimEnd(),
      releaseNotes.CONTENT_END,
    ].join('\n'),
  };
}

function appliedComment({ id = 20, overrideId = 10, overrideUpdatedAt = OVERRIDE_UPDATED_AT, fingerprint = releaseNotes.sha256(GENERATED_SECTION), contentHash = releaseNotes.sha256(CURATED_SECTION), sourceHead = HEAD, appliedHead = APPLIED_HEAD, updatedAt = APPLIED_UPDATED_AT } = {}) {
  return {
    id,
    updated_at: updatedAt,
    user: BOT,
    body: [
      '<!-- dcode-release-notes-applied',
      'package: deepagents-code',
      `version: ${VERSION}`,
      `source-head: ${sourceHead}`,
      `applied-head: ${appliedHead}`,
      `changelog-fingerprint: ${fingerprint}`,
      `override-comment-id: ${overrideId}`,
      `override-comment-updated-at: ${overrideUpdatedAt}`,
      `override-content-hash: ${contentHash}`,
      'state: applied',
      '-->',
    ].join('\n'),
  };
}

function makeCore() {
  return {
    failed: null,
    infos: [],
    warnings: [],
    info(message) { this.infos.push(message); },
    warning(message) { this.warnings.push(message); },
    setFailed(message) { this.failed = message; },
  };
}

function makeGithub({ pr = releasePr(), comments = [], permission = 'write', adminFlag = permission === 'admin', authenticated = BOT, files = new Map(), comparison = 'ahead', malformedContent = false, onGetPr = null, onListComments = null } = {}) {
  const calls = {
    createBlob: [],
    createComment: [],
    createCommit: [],
    createTree: [],
    getCommit: [],
    getContent: [],
    updateComment: [],
    updatePr: [],
    updateRef: [],
  };
  let livePr = structuredClone(pr);
  let getPrCount = 0;
  let listCommentsCount = 0;
  const github = {
    rest: {
      pulls: {
        get: async () => {
          getPrCount += 1;
          if (onGetPr) onGetPr({ count: getPrCount, pr: livePr });
          return { data: structuredClone(livePr) };
        },
        update: async params => {
          calls.updatePr.push(params);
          livePr.body = params.body;
          return { data: structuredClone(livePr) };
        },
      },
      issues: {
        listComments: async () => {
          listCommentsCount += 1;
          if (onListComments) onListComments({ count: listCommentsCount, comments });
          return { data: structuredClone(comments) };
        },
        createComment: async params => {
          calls.createComment.push(params);
          const comment = { id: 100 + calls.createComment.length, updated_at: APPLIED_UPDATED_AT, user: BOT, body: params.body };
          comments.push(comment);
          return { data: comment };
        },
        updateComment: async params => {
          calls.updateComment.push(params);
          const comment = comments.find(item => item.id === params.comment_id);
          if (comment) {
            comment.body = params.body;
            comment.updated_at = APPLIED_UPDATED_AT;
          }
          return { data: comment ?? { id: params.comment_id, updated_at: APPLIED_UPDATED_AT, user: BOT, body: params.body } };
        },
      },
      repos: {
        getCollaboratorPermissionLevel: async () => ({ data: { permission, user: { permissions: { admin: adminFlag } } } }),
        getContent: async params => {
          calls.getContent.push(params);
          if (malformedContent) return { data: [] };
          const fallback = params.ref === APPLIED_HEAD ? changelog(CURATED_SECTION) : changelog();
          const content = files.get(params.ref) ?? fallback;
          return { data: { type: 'file', encoding: 'base64', content: Buffer.from(content).toString('base64') } };
        },
        compareCommitsWithBasehead: async () => ({ data: { status: comparison } }),
      },
      git: {
        getCommit: async params => {
          calls.getCommit.push(params);
          return { data: { tree: { sha: 'tree-base' } } };
        },
        createBlob: async params => {
          calls.createBlob.push(params);
          return { data: { sha: 'blob-curated' } };
        },
        createTree: async params => {
          calls.createTree.push(params);
          return { data: { sha: 'tree-curated' } };
        },
        createCommit: async params => {
          calls.createCommit.push(params);
          return { data: { sha: APPLIED_HEAD } };
        },
        updateRef: async params => {
          calls.updateRef.push(params);
          livePr.head.sha = params.sha;
          return { data: { object: { sha: params.sha } } };
        },
      },
      users: {
        getAuthenticated: async () => ({ data: authenticated }),
      },
    },
    paginate: async (method, params) => (await method(params)).data,
  };
  return { github, calls, getPr: () => livePr, setPr: value => { livePr = structuredClone(value); } };
}

function tempWorkspace(section = GENERATED_SECTION) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'dcode-release-test-'));
  const file = path.join(root, releaseNotes.CHANGELOG_PATH);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, changelog(section));
  return { root, file };
}

test('identifies only the exact deepagents-code release title and branch', () => {
  assert.equal(releaseNotes.releaseVersion(`release(deepagents-code): ${VERSION}`), VERSION);
  assert.equal(releaseNotes.releaseVersion(`release(deepagents): ${VERSION}`), null);
  assert.equal(releaseNotes.isReleasePr(releasePr()), true);
  assert.equal(releaseNotes.isReleasePr(releasePr({ head: { ...releasePr().head, ref: `${releaseNotes.RELEASE_BRANCH}-extra` } })), false);
  assert.equal(releaseNotes.isReleasePr(releasePr({ base: { ref: 'v0.1', repo: { full_name: 'langchain-ai/deepagents' } } })), false);
  assert.equal(releaseNotes.isReleasePr(releasePr({ head: { ...releasePr().head, repo: null } })), false);
});

test('extracts and replaces exactly one version section', () => {
  const section = releaseNotes.extractVersionSection(changelog(), VERSION);
  assert.equal(section, releaseNotes.canonical(GENERATED_SECTION));
  const replaced = releaseNotes.replaceVersionSection(changelog(), VERSION, CURATED_SECTION);
  assert.equal(releaseNotes.extractVersionSection(replaced, VERSION), releaseNotes.canonical(CURATED_SECTION));
  assert.match(replaced, /## \[0\.1\.34\]/);
  assert.throws(() => releaseNotes.extractVersionSection('# Changelog\n', VERSION), /exactly one/);
});

test('requires exactly one PR-body preview terminator', () => {
  const valid = `Release notes preview\n\n${GENERATED_SECTION}\n_End release notes preview._\nFooter\n`;
  assert.equal(releaseNotes.extractPreviewSection(valid, VERSION), releaseNotes.canonical(GENERATED_SECTION));
  assert.throws(
    () => releaseNotes.extractPreviewSection(valid.replace('\n_End release notes preview._', ''), VERSION),
    /exactly one release-notes preview terminator/,
  );
  assert.throws(
    () => releaseNotes.extractPreviewSection(`${valid}\n_End release notes preview._\n`, VERSION),
    /exactly one release-notes preview terminator/,
  );
});

test('fingerprint changes only with the selected release section', () => {
  const original = releaseNotes.extractVersionSection(changelog(), VERSION);
  const unrelated = `${changelog()}\n## unrelated package text\n`;
  assert.equal(releaseNotes.sha256(original), releaseNotes.sha256(releaseNotes.extractVersionSection(unrelated, VERSION)));
  const changed = GENERATED_SECTION.replace('useful feature', 'new entry');
  assert.notEqual(releaseNotes.sha256(original), releaseNotes.sha256(changed));
  assert.equal(releaseNotes.sha256(original.replace(/\n/g, '\r\n')), releaseNotes.sha256(original));
});

test('parses commands in surrounding text and rejects ambiguous comments', () => {
  assert.equal(releaseNotes.commandFromComment('@dcode-release-bot draft'), 'draft');
  assert.equal(releaseNotes.commandFromComment('Please @dcode-release-bot apply when ready.'), 'apply');
  assert.equal(releaseNotes.commandFromComment('@dcode-release-bot draft after fixing the notes'), 'draft');
  assert.equal(releaseNotes.commandFromComment('not@dcode-release-bot apply'), null);
  assert.equal(releaseNotes.commandFromComment('@dcode-release-bot application'), null);
  assert.equal(releaseNotes.commandFromComment('@dcode-release-bot draft and @dcode-release-bot apply'), null);
});

test('trusts only marked comments from the configured bot identity', () => {
  const valid = overrideComment();
  const impostor = { ...overrideComment({ id: 11 }), user: { login: BOT.login, id: 99 } };
  assert.equal(releaseNotes.latestOverride([valid, impostor], BOT.login, BOT.id, VERSION).comment.id, 10);
  assert.equal(releaseNotes.latestOverride([impostor], BOT.login, BOT.id, VERSION), null);
  assert.equal(releaseNotes.parseOverrideComment({ ...valid, body: valid.body.replace('state: draft', 'state: applied') }), null);
});

test('manual commands require write permission and ready status', async () => {
  const context = {
    eventName: 'issue_comment',
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: {
      action: 'created',
      issue: { number: 123, pull_request: {} },
      // A read-access collaborator: an insider (so feedback is allowed) who still
      // lacks the write permission the command requires.
      comment: { body: '@dcode-release-bot apply', user: { login: 'reader' }, author_association: 'COLLABORATOR' },
    },
  };
  const denied = makeGithub({ permission: 'read' });
  assert.equal((await releaseNotes.validateTrigger({ github: denied.github, context, core: makeCore() })).shouldRun, false);
  assert.match(denied.calls.createComment[0].body, /write access/);

  const draft = makeGithub({ pr: releasePr({ draft: true }) });
  assert.equal((await releaseNotes.validateTrigger({ github: draft.github, context, core: makeCore() })).shouldRun, false);
  assert.match(draft.calls.createComment[0].body, /ready for review/);
});

test('manual commands ignore comments authored by the configured bot', async () => {
  const context = {
    eventName: 'issue_comment',
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: {
      action: 'created',
      issue: { number: 123, pull_request: {} },
      comment: { body: '@dcode-release-bot draft', user: BOT },
    },
  };
  const run = makeGithub({ permission: 'write' });
  const result = await releaseNotes.validateTrigger({
    github: run.github,
    context,
    core: makeCore(),
    botLogin: BOT.login,
    botId: BOT.id,
  });
  assert.equal(result.shouldRun, false);
  assert.equal(run.calls.createComment.length, 0);
});

test('ready_for_review automatically validates as draft command', async () => {
  const { github } = makeGithub();
  const context = {
    eventName: 'pull_request_target',
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: { action: 'ready_for_review', pull_request: { number: 123 } },
  };
  const result = await releaseNotes.validateTrigger({ github, context, core: makeCore() });
  assert.deepEqual(result, {
    shouldRun: true,
    command: 'draft',
    number: 123,
    version: VERSION,
    head: HEAD,
    branch: releaseNotes.RELEASE_BRANCH,
  });
});

test('prepares agent input from the exact validated head', async t => {
  const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), 'dcode-release-runner-'));
  t.after(() => fs.rmSync(runnerTemp, { recursive: true, force: true }));
  const { github, calls } = makeGithub();
  const prepared = await releaseNotes.prepareDraft({
    github,
    owner: 'langchain-ai',
    repo: 'deepagents',
    number: 123,
    expectedHead: HEAD,
    runnerTemp,
  });
  assert.match(fs.readFileSync(prepared.input, 'utf8'), /untrusted source material/);
  assert.equal(JSON.parse(fs.readFileSync(prepared.state, 'utf8')).fingerprint, releaseNotes.sha256(GENERATED_SECTION));
  assert.equal(calls.getContent.length, 1);
  assert.equal(calls.getContent[0].ref, HEAD);
});

test('posts a bot-authored draft and refuses stale agent output', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dcode-release-post-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const state = path.join(dir, 'state.json');
  const output = path.join(dir, 'output.md');
  fs.writeFileSync(state, JSON.stringify({ number: 123, version: VERSION, head: HEAD, fingerprint: releaseNotes.sha256(GENERATED_SECTION), heading: HEADING }));
  fs.writeFileSync(output, '### Features\n\n* Add a useful feature.\n');
  const { github, calls } = makeGithub();
  await releaseNotes.postDraft({ github, owner: 'langchain-ai', repo: 'deepagents', stateFile: state, outputFile: output, login: BOT.login, id: BOT.id });
  assert.equal(calls.createComment.length, 1);
  assert.match(calls.createComment[0].body, /changelog-fingerprint:/);
  assert.match(calls.createComment[0].body, /@dcode-release-bot apply/);

  const stale = makeGithub({ pr: releasePr({ head: { ...releasePr().head, sha: 'c'.repeat(40) } }) });
  await assert.rejects(
    releaseNotes.postDraft({ github: stale.github, owner: 'langchain-ai', repo: 'deepagents', stateFile: state, outputFile: output, login: BOT.login, id: BOT.id }),
    /changed while notes were being drafted/,
  );
});

test('prepare apply replaces only the changelog section and records immutable hashes', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const stateFile = path.join(workspace.root, 'apply.json');
  const { github } = makeGithub({ comments: [overrideComment()] });
  const state = await releaseNotes.prepareApply({
    github,
    owner: 'langchain-ai',
    repo: 'deepagents',
    number: 123,
    expectedHead: HEAD,
    changelogFile: workspace.file,
    stateFile,
    login: BOT.login,
    id: BOT.id,
  });
  assert.equal(releaseNotes.extractVersionSection(fs.readFileSync(workspace.file, 'utf8'), VERSION), releaseNotes.canonical(CURATED_SECTION));
  assert.equal(state.overrideId, '10');
  assert.equal(state.overrideUpdatedAt, OVERRIDE_UPDATED_AT);
  assert.equal(state.contentHash, releaseNotes.sha256(CURATED_SECTION));
  assert.equal(state.originalBodyHash, releaseNotes.exactSha256(releasePr().body));
  assert.match(state.body, /\* Add a useful feature/);
});

test('apply rejects concurrent PR-body and override revisions', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const stateFile = path.join(workspace.root, 'apply.json');
  const comments = [overrideComment()];
  const run = makeGithub({ comments });
  await releaseNotes.prepareApply({
    github: run.github,
    owner: 'langchain-ai',
    repo: 'deepagents',
    number: 123,
    expectedHead: HEAD,
    changelogFile: workspace.file,
    stateFile,
    login: BOT.login,
    id: BOT.id,
  });

  run.setPr(releasePr({
    body: `${releasePr().body}\nMaintainer edit\n`,
    head: { ...releasePr().head, sha: APPLIED_HEAD },
  }));
  await assert.rejects(
    releaseNotes.publishAppliedState({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', stateFile, appliedHead: APPLIED_HEAD, login: BOT.login, id: BOT.id }),
    /Release PR changed while apply was preparing/,
  );

  run.setPr(releasePr({ head: { ...releasePr().head, sha: APPLIED_HEAD } }));
  comments[0].updated_at = '2026-07-09T12:01:00Z';
  await assert.rejects(
    releaseNotes.publishAppliedState({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', stateFile, appliedHead: APPLIED_HEAD, login: BOT.login, id: BOT.id }),
    /draft changed while apply was preparing/,
  );
});

test('prepare apply preserves the generated release heading', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const altered = CURATED_SECTION.replace('(2026-07-09)', '(2026-07-10)');
  const { github } = makeGithub({ comments: [overrideComment({ section: altered })] });
  await assert.rejects(
    releaseNotes.prepareApply({ github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD, changelogFile: workspace.file, stateFile: path.join(workspace.root, 'state.json'), login: BOT.login, id: BOT.id }),
    /Keep the generated release version heading unchanged/,
  );
});

test('prepare apply rejects new generated entries but permits idempotent recovery', async t => {
  const changed = GENERATED_SECTION.replace('useful feature', 'brand new feature');
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const files = new Map([[HEAD, changelog(changed)]]);
  const { github } = makeGithub({ comments: [overrideComment()], files });
  await assert.rejects(
    releaseNotes.prepareApply({ github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD, changelogFile: workspace.file, stateFile: path.join(workspace.root, 'state.json'), login: BOT.login, id: BOT.id }),
    /New generated release entries/,
  );

  files.set(HEAD, changelog(CURATED_SECTION));
  const recovered = await releaseNotes.prepareApply({ github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD, changelogFile: workspace.file, stateFile: path.join(workspace.root, 'state.json'), login: BOT.login, id: BOT.id });
  assert.equal(recovered.alreadyApplied, true);
});

test('required check passes only when applied metadata, changelog, body, and ancestry match', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment()] });
  const core = makeCore();
  const result = await releaseNotes.checkCuratedState({
    github,
    context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } },
    core,
    number: 123,
    login: BOT.login,
    id: BOT.id,
  });
  assert.equal(result.status, 'passed', core.failed);
  assert.equal(core.failed, null);
});

test('required check accepts apply after unrelated branch advancement', async () => {
  const advancedHead = 'c'.repeat(40);
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment({ sourceHead: advancedHead })] });
  const core = makeCore();
  const result = await releaseNotes.checkCuratedState({
    github,
    context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } },
    core,
    number: 123,
    login: BOT.login,
    id: BOT.id,
  });
  assert.equal(result.status, 'passed', core.failed);
  assert.equal(core.failed, null);
});

test('required check binds to the expected head and rejects malformed target titles', async () => {
  const staleHead = makeGithub();
  const staleCore = makeCore();
  await releaseNotes.checkCuratedState({ github: staleHead.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: staleCore, number: 123, login: BOT.login, id: BOT.id, expectedHead: APPLIED_HEAD });
  assert.match(staleCore.failed, /head changed before/);

  const malformed = makeGithub({ pr: releasePr({ title: `release(deepagents): ${VERSION}` }) });
  const malformedCore = makeCore();
  await releaseNotes.checkCuratedState({ github: malformed.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: malformedCore, number: 123, login: BOT.login, id: BOT.id });
  assert.match(malformedCore.failed, /title does not match/);
});

test('required check rejects in-flight bypass and comment revisions', async () => {
  const bypass = makeGithub({
    pr: releasePr({ labels: [{ name: releaseNotes.BYPASS_LABEL }] }),
    onGetPr: ({ count, pr }) => {
      if (count === 2) pr.labels = [];
    },
  });
  const bypassCore = makeCore();
  await releaseNotes.checkCuratedState({ github: bypass.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: bypassCore, number: 123, login: BOT.login, id: BOT.id });
  assert.match(bypassCore.failed, /changed while/);

  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const comments = [overrideComment(), appliedComment()];
  const revised = makeGithub({
    pr,
    comments,
    onListComments: ({ count }) => {
      if (count === 2) comments[0].updated_at = '2026-07-09T12:01:00Z';
    },
  });
  const revisedCore = makeCore();
  await releaseNotes.checkCuratedState({ github: revised.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: revisedCore, number: 123, login: BOT.login, id: BOT.id });
  assert.match(revisedCore.failed, /comments changed while/);
});

test('required check fails after override edit and after release-please overwrite', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const edited = CURATED_SECTION.replace('useful', 'excellent');
  const editedBotComment = overrideComment({ section: edited });
  const editedRun = makeGithub({ pr, comments: [editedBotComment, appliedComment()] });
  const editedCore = makeCore();
  await releaseNotes.checkCuratedState({ github: editedRun.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: editedCore, number: 123, login: BOT.login, id: BOT.id });
  assert.match(editedCore.failed, /override changed after apply/);

  const restoredRun = makeGithub({
    pr,
    comments: [
      overrideComment({ updatedAt: '2026-07-09T12:01:00Z' }),
      appliedComment(),
    ],
  });
  const restoredCore = makeCore();
  await releaseNotes.checkCuratedState({ github: restoredRun.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: restoredCore, number: 123, login: BOT.login, id: BOT.id });
  assert.match(restoredCore.failed, /override was revised after apply/);

  const newGenerated = GENERATED_SECTION.replace('useful feature', 'new generated entry');
  const overwrittenPr = releasePr({
    head: { ...releasePr().head, sha: 'd'.repeat(40) },
    body: `Release notes preview\n\n${newGenerated}\n_End release notes preview._\n`,
  });
  const files = new Map([[overwrittenPr.head.sha, changelog(newGenerated)]]);
  const overwritten = makeGithub({ pr: overwrittenPr, comments: [overrideComment(), appliedComment()], files });
  const overwrittenCore = makeCore();
  await releaseNotes.checkCuratedState({ github: overwritten.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: overwrittenCore, number: 123, login: BOT.login, id: BOT.id });
  assert.match(overwrittenCore.failed, /changelog does not contain/);
  assert.equal(overwritten.calls.createComment.length, 1);
  assert.match(overwritten.calls.createComment[0].body, /New generated release entries appeared/);

  await releaseNotes.checkCuratedState({ github: overwritten.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: makeCore(), number: 123, login: BOT.login, id: BOT.id });
  assert.equal(overwritten.calls.createComment.length, 1);
});

test('required check warns when generated entries change after draft before apply', async () => {
  const newGenerated = GENERATED_SECTION.replace('useful feature', 'new generated entry');
  const changedHead = 'd'.repeat(40);
  const pr = releasePr({
    head: { ...releasePr().head, sha: changedHead },
    body: `Release notes preview\n\n${newGenerated}\n_End release notes preview._\n`,
  });
  const comments = [overrideComment()];
  const files = new Map([[changedHead, changelog(newGenerated)]]);
  const { github, calls } = makeGithub({ pr, comments, files });

  const core = makeCore();
  const result = await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.equal(result.status, 'missing');
  assert.match(core.failed, /draft and then/);
  assert.equal(calls.createComment.length, 1);
  assert.match(calls.createComment[0].body, /New generated release entries appeared/);

  await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: makeCore(), number: 123, login: BOT.login, id: BOT.id });
  assert.equal(calls.createComment.length, 1);
});

test('draft, unrelated package, and bypass label pass without metadata', async () => {
  for (const pr of [
    releasePr({ draft: true }),
    releasePr({
      title: `release(deepagents): ${VERSION}`,
      head: { ...releasePr().head, ref: 'release-please--branches--main--components--deepagents' },
    }),
    releasePr({ labels: [{ name: releaseNotes.BYPASS_LABEL }] }),
  ]) {
    const { github } = makeGithub({ pr });
    const core = makeCore();
    await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
    assert.equal(core.failed, null);
  }
});

test('validateDraftOutput rejects empty, metadata, and heading content', () => {
  assert.throws(() => releaseNotes.validateDraftOutput('   \n'), /empty release notes/);
  assert.throws(() => releaseNotes.validateDraftOutput('<!-- dcode-release-notes-applied\npackage: x\n-->\nnotes'), /only section content/);
  assert.throws(() => releaseNotes.validateDraftOutput(`${HEADING}\n\n* smuggled heading`), /only section content/);
  assert.equal(releaseNotes.validateDraftOutput('### Features\n\n* Real note.\n'), '### Features\n\n* Real note.\n');
});

test('postDraft fails when the token is not the configured bot', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dcode-release-auth-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const state = path.join(dir, 'state.json');
  const output = path.join(dir, 'output.md');
  fs.writeFileSync(state, JSON.stringify({ number: 123, version: VERSION, head: HEAD, fingerprint: releaseNotes.sha256(GENERATED_SECTION), heading: HEADING }));
  fs.writeFileSync(output, '### Features\n\n* Add a useful feature.\n');
  const { github, calls } = makeGithub({ authenticated: { login: 'someone-else', id: 7 } });
  await assert.rejects(
    releaseNotes.postDraft({ github, owner: 'langchain-ai', repo: 'deepagents', stateFile: state, outputFile: output, login: BOT.login, id: BOT.id }),
    /authenticated as someone-else/,
  );
  assert.equal(calls.createComment.length, 0);
  assert.equal(calls.updateComment.length, 0);
});

test('required check fails when curated draft or applied metadata is missing', async () => {
  const { github } = makeGithub({ comments: [] });
  const core = makeCore();
  const result = await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.equal(result.status, 'missing');
  assert.match(core.failed, /draft and then/);
});

test('required check fails when applied metadata references an older override', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment({ overrideId: 999 })] });
  const core = makeCore();
  await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.match(core.failed, /older override comment/);
});

test('required check fails when the applied draft is not based on the latest override', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const staleSource = 'f'.repeat(40);
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment({ sourceHead: staleSource })], comparison: 'diverged' });
  const core = makeCore();
  await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.match(core.failed, /not based on the latest curated draft/);
});

test('required check fails when the applied commit is not an ancestor of the head', async () => {
  const headSha = 'e'.repeat(40);
  const pr = releasePr({
    head: { ...releasePr().head, sha: headSha },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const files = new Map([[headSha, changelog(CURATED_SECTION)]]);
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment()], files, comparison: 'diverged' });
  const core = makeCore();
  await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.match(core.failed, /not an ancestor of the current release PR head/);
});

test('parseMetadata rejects malformed, unknown, duplicate, and missing-field comments', () => {
  const valid = overrideComment();
  assert.ok(releaseNotes.parseOverrideComment(valid));

  // Missing prefix: body does not start with the marker at byte 0.
  assert.equal(releaseNotes.parseOverrideComment({ ...valid, body: `noise\n${valid.body}` }), null);
  // Missing close marker.
  assert.equal(releaseNotes.parseOverrideComment({ ...valid, body: valid.body.replace('\n-->', '') }), null);
  // Unknown field inside the metadata block.
  assert.equal(releaseNotes.parseOverrideComment({ ...valid, body: valid.body.replace('state: draft', 'state: draft\nevil: 1') }), null);
  // Duplicate allowed field: a second value must not silently win over the first.
  assert.equal(releaseNotes.parseOverrideComment({ ...valid, body: valid.body.replace('state: draft', `version: ${VERSION}\nstate: draft`) }), null);
  // Malformed line without a "key: value" shape.
  assert.equal(releaseNotes.parseOverrideComment({ ...valid, body: valid.body.replace('state: draft', 'state: draft\ngarbage') }), null);
  // Missing a required field.
  assert.equal(releaseNotes.parseOverrideComment({ ...valid, body: valid.body.replace('state: draft\n', '') }), null);

  // The same strictness applies to applied comments.
  const applied = appliedComment();
  assert.ok(releaseNotes.parseAppliedComment(applied));
  assert.equal(releaseNotes.parseAppliedComment({ ...applied, body: applied.body.replace('state: applied', 'state: applied\nextra: x') }), null);
});

test('extractVersionSection rejects more than one matching heading', () => {
  const twoHeadings = `${changelog()}\n${GENERATED_SECTION}`;
  assert.throws(() => releaseNotes.extractVersionSection(twoHeadings, VERSION), /exactly one/);
});

test('postDraft output round-trips through parseOverrideComment', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dcode-release-rt-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const state = path.join(dir, 'state.json');
  const output = path.join(dir, 'output.md');
  fs.writeFileSync(state, JSON.stringify({ number: 123, version: VERSION, head: HEAD, fingerprint: releaseNotes.sha256(GENERATED_SECTION), heading: HEADING }));
  fs.writeFileSync(output, '### Features\n\n* Add a useful feature.\n');
  const { github, calls } = makeGithub();
  await releaseNotes.postDraft({ github, owner: 'langchain-ai', repo: 'deepagents', stateFile: state, outputFile: output, login: BOT.login, id: BOT.id });
  const parsed = releaseNotes.parseOverrideComment({ user: BOT, body: calls.createComment[0].body });
  assert.ok(parsed, 'override comment should parse');
  assert.equal(parsed.metadata.version, VERSION);
  assert.equal(parsed.metadata['release-heading-hash'], releaseNotes.sha256(HEADING));
});

test('apply uses an exact-parent commit and a non-force branch update', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const stateFile = path.join(workspace.root, 'apply.json');
  const { github, calls } = makeGithub({ comments: [overrideComment()] });
  const state = await releaseNotes.prepareApply({
    github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD,
    changelogFile: workspace.file, stateFile, login: BOT.login, id: BOT.id,
  });
  const committed = await releaseNotes.createApplyCommit({
    github, owner: 'langchain-ai', repo: 'deepagents', stateFile,
    changelogFile: workspace.file, login: BOT.login, id: BOT.id,
  });
  assert.deepEqual(committed, { appliedHead: APPLIED_HEAD, created: true });
  assert.deepEqual(calls.createCommit[0].parents, [HEAD]);
  assert.equal(calls.createTree[0].base_tree, 'tree-base');
  assert.equal(calls.createTree[0].tree[0].path, releaseNotes.CHANGELOG_PATH);
  assert.deepEqual(calls.updateRef[0], {
    owner: 'langchain-ai',
    repo: 'deepagents',
    ref: `heads/${releaseNotes.RELEASE_BRANCH}`,
    sha: APPLIED_HEAD,
    force: false,
  });
  await releaseNotes.publishAppliedState({
    github, owner: 'langchain-ai', repo: 'deepagents', stateFile, appliedHead: APPLIED_HEAD, login: BOT.login, id: BOT.id,
  });
  assert.equal(calls.updatePr.length, 1);
  assert.equal(calls.updatePr[0].body, state.body);
  assert.match(calls.updatePr[0].body, /\* Add a useful feature/);
  assert.equal(calls.createComment.length, 1);
  const parsed = releaseNotes.parseAppliedComment({ body: calls.createComment[0].body });
  assert.ok(parsed, 'applied comment should parse');
  assert.equal(parsed.metadata['override-content-hash'], releaseNotes.sha256(CURATED_SECTION));
  assert.equal(parsed.metadata['applied-head'], APPLIED_HEAD);
});

test('apply refuses a concurrent branch move before updating the ref', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const stateFile = path.join(workspace.root, 'apply.json');
  const run = makeGithub({
    comments: [overrideComment()],
    onGetPr: ({ count, pr }) => {
      if (count === 3) pr.head.sha = 'f'.repeat(40);
    },
  });
  await releaseNotes.prepareApply({
    github: run.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD,
    changelogFile: workspace.file, stateFile, login: BOT.login, id: BOT.id,
  });
  await assert.rejects(
    releaseNotes.createApplyCommit({
      github: run.github, owner: 'langchain-ai', repo: 'deepagents', stateFile,
      changelogFile: workspace.file, login: BOT.login, id: BOT.id,
    }),
    /Release PR changed while apply was preparing/,
  );
  assert.equal(run.calls.updateRef.length, 0);
});

test('an already-applied changelog does not create another commit', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const stateFile = path.join(workspace.root, 'apply.json');
  const run = makeGithub({
    comments: [overrideComment()],
    files: new Map([[HEAD, changelog(CURATED_SECTION)]]),
  });
  const state = await releaseNotes.prepareApply({
    github: run.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD,
    changelogFile: workspace.file, stateFile, login: BOT.login, id: BOT.id,
  });
  assert.equal(state.alreadyApplied, true);
  const result = await releaseNotes.createApplyCommit({
    github: run.github, owner: 'langchain-ai', repo: 'deepagents', stateFile,
    changelogFile: workspace.file, login: BOT.login, id: BOT.id,
  });
  assert.deepEqual(result, { appliedHead: HEAD, created: false });
  assert.equal(run.calls.createCommit.length, 0);
  assert.equal(run.calls.updateRef.length, 0);
});

test('apply rejects a canonically-invisible PR-body edit via the exact hash', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const stateFile = path.join(workspace.root, 'apply.json');
  const run = makeGithub({ comments: [overrideComment()] });
  await releaseNotes.prepareApply({
    github: run.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD,
    changelogFile: workspace.file, stateFile, login: BOT.login, id: BOT.id,
  });
  // Trailing whitespace is erased by canonical()/sha256 but preserved by
  // exactSha256, so only the byte-exact body guard can reject this edit.
  run.setPr(releasePr({
    body: `${releasePr().body}   `,
    head: { ...releasePr().head, sha: APPLIED_HEAD },
  }));
  await assert.rejects(
    releaseNotes.publishAppliedState({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', stateFile, appliedHead: APPLIED_HEAD, login: BOT.login, id: BOT.id }),
    /Release PR changed while apply was preparing/,
  );
  assert.equal(releaseNotes.sha256(`${releasePr().body}   `), releaseNotes.sha256(releasePr().body));
  assert.notEqual(releaseNotes.exactSha256(`${releasePr().body}   `), releaseNotes.exactSha256(releasePr().body));
});

test('postApplyFailure posts once per head and requires the configured bot', async () => {
  const run = makeGithub({ comments: [] });
  await releaseNotes.postApplyFailure({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, head: HEAD, login: BOT.login, id: BOT.id, message: 'boom' });
  assert.equal(run.calls.createComment.length, 1);
  assert.match(run.calls.createComment[0].body, /Applying curated release notes failed/);
  assert.match(run.calls.createComment[0].body, /boom/);
  // Dedup: a second failure for the same head does not add another comment.
  await releaseNotes.postApplyFailure({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, head: HEAD, login: BOT.login, id: BOT.id, message: 'boom again' });
  assert.equal(run.calls.createComment.length, 1);

  const wrong = makeGithub({ authenticated: { login: 'someone-else', id: 7 } });
  await assert.rejects(
    releaseNotes.postApplyFailure({ github: wrong.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, head: HEAD, login: BOT.login, id: BOT.id, message: 'x' }),
    /authenticated as someone-else/,
  );
  assert.equal(wrong.calls.createComment.length, 0);
});

test('postDraftFailure posts non-command failure guidance once per head', async () => {
  const run = makeGithub({ comments: [] });
  await releaseNotes.postDraftFailure({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, head: HEAD, login: BOT.login, id: BOT.id, message: 'boom' });
  assert.equal(run.calls.createComment.length, 1);
  assert.match(run.calls.createComment[0].body, /Automatic release-note drafting failed/);
  assert.match(run.calls.createComment[0].body, /boom/);
  assert.equal(releaseNotes.commandFromComment(run.calls.createComment[0].body), null);
  // Dedup: a second failure for the same head does not add another comment.
  await releaseNotes.postDraftFailure({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, head: HEAD, login: BOT.login, id: BOT.id, message: 'boom again' });
  assert.equal(run.calls.createComment.length, 1);
});

test('manual commands run for maintainers and admins', async () => {
  const context = {
    eventName: 'issue_comment',
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: {
      action: 'created',
      issue: { number: 123, pull_request: {} },
      comment: { body: '@dcode-release-bot apply', user: { login: 'maintainer' } },
    },
  };
  const maintain = makeGithub({ permission: 'maintain' });
  const maintainResult = await releaseNotes.validateTrigger({ github: maintain.github, context, core: makeCore() });
  assert.equal(maintainResult.shouldRun, true);
  assert.equal(maintainResult.command, 'apply');
  assert.equal(maintain.calls.createComment.length, 0);

  // The admin flag grants access even when the permission string is not in the set.
  const admin = makeGithub({ permission: 'read', adminFlag: true });
  assert.equal((await releaseNotes.validateTrigger({ github: admin.github, context, core: makeCore() })).shouldRun, true);
});

test('an explicit command on a non-release PR is explained, not silently ignored', async () => {
  const context = {
    eventName: 'issue_comment',
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: {
      action: 'created',
      issue: { number: 123, pull_request: {} },
      comment: { body: '@dcode-release-bot apply', user: { login: 'maintainer' }, author_association: 'MEMBER' },
    },
  };
  const run = makeGithub({ pr: releasePr({ title: 'feat: something else' }) });
  const result = await releaseNotes.validateTrigger({ github: run.github, context, core: makeCore() });
  assert.equal(result.shouldRun, false);
  assert.equal(run.calls.createComment.length, 1);
  assert.match(run.calls.createComment[0].body, /only applies to the `deepagents-code` release PR/);
});

test('required check fails when applied and override fingerprints differ', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment({ fingerprint: releaseNotes.sha256('different') })] });
  const core = makeCore();
  await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.match(core.failed, /fingerprints differ/);
});

test('required check fails when the PR body preview does not mirror the curated section', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${GENERATED_SECTION}\n_End release notes preview._\n`,
  });
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment()] });
  const core = makeCore();
  await releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.match(core.failed, /does not mirror the curated changelog section/);
});

test('required check fails when the PR changes during the final re-read', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const changed = makeGithub({
    pr,
    comments: [overrideComment(), appliedComment()],
    onGetPr: ({ count, pr: livePr }) => { if (count === 2) livePr.head.sha = 'f'.repeat(40); },
  });
  const core = makeCore();
  await releaseNotes.checkCuratedState({ github: changed.github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core, number: 123, login: BOT.login, id: BOT.id });
  assert.match(core.failed, /changed while the curated-notes check was running/);
});

test('prepareDraft and prepareApply reject a changed release head', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), 'dcode-release-runner-'));
  t.after(() => fs.rmSync(runnerTemp, { recursive: true, force: true }));
  const staleHead = 'c'.repeat(40);
  const draft = makeGithub();
  await assert.rejects(
    releaseNotes.prepareDraft({ github: draft.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: staleHead, runnerTemp }),
    /changed before drafting started/,
  );
  const apply = makeGithub({ comments: [overrideComment()] });
  await assert.rejects(
    releaseNotes.prepareApply({ github: apply.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: staleHead, changelogFile: workspace.file, stateFile: path.join(workspace.root, 'state.json'), login: BOT.login, id: BOT.id }),
    /changed before apply started/,
  );
});

test('required check surfaces an unreadable changelog', async () => {
  const pr = releasePr({
    head: { ...releasePr().head, sha: APPLIED_HEAD },
    body: `Release notes preview\n\n${CURATED_SECTION}\n_End release notes preview._\n`,
  });
  const { github } = makeGithub({ pr, comments: [overrideComment(), appliedComment()], malformedContent: true });
  await assert.rejects(
    releaseNotes.checkCuratedState({ github, context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } }, core: makeCore(), number: 123, login: BOT.login, id: BOT.id }),
    /Could not read/,
  );
});

test('rejects a fork PR whose head branch mimics the release branch', () => {
  // Fork PR: head and base are DIFFERENT repos but the head branch name matches the
  // release branch. The headRepository === baseRepository guard must reject it so a
  // fork can never be treated as the trusted internal release PR.
  const fork = releasePr({
    head: { ref: releaseNotes.RELEASE_BRANCH, sha: HEAD, repo: { full_name: 'attacker/deepagents' } },
    base: { ref: 'main', repo: { full_name: 'langchain-ai/deepagents' } },
  });
  assert.equal(releaseNotes.isReleaseBranchPr(fork), false);
  assert.equal(releaseNotes.isReleasePr(fork), false);
  // The same-repo release PR still passes, so the guard isn't over-broad.
  assert.equal(releaseNotes.isReleaseBranchPr(releasePr()), true);
});

test('rejects a bot impostor in both identity directions', () => {
  // Right id, wrong login: a renamed/cloned account must not impersonate the bot.
  const rightIdWrongLogin = { ...overrideComment({ id: 12 }), user: { login: 'evil-clone', id: BOT.id } };
  assert.equal(releaseNotes.latestOverride([rightIdWrongLogin], BOT.login, BOT.id, VERSION), null);
  // Right login, wrong id: a reused login must not impersonate the bot either.
  const rightLoginWrongId = { ...overrideComment({ id: 13 }), user: { login: BOT.login, id: 999 } };
  assert.equal(releaseNotes.latestOverride([rightLoginWrongId], BOT.login, BOT.id, VERSION), null);
});

test('parseOverrideComment enforces the content-marker boundary', () => {
  const valid = overrideComment();
  assert.ok(releaseNotes.parseOverrideComment(valid));

  // No content markers at all.
  const noMarkers = {
    ...valid,
    body: valid.body.replace(releaseNotes.CONTENT_START, '').replace(releaseNotes.CONTENT_END, ''),
  };
  assert.equal(releaseNotes.parseOverrideComment(noMarkers), null);

  // End marker positioned before the start marker (end <= start).
  const reversed = {
    ...valid,
    body: valid.body
      .replace(releaseNotes.CONTENT_START, '__PLACEHOLDER__')
      .replace(releaseNotes.CONTENT_END, releaseNotes.CONTENT_START)
      .replace('__PLACEHOLDER__', releaseNotes.CONTENT_END),
  };
  assert.equal(releaseNotes.parseOverrideComment(reversed), null);

  // Two version headings inside the content — exercises the fail-closed catch.
  const twoHeadings = {
    ...valid,
    body: valid.body.replace(CURATED_SECTION.trimEnd(), `${CURATED_SECTION.trimEnd()}\n\n${CURATED_SECTION.trimEnd()}`),
  };
  assert.equal(releaseNotes.parseOverrideComment(twoHeadings), null);

  // Text smuggled before the version heading inside the markers — the round-trip
  // guard (canonical(extracted) !== section) must reject content the heading omits.
  const smuggled = {
    ...valid,
    body: valid.body.replace(
      `${releaseNotes.CONTENT_START}\n`,
      `${releaseNotes.CONTENT_START}\nSMUGGLED PREAMBLE\n`,
    ),
  };
  assert.equal(releaseNotes.parseOverrideComment(smuggled), null);
});

test('an ambiguous two-command comment from an insider is explained, not dropped', async () => {
  const context = {
    eventName: 'issue_comment',
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: {
      action: 'created',
      issue: { number: 123, pull_request: {} },
      comment: { body: '@dcode-release-bot draft and then @dcode-release-bot apply', user: { login: 'maintainer' }, author_association: 'MEMBER' },
    },
  };
  const run = makeGithub();
  const result = await releaseNotes.validateTrigger({ github: run.github, context, core: makeCore() });
  assert.equal(result.shouldRun, false);
  assert.equal(run.calls.createComment.length, 1);
  assert.match(run.calls.createComment[0].body, /exactly one/);
});

test('an external comment never amplifies into a bot reply', async () => {
  const context = {
    eventName: 'issue_comment',
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: {
      action: 'created',
      issue: { number: 123, pull_request: {} },
      // A drive-by outsider (association NONE) issuing a valid-looking command.
      comment: { body: '@dcode-release-bot apply', user: { login: 'drive-by' }, author_association: 'NONE' },
    },
  };
  const run = makeGithub({ pr: releasePr({ title: 'feat: unrelated' }) });
  const result = await releaseNotes.validateTrigger({ github: run.github, context, core: makeCore() });
  assert.equal(result.shouldRun, false);
  assert.equal(run.calls.createComment.length, 0);
});

test('re-drafting updates the existing override comment instead of creating a new one', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'dcode-release-redraft-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const state = path.join(dir, 'state.json');
  const output = path.join(dir, 'output.md');
  fs.writeFileSync(state, JSON.stringify({ number: 123, version: VERSION, head: HEAD, fingerprint: releaseNotes.sha256(GENERATED_SECTION), heading: HEADING }));
  fs.writeFileSync(output, '### Features\n\n* Add a useful feature.\n');
  const run = makeGithub({ comments: [overrideComment({ id: 55 })] });
  await releaseNotes.postDraft({ github: run.github, owner: 'langchain-ai', repo: 'deepagents', stateFile: state, outputFile: output, login: BOT.login, id: BOT.id });
  assert.equal(run.calls.updateComment.length, 1);
  assert.equal(run.calls.updateComment[0].comment_id, 55);
  assert.equal(run.calls.createComment.length, 0);
});

test('prepareApply fails when no valid override is present', async t => {
  const workspace = tempWorkspace();
  t.after(() => fs.rmSync(workspace.root, { recursive: true, force: true }));
  const missing = makeGithub({ comments: [] });
  await assert.rejects(
    releaseNotes.prepareApply({ github: missing.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD, changelogFile: workspace.file, stateFile: path.join(workspace.root, 'state.json'), login: BOT.login, id: BOT.id }),
    /No valid bot-authored curated release-note draft exists/,
  );

  // Override present but missing its GitHub revision timestamp.
  const noRevision = makeGithub({ comments: [overrideComment({ updatedAt: '' })] });
  await assert.rejects(
    releaseNotes.prepareApply({ github: noRevision.github, owner: 'langchain-ai', repo: 'deepagents', number: 123, expectedHead: HEAD, changelogFile: workspace.file, stateFile: path.join(workspace.root, 'state.json'), login: BOT.login, id: BOT.id }),
    /missing its GitHub revision/,
  );
});
