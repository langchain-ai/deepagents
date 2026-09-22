const fs = require('node:fs');

function readExistingTopics(outputPath) {
  if (!fs.existsSync(outputPath)) return [];
  const topics = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
  if (!Array.isArray(topics) || topics.some(name =>
    typeof name !== 'string' || !name.startsWith('topic:') || name.length <= 6)) {
    throw new Error('Existing topic manifest must be an array of topic label names');
  }
  return topics;
}

async function syncTopicLabels(github, owner, repo, outputPath) {
  const labels = await github.paginate(github.rest.issues.listLabelsForRepo, {
    owner, repo, per_page: 100,
  });
  const discovered = labels
    .map(label => label.name)
    .filter(name => name.startsWith('topic:') && name.length > 6);
  if (!discovered.length) {
    throw new Error('Repository returned no topic labels; refusing to overwrite the manifest');
  }
  // Labels are created on demand, so absence from the repository does not
  // retire a manifest choice. Removals require an explicit manifest edit.
  const topics = [...new Set([...readExistingTopics(outputPath), ...discovered])].sort();
  fs.writeFileSync(outputPath, `${JSON.stringify(topics, null, 2)}\n`);
  return topics;
}

if (require.main === module) {
  throw new Error('Run this helper through actions/github-script');
}

module.exports = { syncTopicLabels };
