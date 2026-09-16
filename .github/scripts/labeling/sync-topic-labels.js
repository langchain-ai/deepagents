const fs = require('node:fs');

async function syncTopicLabels(github, owner, repo, outputPath) {
  const labels = await github.paginate(github.rest.issues.listLabelsForRepo, {
    owner, repo, per_page: 100,
  });
  const topics = labels
    .map(label => label.name)
    .filter(name => name.startsWith('topic:'))
    .sort();
  fs.writeFileSync(outputPath, `${JSON.stringify(topics, null, 2)}\n`);
  return topics;
}

if (require.main === module) {
  throw new Error('Run this helper through actions/github-script');
}

module.exports = { syncTopicLabels };
