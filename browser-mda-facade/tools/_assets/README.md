`playwright_facade.js` is vendored verbatim from browserbase/stagehand
(`packages/integrations/deepagents/src/stagehand_deepagents/_assets/`), MIT
licensed, Copyright (c) 2024 Browserbase Inc.

It is the Playwright→Stagehand translator: `createPlaywrightCompatRuntime`
turns the worker-local batch object model into Playwright-shaped `page`,
`context` and `browser`. The upstream `stagehand-deepagents` package that ships
it is not on PyPI, so the file is copied rather than depended on. Re-copy it
when upgrading Stagehand rather than editing it here.
