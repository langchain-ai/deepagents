Fixes #

<!-- Update the relationship and number above, or remove the line when none applies. Only `Closes`, `Fixes`, and `Resolves` auto-close the referenced GitHub issue on merge. -->

<!-- For a net new feature or behavior-changing bugfix, replace this comment with a high-level, plain-English summary of the user-visible change. Do not add a heading or label. Remove this comment for chores, refactors, or test-only changes. -->

---

<!-- Explain the motivation and why this solution is the right one. Do not add a `# Summary` or `## Release note` heading. -->

Read the full contributing guidelines: https://docs.langchain.com/oss/python/contributing/overview

> **All contributions must be in English.** See the [language policy](https://docs.langchain.com/oss/python/contributing/overview#language-policy).

If you paste a large clearly AI generated description here your PR may be IGNORED or CLOSED!

Thank you for contributing to Deep Agents! Follow these steps to have your pull request considered as ready for review.

1. PR title: Should follow the format: TYPE(SCOPE): DESCRIPTION

  - Examples:
    - fix(sdk): resolve flag parsing error
    - feat(cli): add multi-tenant support
    - test(acp): update API usage tests
  - Do not include Linear issue-closing markers such as `[closes DCD-52]` in the title. Put issue references and closing metadata in the PR description instead.
  - Allowed TYPE and SCOPE values: https://github.com/langchain-ai/deepagents/blob/main/.github/workflows/pr_lint.yml

2. PR description:

  - Keep an optional issue or PR relationship and, when required, the user-facing summary above the `---`; put the rest of the body below it.
  - For net new features or behavior-changing bugfixes, write a high-level, plain-English summary of the user-visible change without a heading or label.
  - If this PR addresses a specific issue, use `Fixes #ISSUE_NUMBER`, `Closes #ISSUE_NUMBER`, or `Resolves #ISSUE_NUMBER` to automatically close it when the PR is merged.
  - If there are any breaking changes, please clearly describe them.
  - If this PR depends on another PR being merged first, please include `Depends on #PR_NUMBER` in the description.

<!--
Do not add a dedicated "Test plan" or "Testing" section unless this PR is large or the changes are highly consequential. When one is warranted, keep it collapsed:

<details>
<summary>Test plan</summary>

- Describe the verification performed.

</details>
-->

3. Run `make format`, `make lint` and `make test` from the root of the package(s) you've modified.

  - We will not consider a PR unless these three are passing in CI.

Additional guidelines:

  - We ask that if you use generative AI for your contribution, you include a disclaimer.
  - PRs should not touch more than one package unless absolutely necessary.
  - Do not update the `uv.lock` files or add dependencies to `pyproject.toml` files (even optional ones) unless you have explicit permission to do so by a maintainer.

## Social handles (optional)
<!-- If you'd like a shoutout on release, add your socials below -->
Twitter: @
LinkedIn: https://linkedin.com/in/
