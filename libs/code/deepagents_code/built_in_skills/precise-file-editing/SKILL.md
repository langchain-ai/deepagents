---
name: precise-file-editing
description: "Construct reliable edit_file anchors and recover from 'String not found in file' errors. Use this skill on ANY file-editing task, and especially when: (1) an edit_file call returns 'Error: String not found in file', (2) you are about to build an old_string anchor from your own planning prose or freshly generated comments rather than from a read_file result, or (3) the target file contains em-dashes, en-dashes, arrows, or box-drawing separators. Trigger before retrying a failed edit or dropping to shell heredoc/sed/write_file workarounds."
license: MIT
compatibility: designed for deepagents-code
---

# Precise File Editing

## Overview

`edit_file` matches `old_string` byte-for-byte. The most common failure is an anchor the model *reconstructed* from its own prose instead of copying from the file: the LLM tends to render em-dashes (U+2014), en-dashes (U+2013), arrows (→), or box-drawing separators (U+2500) where the file actually has an ASCII hyphen `-`, an ASCII arrow `->`, or nothing at all. The bytes differ, so the match fails even though the text "looks" identical.

## Best Practices

- **Copy anchors verbatim from the latest `read_file` output** of the target region. Never build `old_string` from planning prose, a summary, or comment text you just generated — those are prone to unicode-confusable substitutions.
- **Prefer ASCII-only anchors.** Choose a stable adjacent line that avoids dashes, arrows, and box-drawing characters. When a file uses em-dashes or `─` separators as section rules, anchor on the plain-ASCII line next to them rather than the decorated one.
- **One failure is a signal, not a retry cue.** The same reconstructed anchor will fail again; escalating to `sed`/heredoc/`write_file` hides the real problem (a byte mismatch) instead of fixing it.

## Recovery: on the FIRST "String not found in file"

1. Do NOT resubmit the same anchor, and do NOT fall back to `python3` heredocs, `sed`, `xxd`, `cat -A`, or `write_file`.
2. `read_file` the exact target lines again to see the real bytes.
3. Then either:
   - **(a)** copy the anchor byte-for-byte from that fresh output, or
   - **(b)** narrow `old_string` to a shorter substring that contains no dash, arrow, or separator characters.
4. Retry `edit_file` with the corrected anchor.

## Worked Example

Task: rename a model in a config section whose header the file prints with an ASCII hyphen.

File actually contains (from `read_file`):

```
# model-config
default_model = "sonnet"
```

Failed attempt — anchor reconstructed from prose, em-dash instead of hyphen:

```
edit_file(old_string='# model—config\ndefault_model = "sonnet"', new_string=...)
-> Error: String not found in file
```

Recovery — re-read, then anchor on the ASCII-only line:

```
read_file(...)          # confirms the header is "# model-config" with a plain hyphen
edit_file(old_string='default_model = "sonnet"', new_string='default_model = "opus"')
-> edits applied
```

The second attempt succeeds because the anchor is copied from the file and avoids the dash entirely.
