# Resume Review Agent

A minimal example of a [deepagents](https://github.com/langchain-ai/deepagents)
agent that reviews a resume PDF and generates:

1. **`resume_review.md`** — summary, strengths, weaknesses, skill gaps,
   missing ATS keywords, concrete improvements, and rewritten bullet points.
2. **`ats_interview_report.md`** — an estimated ATS score with breakdown,
   plus technical, project, behavioral, HR, and follow-up interview questions.

The PDF is parsed to plain text locally (via `pypdf`) *before* any LLM call,
so the agent only ever sees `resume.txt` — this keeps token usage low and
avoids feeding raw PDF binary data to the model.

## Setup

```bash
pip install -r requirements.txt
```

Set your Google API key as an environment variable (never hardcode it):

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Usage

```bash
python app.py --pdf sample_resume.pdf --workdir ./resume_workspace
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | *(required)* | Path to the resume PDF |
| `--workdir` | `./resume_workspace` | Where `resume.txt` and reports are written |
| `--model` | `gemini-2.0-flash` | Gemini model name to use |

## Output

After a successful run, the workdir will contain:

```
resume_workspace/
├── resume.txt                 # extracted plain text
├── resume_review.md           # qualitative review
└── ats_interview_report.md    # ATS score + interview prep
```

## Notes

- If the PDF is scanned/image-only (no extractable text), the script will
  raise an error asking you to OCR it first.
- `sample_resume.pdf` is included so you can try the example without
  bringing your own file.
- This example uses `langchain_community` / `ddgs` in `requirements.txt`
  for optional web-search-augmented tooling; the base prompt in
  `prompts.py` doesn't require it, so feel free to trim if you don't need it.
