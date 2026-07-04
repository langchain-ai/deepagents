"""
Resume Review Agent
--------------------
Extracts text from a resume PDF and uses a Gemini-backed deep agent to
generate an ATS-style review and interview-prep report.

Usage:
    export GOOGLE_API_KEY="your-key-here"
    python app.py --pdf sample_resume.pdf --workdir ./resume_workspace

Requires:
    pip install -r requirements.txt
"""

import argparse
import os

from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from prompts import RESUME_REVIEW_PROMPT

DEFAULT_MODEL_NAME = "gemini-3.5-flash"
DEFAULT_WORKDIR = "./resume_workspace"


# ---------------------------------------------------------------------------
# 1. PDF -> text extraction (runs BEFORE the agent, outside any LLM call)
# ---------------------------------------------------------------------------
def extract_pdf_to_text(pdf_path: str, out_dir: str, out_name: str = "resume.txt") -> str:
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    full_text = "\n\n".join(text_parts).strip()

    if not full_text:
        raise ValueError(
            "No extractable text found in the PDF. It may be a scanned/image-only "
            "resume — OCR it first, then rerun."
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return out_name  # relative path, as the agent will see it under root_dir


# ---------------------------------------------------------------------------
# 2. Wire it together
# ---------------------------------------------------------------------------
def run(pdf_path: str, workdir: str, model_name: str, api_key: str) -> None:
    llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)

    resume_filename = extract_pdf_to_text(pdf_path, workdir)
    print(f"[info] Extracted resume text -> {workdir}/{resume_filename}")

    agent = create_deep_agent(
        model=llm,
        system_prompt=RESUME_REVIEW_PROMPT,
        backend=FilesystemBackend(root_dir=workdir, virtual_mode=True),
    )

    for event in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f'Analyze "{resume_filename}" and generate all requested markdown reports.',
                }
            ]
        },
        stream_mode="updates",
    ):
        for node, value in event.items():
            print(f"\n===== {node} =====")
            if not value or "messages" not in value:
                continue
            for msg in value["messages"]:
                print(type(msg).__name__, "-", getattr(msg, "content", "")[:200])

    reports = ["resume_review.md", "ats_interview_report.md"]
    print("\nGenerated reports:\n")
    for report in reports:
        path = os.path.join(workdir, report)
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ Missing: {report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume review deep agent")
    parser.add_argument("--pdf", required=True, help="Path to the resume PDF")
    parser.add_argument(
        "--workdir",
        default=DEFAULT_WORKDIR,
        help=f"Output directory for extracted text and reports (default: {DEFAULT_WORKDIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name (default: {DEFAULT_MODEL_NAME})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "GOOGLE_API_KEY environment variable is not set. "
            "Export it before running, e.g.:\n"
            "  export GOOGLE_API_KEY='your-key-here'"
        )

    run(pdf_path=args.pdf, workdir=args.workdir, model_name=args.model, api_key=api_key)


if __name__ == "__main__":
    main()
