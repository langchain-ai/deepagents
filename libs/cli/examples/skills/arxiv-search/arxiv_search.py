#!/usr/bin/env python3
"""arXiv Search.

Searches the arXiv preprint repository for research papers.
"""

import argparse


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    query : str
        The search query string.
    max_papers : int
        The maximum number of papers to retrieve (default: 10).

    Returns:
        The formatted search results or an error message.
    """
    try:
        import arxiv
    except ImportError:
        return "Error: arxiv package not installed. Install with: pip install arxiv"

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance
        )
        results = "\n\n".join(
            [f"Title: {paper.title}\nSummary: {paper.summary}" for paper in client.results(search)]
        )
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search arXiv for research papers")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum number of papers to retrieve (default: 10)",
    )

    args = parser.parse_args()

    query_arxiv(args.query, max_papers=args.max_papers)


if __name__ == "__main__":
    main()
