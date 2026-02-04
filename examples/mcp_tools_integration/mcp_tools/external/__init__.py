"""External MCP tools for third-party services.

These tools connect to external APIs and services:
- google_search_and_summarize: Google search with web fetching
- weather_forecast: OpenWeatherMap weather forecasts
- sentinel_search: ESA Sentinel satellite imagery search
- arxiv_search: arXiv paper search
- arxiv_download_paper: arXiv paper download
- arxiv_read_paper: Read downloaded arXiv papers
- arxiv_list_papers: List downloaded arXiv papers
"""

from .google_search_and_summarize import google_search_and_summarize
from .weather_forecast import weather_forecast
from .sentinel_search import sentinel_search
from .arxiv_search import arxiv_search
from .arxiv_download_paper import arxiv_download_paper
from .arxiv_read_paper import arxiv_read_paper
from .arxiv_list_papers import arxiv_list_papers

__all__ = [
    "google_search_and_summarize",
    "weather_forecast",
    "sentinel_search",
    "arxiv_search",
    "arxiv_download_paper",
    "arxiv_read_paper",
    "arxiv_list_papers",
]
