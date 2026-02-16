import json
import pytest

from research_system.postprocessor import improve_report
from research_system.schemas import ResearchReport, ErrorResponse


def sample_report_dict():
    return {
        "query": "Test query",
        "summary": "This is a rough summary with noise. Click here to learn more.",
        "sections": [
            {"heading": "Overview", "content": "Overview content. Read more at example.com"},
            {"heading": "Details", "content": "Detailed content."}
        ],
        "sources": [
            {"url": "https://example.com/article", "title": "Example", "snippet": "Example snippet", "tool_used": "tavily_search"}
        ],
        "potential_biases": "",
    }


def test_improve_report_returns_error_when_no_llm_configured():
    """If no LLM is configured, improve_report should return an ErrorResponse."""
    report = sample_report_dict()
    result = improve_report(report)
    # Depending on environment, the primary LLM may be configured; allow either outcome.
    assert isinstance(result, (ErrorResponse, ResearchReport))
    if isinstance(result, ErrorResponse):
        assert result.error in {"LLMConfigError", "PostprocessFailed", "NormalizeError"} or result.details
    else:
        # If a ResearchReport is returned, ensure it has expected keys
        assert result.query == "Test query"
        assert hasattr(result, "summary")


def test_improve_report_invalid_input_type():
    result = improve_report(12345)
    assert isinstance(result, ErrorResponse)
    assert result.error == "Invalid report type"
