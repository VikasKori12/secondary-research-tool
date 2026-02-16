"""
Postprocessing utilities for improving and polishing generated ResearchReport objects.

This module provides `improve_report()` which calls the primary LLM to
restructure, remove noise, and elevate the tone of a `ResearchReport`.
"""
from __future__ import annotations

import json
import logging
from typing import Union, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from .config import get_primary_llm
from .prompts import report_parser
from .schemas import ResearchReport, ErrorResponse

logger = logging.getLogger(__name__)


def improve_report(report: Union[ResearchReport, Dict[str, Any], str], style: str = "professional") -> Union[ResearchReport, ErrorResponse]:
    """Polish and improve a generated report using the primary LLM.

    Args:
        report: The report to improve (Pydantic object, dict, or JSON string).
        style: Desired tone/level (defaults to 'professional').

    Returns:
        ResearchReport on success or ErrorResponse on failure.
    """
    # Normalize report into a dict
    try:
        if isinstance(report, ResearchReport):
            report_dict = report.model_dump()
        elif isinstance(report, dict):
            report_dict = report
        elif isinstance(report, str):
            report_dict = json.loads(report)
        else:
            return ErrorResponse(error="Invalid report type", details=str(type(report)))
    except Exception as e:
        logger.error("Failed to normalize report input: %s", e)
        return ErrorResponse(error="NormalizeError", details=str(e))

    # Build the improvement prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a professional report editor and content strategist.
Your task is to improve the following research report JSON to make it clearer, better organized, and professional in tone.

Strict Requirements:
- Preserve factual content and sources; do NOT add or hallucinate facts.
- Reorder and merge sections into this canonical flow: Executive Summary -> Key Findings -> Analysis/Discussion -> Implications/Recommendations -> Limitations.
- Remove noisy or irrelevant sentences (e.g., 'click here', 'read more', login prompts, navigation boilerplate).
- Summary: produce ~100-200 words (5-8 sentences), crisp and professional.
- Each `sections` entry should contain at least one paragraph (prefer multi-paragraph analysis where appropriate) and be limited to ~300-700 words.
- Validate and correct `relevant_source_indices` so they reference the `sources` list (0-based indices). If unsure, omit the indices rather than mis-index.
- Deduplicate sources by canonical URL; drop low-quality sources (login walls, redirects, trackers) but preserve high-quality citations.
- Append an explicit `References` sub-section listing sources with stable URLs and short attribution lines if the `sources` list is long.
- If evidence is conflicting or sparse, populate `potential_biases` with a concise explanation.

Return ONLY a JSON object that conforms to the ResearchReport schema. Do not include any extra text.

Existing Report JSON:
{existing_report}

Format Instructions:
{format_instructions}

Final JSON Report:
"""
    ).partial(format_instructions=report_parser.get_format_instructions())

    # Build chain: prompt -> llm -> parser
    try:
        llm = get_primary_llm()
    except Exception as e:
        logger.error("Primary LLM not configured for postprocessing: %s", e)
        return ErrorResponse(error="LLMConfigError", details=str(e))

    chain = prompt_template | llm | report_parser

    prompt_context = {
        "existing_report": json.dumps(report_dict, default=str),
    }

    try:
        improved_report: ResearchReport = chain.invoke(prompt_context)
        logger.info("Postprocessing completed for query=%s", improved_report.query if improved_report and improved_report.query else "<unknown>")
        return improved_report
    except Exception as e:
        logger.exception("Error during report postprocessing: %s", e)
        return ErrorResponse(error="PostprocessFailed", details=str(e))
