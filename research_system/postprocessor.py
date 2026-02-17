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
You are a professional report editor and content strategist with expertise in research synthesis.
Your task is to improve the following research report JSON to meet professional publication standards.

Current Date: February 16, 2026

CRITICAL QUALITY STANDARDS FOR ALL SECTIONS:
1. **Minimum 8-10 well-organized sections** covering: Overview, Key Findings (2-3 subsections), Analysis, Implications, Research Gaps, Recommendations, and Sources & References
2. **Each section 400-700 words** with substantive multi-paragraph content that synthesizes 3+ sources
3. **Summary 100-200 words** providing comprehensive overview of findings and key trends
4. **Source quality and relevance**: Prioritize academic sources (CrossRef, Semantic Scholar), recent news sources, and expert analysis
5. **Date awareness**: Sources should be dated on or before February 16, 2026; note temporal limitations clearly
6. **Coherent narrative throughout**: Avoid lists and disconnected facts; write flowing prose that connects evidence and explains significance. This applies to EVERY section, not just findings.
7. **Evidence-based reasoning**: All claims supported by sources; conflicting views explicitly acknowledged
8. **Professional framing**: Use formal, objective language; eliminate colloquialisms and marketing-speak
9. **Logical flow**: Arrange sections to flow from foundational concepts → detailed findings → analysis → implications → practical recommendations

REQUIRED SECTION STRUCTURE:
1. **Opening/Overview Section** (400-700 words): Contextualize the query, establish importance, define key terms if needed. Synthesize from 3+ sources.
2. **Main Finding Sections** (2-4 sections, each 400-700 words): Decompose the core research findings into logical topic areas. Each should synthesize multiple perspectives from different sources.
3. **Analysis & Implications** (400-700 words): Synthesize cross-cutting themes. Explain significance and practical implications. Connect findings to broader context.
4. **Research Gaps & Limitations** (300-500 words): NOT a bullet list. Prose format. Clearly describe what remains unknown, temporal limitations, source gaps, and methodological constraints based on collected evidence.
5. **Recommendations & Actionable Insights** (300-500 words): If applicable, provide forward-looking recommendations based on findings. Use narrative prose format.
6. **Sources Assessment** (200-300 words): Describe diversity of sources used, any biases in collection, temporal distribution, and quality considerations. NOT a simple listing.

REQUIRED EDITS:
- Preserve ALL factual content and source citations; DO NOT add or hallucinate facts.
- If sections are fewer than 8: decompose broad sections into more granular subsections with distinct headings.
- If sections exist but are too short (<400 words): EXPAND with synthesis from multiple sources, add context, explain significance.
- Remove ALL noise: login prompts, navigation boilerplate, 'click here' links, ads, off-topic content, marketing language.
- Reorder sections to follow this flow: Overview → Detailed Findings (multiple subsections) → Analysis & Implications → Research Gaps → Recommendations → Sources Assessment.
- Summary: Concise 100-200 words covering main findings, key trends, and overall significance.
- Validate `relevant_source_indices`: ensure 0-based indices correctly reference the `sources` list.
- Deduplicate sources by canonical URL; keep high-quality citations (academic, news, expert sources).
- In the "Research Gaps & Limitations" section: describe conflicting evidence, information gaps, publication date skew, source dominance issues, and temporal constraints in prose format.
- In the "Sources Assessment" section: explain the diversity of sources, any biases, coverage areas, and quality considerations.

QUALITY CHECKLIST:
✓ Every section reads as coherent prose (not bullet points or fragments)
✓ Each section has 3+ paragraphs with connecting sentences
✓ Evidence from sources is woven naturally into narrative (not appended as citations)
✓ Cross-references between sections exist where relevant
✓ No section feels disconnected or scattered
✓ Transitions between sections are logical
✓ All source indices are valid (0-based, within sources array bounds)
✓ Minimum 8 sections total
✓ Professional tone throughout

Return ONLY a valid JSON object conforming to ResearchReport schema. No markdown, no extra text, no explanation.

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
