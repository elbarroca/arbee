"""
Evidence Extraction and Validation Tools
Helps agents extract structured evidence from web content and assess quality.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from functools import lru_cache
from statistics import mean, pstdev
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings as app_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SUPPORTED_SUPPORT_VALUES = {"pro", "con", "neutral"}
DEFAULT_EVIDENCE_MODEL = "gpt-4o-mini"
DEFAULT_EVIDENCE_TEMPERATURE = 0.1


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class ExtractedEvidence(BaseModel):
    """Structured evidence extracted from a source."""
    subclaim_id: str
    title: str
    url: str
    published_date: str
    source_type: Literal["primary", "high_quality_secondary", "secondary", "weak"]
    claim_summary: str
    support: Literal["pro", "con", "neutral"]
    verifiability_score: float = Field(ge=0.0, le=1.0)
    independence_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    estimated_LLR: float
    extraction_notes: str


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------
@tool
async def extract_evidence_with_multi_rater_validation(
    search_result: Dict[str, Any],
    subclaim: str,
    market_question: str,
    num_raters: int = 3,
) -> Optional[ExtractedEvidence]:
    """
    Extract structured evidence with multi-rater validation to reduce variance.

    Runs multiple LLM passes at different temperatures and aggregates
    results (majority for support, mean for LLR with clamping).
    """
    try:
        title = search_result.get("title", "N/A")
        url = search_result.get("url", "")
        content = search_result.get("content", search_result.get("snippet", ""))
        published_date = search_result.get("published_date", "")

        if not content:
            return None

        source_type = assess_source_type(url, title)
        verifiability_score = calculate_verifiability(content, source_type)
        independence_score = 0.8
        recency_score = calculate_recency_score(published_date)

        temps = [0.0, 0.3, 0.6][: max(1, int(num_raters))]
        analyses: List[Dict[str, Any]] = []
        for t in temps:
            try:
                analyses.append(
                    await analyze_evidence_with_llm(
                        content=content,
                        title=title,
                        subclaim=subclaim,
                        market_question=market_question,
                        source_type=source_type,
                        verifiability=verifiability_score,
                        published_date=published_date,
                        temperature=t,
                    )
                )
            except Exception as e:
                logger.error(f"multi-rater pass failed (temp={t}): {e}")

        if not analyses:
            return None

        llrs = [a["estimated_LLR"] for a in analyses]
        supports = [a["support"] for a in analyses]
        summaries = [a["summary"] for a in analyses]

        try:
            llr_mean = float(mean(llrs))
            llr_std = float(pstdev(llrs)) if len(llrs) > 1 else 0.0
        except Exception:
            llr_mean, llr_std = 0.0, 0.0

        from collections import Counter

        support = Counter(supports).most_common(1)[0][0]
        claim_summary = summaries[0]
        estimated_LLR = clamp_llr_to_source_range(llr_mean, source_type)

        conf = 1.0 - min(llr_std / (abs(llr_mean) + 0.1), 1.0) if llr_mean != 0 else 0.5
        notes = (
            f"Multi-rater (n={len(analyses)}): LLR={llr_mean:+.2f}±{llr_std:.2f}, "
            f"confidence={conf:.2f}, source={source_type}"
        )

        return ExtractedEvidence(
            subclaim_id=subclaim,
            title=title[:800],
            url=url,
            published_date=published_date or "unknown",
            source_type=source_type,
            claim_summary=claim_summary,
            support=support,
            verifiability_score=verifiability_score,
            independence_score=independence_score,
            recency_score=recency_score,
            estimated_LLR=estimated_LLR,
            extraction_notes=notes,
        )
    except Exception as e:
        logger.error(f"extract_evidence_with_multi_rater_validation failed: {e}")
        return None


@tool
async def extract_evidence_tool(
    search_result: Dict[str, Any],
    subclaim: str,
    market_question: str,
) -> Optional[ExtractedEvidence]:
    """
    Extract structured evidence from a single search result via LLM analysis.
    """
    try:
        title = search_result.get("title", "N/A")
        url = search_result.get("url", "")
        content = search_result.get("content", search_result.get("snippet", ""))
        published_date = search_result.get("published_date", "")

        if not content:
            return None

        source_type = assess_source_type(url, title)
        verifiability_score = calculate_verifiability(content, source_type)
        independence_score = 0.8
        recency_score = calculate_recency_score(published_date)

        analysis = await analyze_evidence_with_llm(
            content=content,
            title=title,
            subclaim=subclaim,
            market_question=market_question,
            source_type=source_type,
            verifiability=verifiability_score,
            published_date=published_date,
        )

        support = analysis["support"]
        estimated_LLR = clamp_llr_to_source_range(analysis["estimated_LLR"], source_type)

        return ExtractedEvidence(
            subclaim_id=subclaim,
            title=title[:800],
            url=url,
            published_date=published_date or "unknown",
            source_type=source_type,
            claim_summary=analysis["summary"],
            support=support,
            verifiability_score=verifiability_score,
            independence_score=independence_score,
            recency_score=recency_score,
            estimated_LLR=estimated_LLR,
            extraction_notes=f"LLM-analyzed from {source_type} source, recency={recency_score:.2f}",
        )
    except Exception as e:
        logger.error(f"extract_evidence_tool failed: {e}")
        return None


@tool
async def verify_source_tool(url: str, source_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify the credibility and reliability of a source.
    """
    try:
        domain = extract_domain(url)
        info = KNOWN_SOURCES.get(
            domain,
            {"credibility_score": 0.5, "source_type": "secondary", "reputation": "unknown", "bias_rating": "unknown"},
        )
        return {"url": url, "domain": domain, **info}
    except Exception as e:
        logger.error(f"verify_source_tool failed: {e}")
        return {"url": url, "credibility_score": 0.5, "source_type": "unknown", "error": str(e)}


# ---------------------------------------------------------------------
# LLM Analysis
# ---------------------------------------------------------------------
@lru_cache(maxsize=4)
def _get_cached_llm(model: str, temperature: float) -> ChatOpenAI:
    """Create or retrieve a cached ChatOpenAI instance."""
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=app_settings.OPENAI_API_KEY)


def _prepare_llm_prompt(
    *,
    content: str,
    title: str,
    subclaim: str,
    market_question: str,
    source_type: str,
    published_date: str,
) -> List[SystemMessage | HumanMessage]:
    """Build LangChain messages for the evidence analysis prompt."""
    system_prompt = (
        "You are an evidence analyst for prediction markets. "
        "Read the provided article excerpt and determine whether it supports the given subclaim. "
        "Respond ONLY with a JSON object containing exactly these keys: 'support', 'summary', 'estimated_LLR'. "
        "Do NOT include any explanation or additional text. "
        "\n"
        "Support classification:\n"
        "- 'pro': Evidence supports a YES outcome (positive indicators, success signals, favorable conditions)\n"
        "- 'con': Evidence supports a NO outcome (negative indicators, challenges, unfavorable conditions)\n"
        "- 'neutral': ONLY use when evidence is truly ambiguous or provides general context without clear direction\n"
        "\n"
        "Estimated_LLR must be a NUMBER within calibrated ranges based on source type:\n"
        "- primary: ±1.0 to ±3.0\n"
        "- high_quality_secondary: ±0.3 to ±1.0\n"
        "- secondary: ±0.1 to ±0.5\n"
        "- weak: ±0.01 to ±0.2\n"
        "\n"
        "Be decisive: if evidence leans one way, mark it as pro or con rather than neutral. "
        "Keep the summary under three sentences and focus on verifiable claims."
    )
    trimmed_content = content.strip()
    if len(trimmed_content) > 2000:
        trimmed_content = f"{trimmed_content[:2000]}…"

    human_prompt = (
        f"Market question: {market_question}\n"
        f"Subclaim: {subclaim}\n"
        f"Source type: {source_type}\n"
        f"Title: {title}\n"
        f"Published date: {published_date or 'unknown'}\n"
        "Extract and summarize relevant evidence. "
        "Return ONLY a JSON object in this exact format:\n"
        "{\n"
        '  "support": "pro" | "con" | "neutral",\n'
        '  "estimated_LLR": 0.XX,\n'
        '  "summary": "Brief summary text"\n'
        "}\n\n"
        f"Content excerpt:\n{trimmed_content}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def _extract_text(content: Any) -> str:
    """Normalize AIMessage content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _strip_json_fences(text: str) -> str:
    """Remove Markdown fences around JSON blocks."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


async def analyze_evidence_with_llm(
    *,
    content: str,
    title: str,
    subclaim: str,
    market_question: str,
    source_type: str,
    verifiability: Optional[float] = None,
    published_date: str = "",
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyze evidence content with an LLM to determine support and LLR.
    Falls back to a heuristic when the LLM is unavailable.
    """
    if not app_settings.OPENAI_API_KEY:
        return _heuristic_evidence_analysis(content=content, subclaim=subclaim, source_type=source_type, verifiability=verifiability)

    model_name = getattr(app_settings, "EVIDENCE_LLM_MODEL", DEFAULT_EVIDENCE_MODEL) or DEFAULT_EVIDENCE_MODEL
    if temperature is None:
        tv = getattr(app_settings, "EVIDENCE_LLM_TEMPERATURE", DEFAULT_EVIDENCE_TEMPERATURE)
        try:
            temperature = float(tv)
        except (TypeError, ValueError):
            temperature = DEFAULT_EVIDENCE_TEMPERATURE

    try:
        llm = _get_cached_llm(model_name, temperature)
        messages = _prepare_llm_prompt(
            content=content,
            title=title,
            subclaim=subclaim,
            market_question=market_question,
            source_type=source_type,
            published_date=published_date,
        )
        response = await llm.ainvoke(messages)
        raw_text = _strip_json_fences(_extract_text(response.content))
        raw_text = re.sub(r"```json\s*", "", raw_text).strip()
        raw_text = re.sub(r"```\s*$", "", raw_text).strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*?\}", raw_text)
            if not match:
                raise
            data = json.loads(match.group(0))

        support_value = str(data.get("support", "neutral")).lower()
        if support_value not in SUPPORTED_SUPPORT_VALUES:
            support_value = "neutral"

        summary_text = data.get("summary")
        if not isinstance(summary_text, str) or not summary_text.strip():
            summary_text = create_claim_summary(content, subclaim)

        llr_raw = data.get("estimated_LLR", 0.0)
        if isinstance(llr_raw, str):
            m = re.match(r"([+-]?\d*\.?\d+)", llr_raw)
            llr_value = float(m.group(1)) if m else 0.0
        elif isinstance(llr_raw, (int, float)):
            llr_value = float(llr_raw)
        else:
            llr_value = 0.0

        llr_value = _align_llr_with_support(llr_value, support_value)
        return {"support": support_value, "estimated_LLR": llr_value, "summary": summary_text.strip()}
    except Exception as exc:
        logger.error(f"LLM analysis failed: {exc}")
        return _heuristic_evidence_analysis(content=content, subclaim=subclaim, source_type=source_type, verifiability=verifiability)


# ---------------------------------------------------------------------
# Heuristics & Helpers
# ---------------------------------------------------------------------
def assess_source_type(url: str, title: str) -> Literal["primary", "high_quality_secondary", "secondary", "weak"]:
    """Assess source type from URL and title."""
    domain = extract_domain(url).lower()

    if any(x in domain for x in ["gov", ".edu", "who.int", "census.gov"]):
        return "primary"

    if any(x in domain for x in ["nytimes.com", "wsj.com", "reuters.com", "bloomberg.com", "apnews.com", "bbc.com", "economist.com", "ft.com"]):
        return "high_quality_secondary"

    if any(x in domain for x in [".com", ".org", "news", "journal", "times"]):
        return "secondary"

    return "weak"


def calculate_verifiability(content: str, source_type: str) -> float:
    """Calculate verifiability score based on content and source type."""
    score = 0.5 + {"primary": 0.4, "high_quality_secondary": 0.3, "secondary": 0.1, "weak": 0.0}.get(source_type, 0.0)
    if re.search(r"\d+%|\d+\.\d+%|\d+,\d+", content):
        score += 0.1
    if "study" in content.lower() or "research" in content.lower():
        score += 0.05
    return min(1.0, score)


def calculate_recency_score(published_date: str) -> float:
    """Calculate recency score from published date."""
    if not published_date or published_date == "unknown":
        return 0.3
    try:
        d = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        days_old = (datetime.now() - d).days
        if days_old <= 7:
            return 1.0
        if days_old <= 30:
            return 0.8
        if days_old <= 90:
            return 0.6
        if days_old <= 365:
            return 0.4
        return 0.2
    except Exception:
        return 0.3


def _heuristic_evidence_analysis(content: str, subclaim: str, source_type: str, verifiability: float) -> Dict[str, Any]:
    """Heuristic analysis fallback when LLM is unavailable."""
    support = determine_support(content, subclaim)
    llr = estimate_llr(source_type, verifiability, support, content)
    summary = create_claim_summary(content, subclaim)
    return {"support": support, "estimated_LLR": llr, "summary": summary}


def determine_support(content: str, subclaim: str) -> Literal["pro", "con", "neutral"]:
    """Heuristic support direction classifier (fallback when LLM unavailable)."""
    c = content.lower()
    if any(w in c for w in ["not", "unlikely", "fails", "denied", "rejected"]):
        return "con"
    if any(w in c for w in ["confirms", "shows", "indicates", "supports", "likely"]):
        return "pro"
    return "neutral"


def estimate_llr(source_type: str, verifiability: float, support: Literal["pro", "con", "neutral"], content: str) -> float:
    """Estimate LLR using calibrated ranges, verifiability, and content length."""
    ranges = {"primary": (1.0, 3.0), "high_quality_secondary": (0.3, 1.0), "secondary": (0.1, 0.5), "weak": (0.01, 0.2)}
    mn, mx = ranges.get(source_type, (0.1, 0.3))
    base = mn + (mx - mn) * verifiability
    if len(content) < 100:
        base *= 0.7
    if support == "pro":
        return base
    if support == "con":
        return -base
    return 0.0


def create_claim_summary(content: str, subclaim: str) -> str:
    """Create concise claim summary from content."""
    if len(content) <= 500:
        return content
    sentences = content.split(". ")
    if sentences and len(sentences[0]) <= 500:
        return sentences[0] + "."
    return content[:500] + "..."


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except Exception:
        return url.split("/")[2] if "/" in url else url


def clamp_llr_to_source_range(llr: float, source_type: str) -> float:
    """Clamp LLR to calibrated ranges based on source quality."""
    ranges = {"primary": (-3.0, 3.0), "high_quality_secondary": (-1.0, 1.0), "secondary": (-0.5, 0.5), "weak": (-0.2, 0.2)}
    mn, mx = ranges.get(source_type, (-0.5, 0.5))
    return max(min(llr, mx), mn)


def _align_llr_with_support(llr: float, support: str) -> float:
    """Ensure LLR sign/magnitude is consistent with support classification."""
    if support == "neutral":
        return 0.0
    mag = abs(llr)
    if mag < 1e-6:
        mag = 0.0
    return mag if support == "pro" else -mag


# ---------------------------------------------------------------------
# Source Registry (simple, extend as needed)
# ---------------------------------------------------------------------
KNOWN_SOURCES = {
    # High-quality news sources
    "nytimes.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center-left",
    },
    "wsj.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center-right",
    },
    "reuters.com": {
        "credibility_score": 0.95,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center",
    },
    "bloomberg.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center",
    },
    "apnews.com": {
        "credibility_score": 0.95,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center",
    },
    "washingtonpost.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center-left",
    },
    "theguardian.com": {
        "credibility_score": 0.85,
        "source_type": "high_quality_secondary",
        "reputation": "credible",
        "bias_rating": "left",
    },
    "bbc.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center",
    },
    "economist.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "highly_credible",
        "bias_rating": "center",
    },
    "npr.org": {
        "credibility_score": 0.85,
        "source_type": "high_quality_secondary",
        "reputation": "credible",
        "bias_rating": "center-left",
    },
    "propublica.org": {
        "credibility_score": 0.95,
        "source_type": "high_quality_secondary",
        "reputation": "investigative_journalism",
        "bias_rating": "center-left",
    },
    # Fact-Checking & Data-Driven Sites
    "fivethirtyeight.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "data-driven",
        "bias_rating": "center",
    },
    "politifact.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "fact_checker",
        "bias_rating": "center-left",
    },
    "snopes.com": {
        "credibility_score": 0.85,
        "source_type": "high_quality_secondary",
        "reputation": "fact_checker",
        "bias_rating": "center",
    },
    "factcheck.org": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "fact_checker",
        "bias_rating": "center",
    },
    # Technology & Business News
    "techcrunch.com": {
        "credibility_score": 0.8,
        "source_type": "high_quality_secondary",
        "reputation": "tech_industry_news",
        "bias_rating": "center",
    },
    "theverge.com": {
        "credibility_score": 0.85,
        "source_type": "high_quality_secondary",
        "reputation": "tech_journalism",
        "bias_rating": "center-left",
    },
    "arstechnica.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "in_depth_tech",
        "bias_rating": "center",
    },
    # Sports Journalism
    "espn.com": {
        "credibility_score": 0.8,
        "source_type": "high_quality_secondary",
        "reputation": "sports_news",
        "bias_rating": "center",
    },
    "theathletic.com": {
        "credibility_score": 0.9,
        "source_type": "high_quality_secondary",
        "reputation": "in_depth_sports",
        "bias_rating": "center",
    },
    # Arts & Entertainment
    "variety.com": {
        "credibility_score": 0.85,
        "source_type": "high_quality_secondary",
        "reputation": "entertainment_trade",
        "bias_rating": "center",
    },
    "hollywoodreporter.com": {
        "credibility_score": 0.85,
        "source_type": "high_quality_secondary",
        "reputation": "entertainment_trade",
        "bias_rating": "center",
    },
    # Other News Sources (Partisan or Mixed Quality)
    "foxnews.com": {
        "credibility_score": 0.5,
        "source_type": "secondary",
        "reputation": "mixed",
        "bias_rating": "right",
    },
    "huffpost.com": {
        "credibility_score": 0.5,
        "source_type": "secondary",
        "reputation": "mixed",
        "bias_rating": "left",
    },
    "breitbart.com": {
        "credibility_score": 0.2,
        "source_type": "weak",
        "reputation": "hyper_partisan",
        "bias_rating": "far-right",
    },
    "dailymail.co.uk": {
        "credibility_score": 0.3,
        "source_type": "weak",
        "reputation": "tabloid",
        "bias_rating": "right",
    },
    # Primary sources (scientific, government)
    "arxiv.org": {
        "credibility_score": 0.7,
        "source_type": "primary",
        "reputation": "academic_preprint",
        "bias_rating": "neutral",
    },
    "nature.com": {
        "credibility_score": 0.95,
        "source_type": "primary",
        "reputation": "top_tier_journal",
        "bias_rating": "neutral",
    },
    "science.org": {
        "credibility_score": 0.95,
        "source_type": "primary",
        "reputation": "top_tier_journal",
        "bias_rating": "neutral",
    },
    "thelancet.com": {
        "credibility_score": 0.95,
        "source_type": "primary",
        "reputation": "top_tier_medical_journal",
        "bias_rating": "neutral",
    },
    "nejm.org": {
        "credibility_score": 0.95,
        "source_type": "primary",
        "reputation": "top_tier_medical_journal",
        "bias_rating": "neutral",
    },
    ".gov": {  # Generic handler for government sites
        "credibility_score": 0.9,
        "source_type": "primary",
        "reputation": "government_source",
        "bias_rating": "neutral",
    },
    "cdc.gov": {
        "credibility_score": 0.95,
        "source_type": "primary",
        "reputation": "government_health_agency",
        "bias_rating": "neutral",
    },
    "nasa.gov": {
        "credibility_score": 0.95,
        "source_type": "primary",
        "reputation": "government_space_agency",
        "bias_rating": "neutral",
    },
    "bls.gov": {
        "credibility_score": 0.95,
        "source_type": "primary",
        "reputation": "government_statistics",
        "bias_rating": "neutral",
    },
    "who.int": {
        "credibility_score": 0.9,
        "source_type": "primary",
        "reputation": "international_health_organization",
        "bias_rating": "neutral",
    },
    "un.org": {
        "credibility_score": 0.85,
        "source_type": "primary",
        "reputation": "international_organization",
        "bias_rating": "neutral",
    },
    # Reference & Aggregators
    "wikipedia.org": {
        "credibility_score": 0.4,
        "source_type": "secondary",
        "reputation": "encyclopedia_aggregator",
        "bias_rating": "neutral",
    },
    "rottentomatoes.com": {
        "credibility_score": 0.5,
        "source_type": "secondary",
        "reputation": "review_aggregator",
        "bias_rating": "neutral",
    },
}


# ---------------------------------------------------------------------
# Optional: simple integrity probe for a single source snippet
# ---------------------------------------------------------------------
def integrity_report(content: str, url: str, published_date: str) -> Dict[str, Any]:
    """Return compact integrity metrics for a content snippet and source."""
    st = assess_source_type(url, "")
    return {
        "source_type": st,
        "verifiability": calculate_verifiability(content, st),
        "recency": calculate_recency_score(published_date),
        "length": len(content or ""),
        "domain": extract_domain(url),
        "llr_bounds": {
            "primary": (-3.0, 3.0),
            "high_quality_secondary": (-1.0, 1.0),
            "secondary": (-0.5, 0.5),
            "weak": (-0.2, 0.2),
        }.get(st, (-0.5, 0.5)),
    }