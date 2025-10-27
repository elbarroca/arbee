"""
Evidence Extraction and Validation Tools
Helps agents extract structured evidence from web content and assess quality
"""
import json
import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings as app_settings

logger = logging.getLogger(__name__)

SUPPORTED_SUPPORT_VALUES = {"pro", "con", "neutral"}

DEFAULT_EVIDENCE_MODEL = "gpt-4o-mini"
DEFAULT_EVIDENCE_TEMPERATURE = 0.1

STOPWORDS: Set[str] = {
    "the", "and", "with", "from", "that", "this", "will", "have", "has", "into", "their", "about",
    "under", "over", "after", "before", "into", "could", "should", "would", "there", "which",
    "while", "where", "whose", "been", "being", "during", "around", "among", "because", "since",
    "toward", "towards", "against", "between", "within", "without", "across", "through", "into",
    "onto", "upon", "does", "did", "doing", "done"
}

GENERIC_ENTITY_STOPWORDS: Set[str] = {
    "will", "what", "which", "when", "where", "why", "how", "who", "weather", "climate",
    "global", "future", "average", "running", "runner", "runners", "minutes"
}

GENERIC_MARKET_TERMS: Set[str] = {
    "will", "would", "could", "should", "might", "market", "question", "outcome", "probability",
    "forecast", "forecasting", "minutes", "minute", "second", "seconds", "hour", "hours", "under",
    "over", "times", "time", "race", "running", "runner", "fitness", "training", "marathon",
    "climate", "weather", "global", "health", "issues", "issue", "trend", "trends", "average"
}

SUBJECT_ALIAS_MAP: Dict[str, Set[str]] = {
    "diplo": {
        "thomas wesley pentz",
        "thomas wesley",
        "wesley pentz",
        "thomas pentz",
        "pentz",
        "dj diplo"
    }
}


class ExtractedEvidence(BaseModel):
    """Structured evidence extracted from source"""
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


@tool
async def extract_evidence_with_multi_rater_validation(
    search_result: Dict[str, Any],
    subclaim: str,
    market_question: str,
    num_raters: int = 3
) -> Optional[ExtractedEvidence]:
    """
    Extract structured evidence with multi-rater validation to reduce variance.

    This tool extracts evidence multiple times with different LLM temperatures
    and averages the LLR estimates to reduce subjective judgment variance.

    Args:
        search_result: Dict with 'title', 'url', 'content/snippet', 'published_date'
        subclaim: The specific subclaim this evidence relates to
        market_question: The main market question for context
        num_raters: Number of independent extractions (default 3)

    Returns:
        ExtractedEvidence with averaged LLR and confidence score

    Example:
        >>> evidence = await extract_evidence_with_multi_rater_validation(result, subclaim, question)
        >>> print(f"LLR: {evidence.estimated_LLR:.2f} ± {evidence.extraction_notes}")
    """
    try:
        logger.info(f"🔬 Multi-rater extraction from: {search_result.get('title', 'Unknown')[:60]}")

        # Extract basic fields
        title = search_result.get('title', 'N/A')
        url = search_result.get('url', '')
        content = search_result.get('content', search_result.get('snippet', ''))
        published_date = search_result.get('published_date', '')

        # Skip only if completely empty
        if not content:
            logger.debug("No content found, skipping")
            return None

        # Let LLM judge relevance - removed strict keyword filtering

        # Assess source type (same for all raters)
        source_type = assess_source_type(url, title)
        verifiability_score = calculate_verifiability(content, source_type)
        independence_score = 0.8
        recency_score = calculate_recency_score(published_date)

        # Run multiple extractions with different temperatures
        temperatures = [0.0, 0.3, 0.6][:num_raters]
        llm_analyses = []

        for temp in temperatures:
            try:
                analysis = await analyze_evidence_with_llm(
                    content=content,
                    title=title,
                    subclaim=subclaim,
                    market_question=market_question,
                    source_type=source_type,
                    verifiability=verifiability_score,
                    published_date=published_date,
                    temperature=temp
                )
                llm_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Extraction failed at temp={temp}: {e}")

        if not llm_analyses:
            logger.error("All extractions failed")
            return None

        # Aggregate results
        import numpy as np

        llrs = [a['estimated_LLR'] for a in llm_analyses]
        supports = [a['support'] for a in llm_analyses]
        summaries = [a['summary'] for a in llm_analyses]

        # Average LLR
        mean_llr = float(np.mean(llrs))
        std_llr = float(np.std(llrs)) if len(llrs) > 1 else 0.0

        # Majority vote on support direction
        from collections import Counter
        support_counts = Counter(supports)
        support = support_counts.most_common(1)[0][0]

        # Use first summary (or could combine)
        claim_summary = summaries[0]

        # Clamp LLR to source range
        estimated_LLR = clamp_llr_to_source_range(mean_llr, source_type)

        # Build extraction notes with confidence info
        llr_confidence = 1.0 - min(std_llr / (abs(mean_llr) + 0.1), 1.0) if mean_llr != 0 else 0.5
        extraction_notes = (
            f"Multi-rater validated (n={len(llm_analyses)}): "
            f"LLR={mean_llr:+.2f}±{std_llr:.2f}, "
            f"confidence={llr_confidence:.2f}, "
            f"source={source_type}"
        )

        evidence = ExtractedEvidence(
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
            extraction_notes=extraction_notes
        )

        logger.info(
            f"✅ Multi-rater evidence: LLR={estimated_LLR:+.2f}±{std_llr:.2f}, "
            f"support={support}, confidence={llr_confidence:.2f}"
        )

        return evidence

    except Exception as e:
        logger.error(f"❌ Multi-rater extraction failed: {e}")
        return None


@tool
async def extract_evidence_tool(
    search_result: Dict[str, Any],
    subclaim: str,
    market_question: str
) -> Optional[ExtractedEvidence]:
    """
    Extract structured evidence from a single search result using LLM analysis.

    Use this tool to parse web search results into structured evidence items
    with proper scoring and LLR estimation based on content analysis.

    Args:
        search_result: Dict with 'title', 'url', 'content/snippet', 'published_date'
        subclaim: The specific subclaim this evidence relates to
        market_question: The main market question for context

    Returns:
        ExtractedEvidence object or None if not relevant

    Example:
        >>> result = {"title": "NYT: Trump leads in Arizona poll",
        ...           "url": "https://nyt.com/...",
        ...           "content": "New poll shows...",
        ...           "published_date": "2024-10-15"}
        >>> evidence = await extract_evidence_tool(result, "Trump will win Arizona", "Will Trump win 2024?")
    """
    try:
        logger.info(f"🔬 Extracting evidence from: {search_result.get('title', 'Unknown')[:60]}")

        # Extract basic fields
        title = search_result.get('title', 'N/A')
        url = search_result.get('url', '')
        content = search_result.get('content', search_result.get('snippet', ''))
        published_date = search_result.get('published_date', '')

        # Skip only if completely empty
        if not content:
            logger.debug("No content found, skipping")
            return None

        # Let LLM judge relevance - removed strict keyword filtering
        # All search results passed to LLM for relevance analysis

        # Assess source type from URL/title
        source_type = assess_source_type(url, title)

        # Score verifiability (based on source type and content specificity)
        verifiability_score = calculate_verifiability(content, source_type)

        # Score independence (basic heuristic - can be improved)
        independence_score = 0.8  # Default, should check for common sources

        # Score recency
        recency_score = calculate_recency_score(published_date)

        # Use LLM to analyze content for support direction and strength
        llm_analysis = await analyze_evidence_with_llm(
            content=content,
            title=title,
            subclaim=subclaim,
            market_question=market_question,
            source_type=source_type,
            verifiability=verifiability_score,
            published_date=published_date
        )

        support = llm_analysis['support']
        estimated_LLR = llm_analysis['estimated_LLR']
        claim_summary = llm_analysis['summary']

        # Apply quality adjustments to LLR
        # Clamp extreme LLRs to calibrated ranges
        estimated_LLR = clamp_llr_to_source_range(estimated_LLR, source_type)

        evidence = ExtractedEvidence(
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
            extraction_notes=f"LLM-analyzed from {source_type} source, recency={recency_score:.2f}"
        )

        logger.info(f"✅ Evidence extracted: LLR={estimated_LLR:+.2f}, support={support}")

        return evidence

    except Exception as e:
        logger.error(f"❌ Evidence extraction failed: {e}")
        return None


def _heuristic_evidence_analysis(
    *,
    content: str,
    subclaim: str,
    source_type: str,
    verifiability: Optional[float] = None
) -> Dict[str, Any]:
    """Fallback analysis when LLM is unavailable or fails."""
    verifiability_score = (
        verifiability
        if verifiability is not None
        else calculate_verifiability(content, source_type)
    )
    support_direction = determine_support(content, subclaim)
    estimated_llr = estimate_llr(source_type, verifiability_score, support_direction, content)
    summary = create_claim_summary(content, subclaim)
    return {
        'support': support_direction,
        'estimated_LLR': estimated_llr,
        'summary': summary
    }


@lru_cache(maxsize=4)
def _get_cached_llm(model: str, temperature: float) -> ChatOpenAI:
    """Create or retrieve a cached ChatOpenAI instance."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=app_settings.OPENAI_API_KEY
    )


def _prepare_llm_prompt(
    *,
    content: str,
    title: str,
    subclaim: str,
    market_question: str,
    source_type: str,
    published_date: str
) -> List[SystemMessage | HumanMessage]:
    """Build LangChain messages for the evidence analysis prompt."""
    system_prompt = (
        "You are an evidence analyst for prediction markets. "
        "Read the provided article excerpt and determine whether it supports the given subclaim. "
        "Respond ONLY with a JSON object containing exactly these keys: 'support', 'summary', 'estimated_LLR'. "
        "Do NOT include any explanation or additional text. "
        "Support must be one of: 'pro', 'con', 'neutral'. "
        "Estimated_LLR must be a NUMBER within calibrated ranges based on source type:\n"
        "- primary: ±1.0 to ±3.0\n"
        "- high_quality_secondary: ±0.3 to ±1.0\n"
        "- secondary: ±0.1 to ±0.5\n"
        "- weak: ±0.01 to ±0.2\n"
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
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]


def _extract_text(content: Any) -> str:
    """Normalize AIMessage content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)


def _strip_json_fences(text: str) -> str:
    """Remove Markdown fences around JSON blocks."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop opening fence
        lines = lines[1:]
        # Drop closing fence if present
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


async def analyze_evidence_with_llm(
    *,
    content: str,
    title: str,
    subclaim: str,
    market_question: str,
    source_type: str,
    verifiability: Optional[float] = None,
    published_date: str = "",
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze evidence content with an LLM to determine support direction and LLR.

    Falls back to heuristic analysis if the OpenAI key is unavailable or the LLM call fails.

    Args:
        temperature: Optional temperature override for LLM (default from settings)
    """
    if not app_settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY missing; falling back to heuristic evidence analysis")
        return _heuristic_evidence_analysis(
            content=content,
            subclaim=subclaim,
            source_type=source_type,
            verifiability=verifiability
        )

    model_name = getattr(app_settings, "EVIDENCE_LLM_MODEL", DEFAULT_EVIDENCE_MODEL) or DEFAULT_EVIDENCE_MODEL

    if temperature is None:
        temperature_value = getattr(app_settings, "EVIDENCE_LLM_TEMPERATURE", DEFAULT_EVIDENCE_TEMPERATURE)
        try:
            temperature = float(temperature_value)
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
            published_date=published_date
        )
        response = await llm.ainvoke(messages)
        raw_text = _strip_json_fences(_extract_text(response.content))

        # Clean up common LLM response artifacts
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*$', '', raw_text)
        raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            json_match = re.search(r'\{[\s\S]*?\}', raw_text)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from response: {raw_text[:200]}")
                    raise
            else:
                logger.error(f"No JSON found in response: {raw_text[:200]}")
                raise

        support_value = str(data.get('support', 'neutral')).lower()
        if support_value not in SUPPORTED_SUPPORT_VALUES:
            support_value = 'neutral'

        summary_text = data.get('summary')
        if not isinstance(summary_text, str) or not summary_text.strip():
            summary_text = create_claim_summary(content, subclaim)

        estimated_llr_raw = data.get('estimated_LLR', 0.0)

        # Handle various malformed LLR formats from LLM
        if isinstance(estimated_llr_raw, str):
            # Extract numeric part from strings like "0.5 independence_"
            numeric_match = re.match(r'([+-]?\d*\.?\d+)', estimated_llr_raw)
            if numeric_match:
                estimated_llr_value = float(numeric_match.group(1))
            else:
                estimated_llr_value = 0.0
        elif isinstance(estimated_llr_raw, (int, float)):
            estimated_llr_value = float(estimated_llr_raw)
        else:
            estimated_llr_value = 0.0

        estimated_llr_value = _align_llr_with_support(estimated_llr_value, support_value)
        return {
            'support': support_value,
            'estimated_LLR': estimated_llr_value,
            'summary': summary_text.strip()
        }

    except Exception as exc:
        logger.error(f"LLM evidence analysis failed: {exc}")
        return _heuristic_evidence_analysis(
            content=content,
            subclaim=subclaim,
            source_type=source_type,
            verifiability=verifiability
        )


@tool
async def verify_source_tool(
    url: str,
    source_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify the credibility and reliability of a source.

    Use this to assess whether a source is trustworthy before using its
    evidence in analysis.

    Args:
        url: Source URL
        source_name: Optional source name/domain

    Returns:
        Dict with credibility_score, source_type, reputation, bias_rating

    Example:
        >>> verification = await verify_source_tool("https://www.nytimes.com/article")
        >>> print(verification['credibility_score'])  # 0.9
    """
    try:
        logger.info(f"🔍 Verifying source: {url[:60]}")

        # Extract domain
        domain = extract_domain(url)

        # Check against known sources database
        source_info = KNOWN_SOURCES.get(domain, {
            'credibility_score': 0.5,
            'source_type': 'secondary',
            'reputation': 'unknown',
            'bias_rating': 'unknown'
        })

        logger.info(f"✅ Source verified: {domain} - credibility={source_info['credibility_score']}")

        return {
            'url': url,
            'domain': domain,
            **source_info
        }

    except Exception as e:
        logger.error(f"❌ Source verification failed: {e}")
        return {
            'url': url,
            'credibility_score': 0.5,
            'source_type': 'unknown',
            'error': str(e)
        }


# Helper functions

def assess_source_type(url: str, title: str) -> Literal["primary", "high_quality_secondary", "secondary", "weak"]:
    """Assess source type from URL and title"""
    domain = extract_domain(url).lower()

    # Primary sources
    if any(x in domain for x in ['gov', '.edu', 'who.int', 'census.gov']):
        return "primary"

    # High-quality secondary
    if any(x in domain for x in [
        'nytimes.com', 'wsj.com', 'reuters.com', 'bloomberg.com',
        'apnews.com', 'bbc.com', 'economist.com', 'ft.com'
    ]):
        return "high_quality_secondary"

    # Secondary
    if any(x in domain for x in [
        '.com', '.org', 'news', 'journal', 'times'
    ]):
        return "secondary"

    # Weak
    return "weak"


def calculate_verifiability(content: str, source_type: str) -> float:
    """Calculate verifiability score based on content and source"""
    score = 0.5  # Base score

    # Bonus for source type
    type_scores = {
        "primary": 0.4,
        "high_quality_secondary": 0.3,
        "secondary": 0.1,
        "weak": 0.0
    }
    score += type_scores.get(source_type, 0.0)

    # Bonus for specific numbers/dates
    if re.search(r'\d+%|\d+\.\d+%|\d+,\d+', content):
        score += 0.1

    # Bonus for citations
    if 'study' in content.lower() or 'research' in content.lower():
        score += 0.05

    return min(1.0, score)


def calculate_recency_score(published_date: str) -> float:
    """Calculate recency score from published date"""
    if not published_date or published_date == 'unknown':
        return 0.3  # Default for unknown dates

    try:
        # Parse date (handle various formats)
        date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
        days_old = (datetime.now() - date).days

        if days_old <= 7:
            return 1.0
        elif days_old <= 30:
            return 0.8
        elif days_old <= 90:
            return 0.6
        elif days_old <= 365:
            return 0.4
        else:
            return 0.2

    except:
        return 0.3


def determine_support(content: str, subclaim: str) -> Literal["pro", "con", "neutral"]:
    """Determine if content supports or contradicts subclaim"""
    # Simplified heuristic - in production, use LLM
    content_lower = content.lower()

    # Look for negations
    if any(word in content_lower for word in ['not', 'unlikely', 'fails', 'denied', 'rejected']):
        return "con"

    # Look for affirmations
    if any(word in content_lower for word in ['confirms', 'shows', 'indicates', 'supports', 'likely']):
        return "pro"

    return "neutral"


def estimate_llr(
    source_type: str,
    verifiability: float,
    support: Literal["pro", "con", "neutral"],
    content: str
) -> float:
    """Estimate Log-Likelihood Ratio for evidence"""
    # Base LLR from source type (calibrated ranges from CLAUDE.md)
    base_llr_ranges = {
        "primary": (1.0, 3.0),
        "high_quality_secondary": (0.3, 1.0),
        "secondary": (0.1, 0.5),
        "weak": (0.01, 0.2)
    }

    min_llr, max_llr = base_llr_ranges.get(source_type, (0.1, 0.3))

    # Use verifiability to interpolate within range
    base_llr = min_llr + (max_llr - min_llr) * verifiability

    # Check content strength
    if len(content) < 100:
        base_llr *= 0.7  # Penalty for short content

    # Apply direction
    if support == "pro":
        return base_llr
    elif support == "con":
        return -base_llr
    else:  # neutral
        return 0.0


def create_claim_summary(content: str, subclaim: str) -> str:
    """Create concise claim summary from content"""
    # Simplified - just truncate intelligently
    if len(content) <= 500:
        return content

    # Find first sentence or first 500 chars
    sentences = content.split('. ')
    if sentences and len(sentences[0]) <= 500:
        return sentences[0] + '.'

    return content[:500] + '...'


def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return url.split('/')[2] if '/' in url else url


def clamp_llr_to_source_range(llr: float, source_type: str) -> float:
    """Clamp LLR to calibrated ranges based on source quality."""
    ranges = {
        "primary": (-3.0, 3.0),
        "high_quality_secondary": (-1.0, 1.0),
        "secondary": (-0.5, 0.5),
        "weak": (-0.2, 0.2)
    }
    min_llr, max_llr = ranges.get(source_type, (-0.5, 0.5))
    return max(min(llr, max_llr), min_llr)


def _align_llr_with_support(llr: float, support: str) -> float:
    """Ensure LLR sign/magnitude is consistent with support classification."""
    if support == "neutral":
        return 0.0
    magnitude = abs(llr)
    if magnitude < 1e-6:
        magnitude = 0.0
    if support == "pro":
        return magnitude
    if support == "con":
        return -magnitude
    return 0.0


def extract_entities(text: str) -> Set[str]:
    """Extract capitalized entities from text for relevance checks."""
    return {
        match.lower()
        for match in re.findall(r"\b[A-Z][a-zA-Z']+\b", text or "")
        if match.lower() not in STOPWORDS
    }


def extract_keywords(text: str) -> Set[str]:
    """Extract descriptive keywords from text for relevance checks."""
    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z]{4,}", text or "")
        if token.lower() not in STOPWORDS
    }


def _normalize_for_matching(text: str) -> str:
    """Lowercase text and normalise common punctuation for matching."""
    if not text:
        return ""
    normalized = text.lower()
    # Normalize smart quotes/apostrophes
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return normalized


def _expand_term_variants(term: str) -> Set[str]:
    """Generate simple alias variants for subject matching."""
    variants: Set[str] = set()
    if not term:
        return variants

    base = re.sub(r"[^a-z0-9' ]+", " ", term.lower()).strip()
    if not base:
        return variants

    variants.add(base)
    variants.add(base.replace(" ", ""))

    if base.endswith("'s"):
        variants.add(base[:-2])
    if base.endswith("s") and len(base) > 4:
        variants.add(base[:-1])
    if "'" in base:
        variants.add(base.replace("'", ""))

    alias_terms = SUBJECT_ALIAS_MAP.get(base, set())
    for alias in alias_terms:
        alias_base = re.sub(r"[^a-z0-9' ]+", " ", alias.lower()).strip()
        if not alias_base:
            continue
        variants.add(alias_base)
        variants.add(alias_base.replace(" ", ""))
        if alias_base.endswith("'s"):
            variants.add(alias_base[:-2])
        if "'" in alias_base:
            variants.add(alias_base.replace("'", ""))

    return {variant for variant in variants if variant}


def _extract_subject_terms(market_question: str) -> tuple[Set[str], Set[str]]:
    """Identify subject terms from the market question for relevance gating.

    Returns a tuple of (raw_terms, expanded_terms) where `raw_terms` contains the
    canonical tokens appearing in the question and `expanded_terms` contains
    simple alias variants for matching within source text.
    """
    entities = {
        ent for ent in extract_entities(market_question)
        if ent not in GENERIC_ENTITY_STOPWORDS
    }
    keywords = {
        kw for kw in extract_keywords(market_question)
        if kw not in GENERIC_MARKET_TERMS
    }
    raw_terms = entities | keywords
    expanded_terms: Set[str] = set()
    for term in raw_terms:
        expanded_terms |= _expand_term_variants(term)
    return raw_terms, expanded_terms


def _text_contains_any_term(text: str, terms: Set[str]) -> bool:
    """Check whether text contains any of the provided (possibly multi-word) terms."""
    if not text or not terms:
        return False
    for term in terms:
        if term and term in text:
            return True
    return False


# Known sources database (expand this in production)
KNOWN_SOURCES = {
    # High-quality news sources
    'nytimes.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center-left',
    },
    'wsj.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center-right',
    },
    'reuters.com': {
        'credibility_score': 0.95,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center',
    },
    'bloomberg.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center',
    },
    'apnews.com': {
        'credibility_score': 0.95,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center',
    },
    'washingtonpost.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center-left',
    },
    'theguardian.com': {
        'credibility_score': 0.85,
        'source_type': 'high_quality_secondary',
        'reputation': 'credible',
        'bias_rating': 'left',
    },
    'bbc.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center',
    },
    'economist.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'highly_credible',
        'bias_rating': 'center',
    },
    'npr.org': {
        'credibility_score': 0.85,
        'source_type': 'high_quality_secondary',
        'reputation': 'credible',
        'bias_rating': 'center-left',
    },
    'propublica.org': {
        'credibility_score': 0.95,
        'source_type': 'high_quality_secondary',
        'reputation': 'investigative_journalism',
        'bias_rating': 'center-left',
    },
    # Fact-Checking & Data-Driven Sites
    'fivethirtyeight.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'data-driven',
        'bias_rating': 'center',
    },
    'politifact.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'fact_checker',
        'bias_rating': 'center-left',
    },
    'snopes.com': {
        'credibility_score': 0.85,
        'source_type': 'high_quality_secondary',
        'reputation': 'fact_checker',
        'bias_rating': 'center',
    },
    'factcheck.org': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'fact_checker',
        'bias_rating': 'center',
    },
    # Technology & Business News
    'techcrunch.com': {
        'credibility_score': 0.8,
        'source_type': 'high_quality_secondary',
        'reputation': 'tech_industry_news',
        'bias_rating': 'center',
    },
    'theverge.com': {
        'credibility_score': 0.85,
        'source_type': 'high_quality_secondary',
        'reputation': 'tech_journalism',
        'bias_rating': 'center-left',
    },
    'arstechnica.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'in_depth_tech',
        'bias_rating': 'center',
    },
    # Sports Journalism
    'espn.com': {
        'credibility_score': 0.8,
        'source_type': 'high_quality_secondary',
        'reputation': 'sports_news',
        'bias_rating': 'center',
    },
    'theathletic.com': {
        'credibility_score': 0.9,
        'source_type': 'high_quality_secondary',
        'reputation': 'in_depth_sports',
        'bias_rating': 'center',
    },
    # Arts & Entertainment
    'variety.com': {
        'credibility_score': 0.85,
        'source_type': 'high_quality_secondary',
        'reputation': 'entertainment_trade',
        'bias_rating': 'center',
    },
    'hollywoodreporter.com': {
        'credibility_score': 0.85,
        'source_type': 'high_quality_secondary',
        'reputation': 'entertainment_trade',
        'bias_rating': 'center',
    },
    # Other News Sources (Partisan or Mixed Quality)
    'foxnews.com': {
        'credibility_score': 0.5,
        'source_type': 'secondary',
        'reputation': 'mixed',
        'bias_rating': 'right',
    },
    'huffpost.com': {
        'credibility_score': 0.5,
        'source_type': 'secondary',
        'reputation': 'mixed',
        'bias_rating': 'left',
    },
    'breitbart.com': {
        'credibility_score': 0.2,
        'source_type': 'weak',
        'reputation': 'hyper_partisan',
        'bias_rating': 'far-right',
    },
    'dailymail.co.uk': {
        'credibility_score': 0.3,
        'source_type': 'weak',
        'reputation': 'tabloid',
        'bias_rating': 'right',
    },
    # Primary sources (scientific, government)
    'arxiv.org': {
        'credibility_score': 0.7,  # Pre-prints, not peer-reviewed
        'source_type': 'primary',
        'reputation': 'academic_preprint',
        'bias_rating': 'neutral',
    },
    'nature.com': {
        'credibility_score': 0.95,
        'source_type': 'primary',
        'reputation': 'top_tier_journal',
        'bias_rating': 'neutral',
    },
    'science.org': {
        'credibility_score': 0.95,
        'source_type': 'primary',
        'reputation': 'top_tier_journal',
        'bias_rating': 'neutral',
    },
    'thelancet.com': {
        'credibility_score': 0.95,
        'source_type': 'primary',
        'reputation': 'top_tier_medical_journal',
        'bias_rating': 'neutral',
    },
    'nejm.org': {
        'credibility_score': 0.95,
        'source_type': 'primary',
        'reputation': 'top_tier_medical_journal',
        'bias_rating': 'neutral',
    },
    '.gov': {  # Generic handler for government sites
        'credibility_score': 0.9,
        'source_type': 'primary',
        'reputation': 'government_source',
        'bias_rating': 'neutral',
    },
    'cdc.gov': {
        'credibility_score': 0.95,
        'source_type': 'primary',
        'reputation': 'government_health_agency',
        'bias_rating': 'neutral',
    },
    'nasa.gov': {
        'credibility_score': 0.95,
        'source_type': 'primary',
        'reputation': 'government_space_agency',
        'bias_rating': 'neutral',
    },
    'bls.gov': {
        'credibility_score': 0.95,
        'source_type': 'primary',
        'reputation': 'government_statistics',
        'bias_rating': 'neutral',
    },
    'who.int': {
        'credibility_score': 0.9,
        'source_type': 'primary',
        'reputation': 'international_health_organization',
        'bias_rating': 'neutral',
    },
    'un.org': {
        'credibility_score': 0.85,
        'source_type': 'primary',
        'reputation': 'international_organization',
        'bias_rating': 'neutral',
    },
    # Reference & Aggregators
    'wikipedia.org': {
        'credibility_score': 0.4,  # Varies, not a primary source
        'source_type': 'secondary',
        'reputation': 'encyclopedia_aggregator',
        'bias_rating': 'neutral',
    },
    'rottentomatoes.com': {
        'credibility_score': 0.5,  # Aggregates opinions, not a source of fact
        'source_type': 'secondary',
        'reputation': 'review_aggregator',
        'bias_rating': 'neutral',
    },
}
