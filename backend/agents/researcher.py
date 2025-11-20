"""
Autonomous ResearcherAgent with ReAct Pattern
Pilot implementation demonstrating autonomous reasoning + tool use
"""
import json
import re
import logging
from typing import List, Dict, Any, Literal, Optional

from langchain_core.tools import BaseTool

from agents.base import AutonomousReActAgent, AgentState
from agents.schemas import ResearcherOutput, Evidence
from tools.search import web_search_tool, multi_query_search_tool
from tools.evidence import extract_evidence_tool, verify_source_tool, ExtractedEvidence
from tools.memory import search_similar_markets_tool

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """
    Normalize a search query to detect semantic duplicates.

    - Removes punctuation
    - Lowercases
    - Sorts tokens alphabetically

    Args:
        query: Search query string.

    Returns:
        Normalized string with sorted tokens.
    """
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    return " ".join(sorted(cleaned.split()))


class AutonomousResearcherAgent(AutonomousReActAgent):
    """
    Autonomous Researcher Agent that gathers and scores evidence via iterative reasoning (ReAct).

    Capabilities:
      - Autonomous multi-angle search strategy
      - Quality validation and source verification
      - Structured evidence extraction with LLR estimation
      - Memory-aware deduplication and circuit breakers
    """

    def __init__(
        self,
        direction: Literal["pro", "con", "general"] = "general",
        min_evidence_items: int = 5,
        max_search_attempts: int = 10,
        **kwargs: Any,
    ):
        """
        Args:
            direction: Research direction focus ("pro", "con", "general").
            min_evidence_items: Minimum evidence items before completion.
            max_search_attempts: Maximum search queries to try.
            **kwargs: Forwarded to AutonomousReActAgent.
        """
        assert direction in {"pro", "con", "general"}
        assert min_evidence_items >= 1 and max_search_attempts >= 1
        super().__init__(**kwargs)
        self.direction = direction
        self.min_evidence_items = min_evidence_items
        self.max_search_attempts = max_search_attempts
        self.logger.info(
            f"AutonomousResearcherAgent init: dir={direction}, min_evidence={min_evidence_items}, max_search={max_search_attempts}"
        )

    def get_system_prompt(self) -> str:
        """
        System prompt for autonomous research with tool usage guidelines.
        """
        direction_instruction = {
            "pro": "You are seeking evidence that SUPPORTS a YES outcome.",
            "con": "You are seeking evidence that SUPPORTS a NO outcome.",
            "general": "You are seeking neutral contextual evidence.",
        }

        return f"""You are an Autonomous Researcher Agent in POLYSEER.

{direction_instruction[self.direction]}

## Your Mission
Find HIGH-QUALITY, VERIFIABLE evidence for the given subclaims related to a prediction market question.
You will use multiple tools iteratively until you have gathered sufficient evidence.

## 🎭 MULTI-PERSPECTIVE GATHERING (Phase 2 Enhancement)

**IMPORTANT**: When gathering evidence, you must consider THREE perspectives:

1. **Assertive Perspective** - Evidence supporting confident YES outcome
   - Search for: Strong confirmations, success stories, positive indicators
   - Examples: "X achieved Y", "expert predicts success", "data shows improvement"
   - Tag evidence with: perspective="assertive"

2. **Skeptical Perspective** - Evidence supporting cautious NO outcome
   - Search for: Challenges, failures, negative indicators, limitations
   - Examples: "X failed at Y", "concerns about Z", "historical failure rate"
   - Tag evidence with: perspective="skeptical"

3. **Neutral Perspective** - Contextual, balanced, or ambiguous evidence
   - Search for: Background info, base rates, general context, mixed signals
   - Examples: "typical outcomes for X", "historical precedent", "mixed results"
   - Tag evidence with: perspective="neutral"

**Your {self.direction.upper()} direction primarily determines your SEARCH FOCUS, but you should still gather evidence from all three perspectives within that focus.**

**Examples**:
- If direction="pro", focus on YES evidence, but include both assertive PRO evidence ("strong success signals") and skeptical CONTEXT ("challenges to overcome")
- If direction="con", focus on NO evidence, but include both assertive CON evidence ("clear failure indicators") and neutral CONTEXT ("historical base rates")

**Tagging Guidelines**:
- After extracting evidence, manually add 'perspective' field to each evidence item
- Track perspective counts as you gather evidence
- Aim for balanced gathering within your direction (at least 2-3 items per perspective)

## Available Tools
1. **web_search_tool** - Search the web for information.
2. **multi_query_search_tool** - Execute multiple searches in parallel.
3. **verify_source_tool** - Check source credibility.
4. **extract_evidence_tool** - Parse best results into structured evidence.
5. **search_similar_markets_tool** - Learn from similar past analyses.

## Task Input Format
- **search_seeds**: Starting queries (use exactly as provided first).
- **subclaims**: Claims to find evidence for.
- **market_question**: Main prediction market question.

**CRITICAL**: Always use the provided search_seeds for initial searches.

## Reasoning Process
**Step 1: Understand the Task**
- Review market question, subclaims, and search seeds.
- Check similar markets (search_similar_markets_tool).
- Plan initial strategy using the exact seeds.

**Step 2: Execute Searches**
- Start with provided seeds using web_search_tool/multi_query_search_tool.
- If results are poor, try alternative queries on same topics with names/dates/context.

**MEMORY AWARENESS BEFORE EVERY SEARCH**
- Use intermediate_results['attempted_queries'] and 'last_N_queries' to avoid repeats.
- If last 3 queries identical → STOP & adapt with different angle/timeframe/source.

**ADAPTIVE STRATEGY (when 'force_new_strategy' is True)**
- Change timeframe/specificity/sources/angle as needed.

**Step 3: Extract Evidence + Tag Perspective (CRITICAL - REQUIRED)**
- You MUST call extract_evidence_tool on promising search results to complete your task.
- extract_evidence_tool takes a search_result dict and returns structured evidence.
- You need at least {self.min_evidence_items} evidence items to finish - searches alone are not enough.
- Call extract_evidence_tool immediately after finding good search results.
- The evidence will be automatically stored - no additional action needed.

**MEMORY AWARENESS FOR EXTRACTION**
- Skip URLs in 'blocked_urls' or attempted 2+ times.
- If extraction returns null twice → block URL and move on.

**Step 5: Decide Completion**
- Enough evidence (≥{self.min_evidence_items}), diversity, or diminishing returns → finish.
- Absence of evidence is information; do not loop endlessly.

## Quality Standards
- Verifiable, recent, diverse, specific, relevant.

## Response Format
- After extract_evidence_tool returns, immediately store the structured object in intermediate_results['evidence_items'].
- Do not write human-readable summaries—store objects only.
- Completion message only; extract_final_output will parse stored evidence.

Remember: Quality over quantity. {self.min_evidence_items} excellent sources beats 50 weak ones.
"""

    def get_tools(self) -> List[BaseTool]:
        """Return available research tools."""
        tools: List[BaseTool] = [
            web_search_tool,
            multi_query_search_tool,
            extract_evidence_tool,
            verify_source_tool,
        ]
        if getattr(self, "store", None):
            tools.append(search_similar_markets_tool)
        return tools

    def _initialize_memory_tracking(self, state: AgentState) -> None:
        """Ensure all memory tracking containers exist (idempotent)."""
        intermediate = state.setdefault("intermediate_results", {})
        for k, factory in (
            ("attempted_queries", set),
            ("query_results", dict),
            ("last_N_queries", list),
            ("blocked_urls", set),
        ):
            intermediate.setdefault(k, factory())
        intermediate.setdefault("consecutive_failed_searches", 0)
        intermediate.setdefault("extraction_success_rate", 1.0)

    def _select_relevant_subclaim(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """
        Pick the best subclaim to attach evidence to.
        Preference order:
          1) First subclaim matching this agent's direction (pro/con)
          2) Any subclaim (fallback)
        """
        task_input = state.get("task_input", {})
        subclaims = task_input.get("subclaims", [])
        assert isinstance(subclaims, list), "Subclaims must be list"
        for sc in subclaims:
            assert isinstance(sc, dict), "Subclaim must be dict"
            if sc.get("direction") == self.direction:
                return sc
        return subclaims[0] if subclaims else None

    async def handle_tool_message(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_message: Any,
    ) -> None:
        """
        Track search usage and persist extracted evidence with memory-aware safeguards.
        - Supports both web_search_tool and multi_query_search_tool outputs.
        - Always tries to extract from the top few fresh results after a search.
        """
        intermediate = state.setdefault("intermediate_results", {})
        self._initialize_memory_tracking(state)

        # Helper to parse tool_message payload into a list of result dicts
        def _parse_results() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []
            artifact = getattr(tool_message, "artifact", None)
            content = getattr(tool_message, "content", None)

            # multi_query_search_tool typically returns a dict: {query: [results]}
            if tool_name == "multi_query_search_tool":
                if isinstance(artifact, dict):
                    for v in artifact.values():
                        if isinstance(v, list):
                            results.extend([r for r in v if isinstance(r, dict)])
                elif isinstance(content, dict):
                    for v in content.values():
                        if isinstance(v, list):
                            results.extend([r for r in v if isinstance(r, dict)])
                elif isinstance(content, str):
                    assert content.strip().startswith("{") or content.strip().startswith("["), "Content must be JSON"
                    parsed = json.loads(content)
                    assert isinstance(parsed, dict), "Parsed content must be dict"
                    for v in parsed.values():
                        if isinstance(v, list):
                            results.extend([r for r in v if isinstance(r, dict)])
            # web_search_tool usually returns a list
            elif tool_name == "web_search_tool":
                if isinstance(artifact, list):
                    results = [r for r in artifact if isinstance(r, dict)]
                elif isinstance(content, list):
                    results = [r for r in content if isinstance(r, dict)]
                elif isinstance(content, str):
                    assert content.strip().startswith("[") or content.strip().startswith("{"), "Content must be JSON"
                    parsed = json.loads(content)
                    assert isinstance(parsed, list), "Parsed content must be list"
                    results = [r for r in parsed if isinstance(r, dict)]
            return results

        # Record queries used (normalize to avoid duplicates)
        def _record_queries() -> int:
            inc = 0
            if tool_name == "web_search_tool":
                q = tool_args.get("query", "")
                if q:
                    normalized = normalize_query(q)
                    intermediate["attempted_queries"].add(normalized)
                    last = intermediate["last_N_queries"]
                    last.append(normalized)
                    intermediate["last_N_queries"] = last[-10:]
                    inc = 1
            elif tool_name == "multi_query_search_tool":
                qs = tool_args.get("queries", []) or []
                for q in qs:
                    if not isinstance(q, str):
                        continue
                    normalized = normalize_query(q)
                    intermediate["attempted_queries"].add(normalized)
                    last = intermediate["last_N_queries"]
                    last.append(normalized)
                    intermediate["last_N_queries"] = last[-10:]
                inc = max(len(qs), 1)
            intermediate["search_count"] = intermediate.get("search_count", 0) + inc
            return inc

        # --- Handle search tools: parse results and auto-extract from top K ---
        if tool_name in {"web_search_tool", "multi_query_search_tool"}:
            _record_queries()
            results_list = _parse_results()
            result_count = len(results_list)
            # Store result count for the latest queries (best-effort)
            if tool_name == "web_search_tool":
                q = tool_args.get("query", "")
                if q:
                    intermediate["query_results"][normalize_query(q)] = result_count
            elif tool_name == "multi_query_search_tool":
                qs = tool_args.get("queries", []) or []
                for q in qs:
                    if isinstance(q, str):
                        intermediate["query_results"][normalize_query(q)] = result_count

            # Early exit if no results to process
            if result_count == 0:
                return

            # Choose a subclaim to attach evidence to
            target_sc = self._select_relevant_subclaim(state)
            subclaim_id = (target_sc.get("id") or target_sc.get("text")) if target_sc else "unknown"
            subclaim_text = target_sc.get("text") if target_sc else ""

            # Auto-extract from top few fresh results until we hit min_evidence_items
            evidence_items = intermediate.setdefault("evidence_items", [])
            blocked = intermediate.setdefault("blocked_urls", set())
            attempted_urls = intermediate.setdefault("attempted_urls", {})
            top_k = 3  # try the first 3 results per search
            extracted = 0

            for sr in results_list[:top_k]:
                url = sr.get("url") or sr.get("link") or ""
                if not url or url in blocked:
                    continue
                if attempted_urls.get(url, 0) >= 2:
                    continue

                # Basic de-dup check against existing evidence
                def _url_of(item: Any) -> str:
                    if isinstance(item, ExtractedEvidence):
                        return item.url
                    if isinstance(item, dict):
                        return item.get("url", "")
                    return getattr(item, "url", "")

                if any(_url_of(e) == url for e in evidence_items):
                    continue

                market_q = state.get("task_input", {}).get("market_question", "")
                assert isinstance(market_q, str), "Market question must be string"
                if subclaim_text:
                    if self.direction == "pro":
                        enhanced_claim = f"Evidence supporting YES: {subclaim_text}"
                    elif self.direction == "con":
                        enhanced_claim = f"Evidence supporting NO: {subclaim_text}"
                    else:
                        enhanced_claim = subclaim_text
                else:
                    enhanced_claim = subclaim_id

                extraction_result = await extract_evidence_tool.ainvoke({
                    "search_result": sr,
                    "subclaim": enhanced_claim,
                    "market_question": market_q
                })
                if not extraction_result:
                    attempted_urls[url] = attempted_urls.get(url, 0) + 1
                    if attempted_urls[url] >= 2:
                        blocked.add(url)
                    continue

                # Success path: persist evidence
                attempted_urls[url] = attempted_urls.get(url, 0) + 1
                evidence_items.append(extraction_result)
                extracted += 1
                self.logger.info(f"Auto-extracted evidence from search result: {sr.get('title', 'Unknown')[:80]}")

                # Stop early if we reached the per-agent minimum
                if len(evidence_items) >= self.min_evidence_items:
                    break

            # If we extracted nothing from non-empty results, nudge the success rate down
            if extracted == 0:
                current_rate = intermediate.get("extraction_success_rate", 1.0)
                intermediate["extraction_success_rate"] = max(0.0, 0.7 * current_rate - 0.1)

            return

        # --- Handle direct extraction tool messages (manual calls from the LLM) ---
        if tool_name != "extract_evidence_tool":
            return

        extraction_url = tool_args.get("search_result", {}).get("url", "unknown")
        attempted_urls = intermediate.setdefault("attempted_urls", {})
        attempted_urls[extraction_url] = attempted_urls.get(extraction_url, 0) + 1

        if extraction_url in intermediate.get("blocked_urls", set()):
            return

        evidence = self._coerce_extract_evidence(tool_message)

        if evidence is None:
            failed = intermediate.setdefault("failed_extractions", [])
            failed.append(extraction_url)
            intermediate["consecutive_failed_searches"] += 1
            if attempted_urls[extraction_url] >= 2:
                intermediate["blocked_urls"].add(extraction_url)
            return

        # success path
        intermediate["failed_extractions"] = []
        intermediate["consecutive_failed_searches"] = 0
        current_rate = intermediate.get("extraction_success_rate", 1.0)
        intermediate["extraction_success_rate"] = 0.7 * current_rate + 0.3

        evidence_items = intermediate.setdefault("evidence_items", [])
        new_url = getattr(evidence, "url", "") or ""

        def _url_of(item: Any) -> str:
            if isinstance(item, ExtractedEvidence):
                return item.url
            if isinstance(item, dict):
                return item.get("url", "")
            return getattr(item, "url", "")

        if any(_url_of(ex) and new_url and _url_of(ex) == new_url for ex in evidence_items):
            return

        evidence_items.append(evidence)

    def _coerce_extract_evidence(self, tool_message: Any) -> Optional[ExtractedEvidence]:
        """
        Convert a tool message payload into an ExtractedEvidence object.
        Supports artifacts, dicts, Pydantic-like objects, JSON text, and repr-style key=value text.
        """
        artifact = getattr(tool_message, "artifact", None)
        if artifact is None:
            artifact = getattr(tool_message, "additional_kwargs", {}).get("return_value")

        if isinstance(artifact, ExtractedEvidence):
            return artifact
        if isinstance(artifact, dict):
            return ExtractedEvidence(**artifact)
        if hasattr(artifact, "model_dump"):
            return ExtractedEvidence(**artifact.model_dump())

        text_payload = self._message_text(tool_message)
        if not text_payload:
            return None

        if text_payload.strip().startswith("{") or text_payload.strip().startswith("["):
            data = json.loads(text_payload)
            assert isinstance(data, dict), "Parsed data must be dict"
            return ExtractedEvidence(**data)

        kv_pairs = re.findall(r'(\w+)=(".*?"|\'.*?\'|[^\s,]+)', text_payload)
        if not kv_pairs:
            return None

        parsed: Dict[str, Any] = {}
        for key, raw in kv_pairs:
            val = raw.strip().strip('"\'')

            if val.endswith((",", ")")):
                val = val.rstrip(",)")
            parsed[key] = val

        for fk in ("verifiability_score", "independence_score", "recency_score", "estimated_LLR"):
            if fk in parsed:
                val = parsed[fk]
                if isinstance(val, str):
                    val = val.rstrip(",)")
                assert val.replace(".", "").replace("-", "").isdigit() or val == "", f"{fk} must be numeric"
                parsed[fk] = float(val) if val else 0.0

        if "support" in parsed:
            parsed["support"] = str(parsed["support"]).lower()

        assert all(k in parsed for k in ["subclaim_id", "title", "url"]), "Missing required fields"
        return ExtractedEvidence(**parsed)

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Decide if the research task is complete using circuit breakers and diminishing-returns checks.
        """
        intermediate = state.get("intermediate_results", {})
        evidence_count = len(intermediate.get("evidence_items", []))
        search_count = intermediate.get("search_count", 0)
        iteration = state.get("iteration_count", 0)

        self._initialize_memory_tracking(state)

        # Hard guard: do not declare completion with zero evidence unless we've clearly exhausted attempts
        if evidence_count == 0 and search_count < max(6, self.max_search_attempts // 2):
            return False

        if evidence_count >= self.min_evidence_items:
            return True

        last_queries = intermediate.get("last_N_queries", [])
        if len(last_queries) >= 3:
            recent_3 = last_queries[-3:]
            if len(set(recent_3)) == 1:
                return True
            words_sets = [set(q.split()) for q in recent_3]
            all_words = set().union(*words_sets)
            if all_words:
                overlap = len(words_sets[0] & words_sets[1] & words_sets[2]) / len(all_words)
                if overlap > 0.8:
                    intermediate["force_new_strategy"] = True

        query_results = intermediate.get("query_results", {})
        if len(query_results) >= 3:
            recent = last_queries[-3:]
            if sum(query_results.get(q, 0) for q in recent) == 0:
                return True

        attempted_urls_count = len(intermediate.get("attempted_urls", {}))
        if attempted_urls_count >= 5 and intermediate.get("extraction_success_rate", 1.0) < 0.2:
            return True

        if len(intermediate.get("blocked_urls", set())) >= 5 and evidence_count >= 3:
            return True

        if intermediate.get("consecutive_failed_searches", 0) >= 5:
            return True

        if evidence_count >= 3:
            last_msg = state.get("messages", [])[-1] if state.get("messages") else None
            has_tool_calls = bool(getattr(last_msg, "tool_calls", None)) if last_msg else False
            if not has_tool_calls and iteration >= 3:
                return True

        if search_count >= self.max_search_attempts:
            return True

        if search_count >= 4 and evidence_count <= 2 and iteration >= 6:
            return True

        if evidence_count >= 3 and search_count >= 6 and iteration >= 8:
            return True

        if len(intermediate.get("failed_extractions", [])) >= 5:
            return True

        last_ev_ct = intermediate.get("_last_evidence_count", 0)
        if iteration > 0 and evidence_count == last_ev_ct:
            stagnant = intermediate.get("_stagnant_iterations", 0) + 1
            intermediate["_stagnant_iterations"] = stagnant
            if stagnant >= 3 and iteration >= 5:
                return True
        else:
            intermediate["_stagnant_iterations"] = 0

        intermediate["_last_evidence_count"] = evidence_count

        if iteration >= 15:
            return True

        return False

    async def extract_final_output(self, state: AgentState) -> ResearcherOutput:
        """
        Build a ResearcherOutput from final agent state.
        """
        evidence_items = state.get("intermediate_results", {}).get("evidence_items", [])
        evidence_list: List[Evidence] = []

        for item in evidence_items:
            if isinstance(item, Evidence):
                evidence_list.append(item)
                continue
            if isinstance(item, dict):
                evidence_list.append(Evidence(**item))
                continue
            if isinstance(item, ExtractedEvidence):
                from datetime import datetime, date as _date

                pub_date_str = getattr(item, "published_date", "unknown")
                if pub_date_str != "unknown" and isinstance(pub_date_str, str):
                    assert len(pub_date_str) == 10 and pub_date_str.count("-") == 2, f"Invalid date format: {pub_date_str}"
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()
                else:
                    pub_date = _date.today()

                evidence_list.append(
                    Evidence(
                        subclaim_id=item.subclaim_id,
                        title=item.title,
                        url=item.url,
                        published_date=pub_date,
                        source_type=item.source_type,
                        claim_summary=item.claim_summary,
                        support=item.support,
                        verifiability_score=item.verifiability_score,
                        independence_score=item.independence_score,
                        recency_score=item.recency_score,
                        estimated_LLR=item.estimated_LLR,
                        extraction_notes=item.extraction_notes,
                    )
                )

        pro_count = sum(1 for e in evidence_list if e.support == "pro")
        con_count = sum(1 for e in evidence_list if e.support == "con")
        neutral_count = len(evidence_list) - pro_count - con_count

        total_pro_llr = sum(e.estimated_LLR for e in evidence_list if e.support == "pro" and e.estimated_LLR > 0)
        total_con_llr = sum(abs(e.estimated_LLR) for e in evidence_list if e.support == "con" and e.estimated_LLR < 0)
        net_llr = sum(e.estimated_LLR for e in evidence_list)

        subclaims_data = state.get("task_input", {}).get("subclaims", [])
        subclaim_direction_map = {sc.get("id"): sc.get("direction") for sc in subclaims_data if sc.get("id") and sc.get("direction")}
        directional_items = aligned_items = 0
        for ev in evidence_list:
            expected = subclaim_direction_map.get(ev.subclaim_id)
            if expected in {"pro", "con"} and ev.support != "neutral":
                directional_items += 1
                if ev.support == expected:
                    aligned_items += 1

        context_alignment_score = (aligned_items / directional_items) if directional_items else 0.0

        # Log evidence gathering with rich logger
        self.rich_logger.log_evidence_gathering(
            direction=self.direction,
            count=len(evidence_list),
            total_llr=net_llr,
            sample_evidence=[
                {"title": e.title[:60], "estimated_LLR": e.estimated_LLR}
                for e in evidence_list[:3]
            ],
        )

        self.logger.info(
            f"Final: items={len(evidence_list)} pro={pro_count} con={con_count} neutral={neutral_count} netLLR={net_llr:+.2f} align={context_alignment_score:.2f}"
        )

        search_strategy = (
            f"{self.direction.upper()} search captured {len(evidence_list)} items "
            f"(pro={pro_count}, con={con_count}, neutral={neutral_count}); net LLR {net_llr:+.2f}"
        )

        return ResearcherOutput(
            evidence_items=evidence_list,
            total_pro_count=pro_count,
            total_con_count=con_count,
            total_pro_llr=total_pro_llr,
            total_con_llr=total_con_llr,
            net_llr=net_llr,
            context_alignment_score=context_alignment_score,
            search_strategy=search_strategy,
        )

    async def run_research(
        self,
        search_seeds: List[str],
        subclaims: List[Dict[str, Any]],
        market_question: str,
        **kwargs: Any,
    ) -> ResearcherOutput:
        """
        Compatibility wrapper with the older ResearcherAgent interface.
        """
        assert isinstance(search_seeds, list) and isinstance(subclaims, list) and isinstance(market_question, str)
        
        # Log memory context if available
        if self.store and self.enable_auto_memory_query:
            self.logger.info(f"Querying memory for similar markets before {self.direction.upper()} research")
        
        return await self.run(
            task_description=f"Gather {self.direction.upper()} evidence for market question",
            task_input={"search_seeds": search_seeds, "subclaims": subclaims, "market_question": market_question, **kwargs},
        )

    def integrity_report(self, state: AgentState) -> Dict[str, Any]:
        """
        Compact integrity report for diagnostics.

        Returns:
            Dict with key metrics and simple rule-based violations.
        """
        inter = state.get("intermediate_results", {})
        ev_ct = len(inter.get("evidence_items", []))
        srch_ct = inter.get("search_count", 0)
        uniq_q = len(inter.get("attempted_queries", set()))
        blocked = len(inter.get("blocked_urls", set()))
        attempted_urls = len(inter.get("attempted_urls", {}))
        succ_rate = inter.get("extraction_success_rate", 1.0)
        coverage = ev_ct / max(1, self.min_evidence_items)

        violations = []
        if srch_ct > self.max_search_attempts:
            violations.append("search_attempts_exceeded")
        if succ_rate < 0.2 and attempted_urls >= 5:
            violations.append("low_extraction_success")
        if blocked >= 5 and ev_ct < 3:
            violations.append("too_many_blocked_with_low_evidence")

        return {
            "evidence_count": ev_ct,
            "search_count": srch_ct,
            "unique_queries": uniq_q,
            "attempted_urls": attempted_urls,
            "blocked_urls": blocked,
            "extraction_success_rate": round(float(succ_rate), 3),
            "coverage_ratio": round(coverage, 3),
            "violations": violations,
        }


# Convenience function for parallel execution
async def run_parallel_autonomous_research(
    search_seeds_pro: List[str],
    search_seeds_con: List[str],
    search_seeds_general: List[str],
    subclaims: List[Dict[str, Any]],
    market_question: str,
    **kwargs: Any,
) -> Dict[str, ResearcherOutput]:
    """
    Run PRO, CON, and GENERAL autonomous researchers in parallel.

    Returns:
        Dict[str, ResearcherOutput] with 'pro', 'con', and 'general' keys.
    """
    import asyncio

    assert all(isinstance(x, list) for x in (search_seeds_pro, search_seeds_con, search_seeds_general))
    assert isinstance(subclaims, list) and isinstance(market_question, str)

    # Extract store to determine if memory querying should be enabled
    store = kwargs.get('store')
    agent_kwargs = {**kwargs, 'enable_auto_memory_query': store is not None}

    researchers = {
        "pro": AutonomousResearcherAgent(direction="pro", **agent_kwargs),
        "con": AutonomousResearcherAgent(direction="con", **agent_kwargs),
        "general": AutonomousResearcherAgent(direction="general", **agent_kwargs),
    }

    results = await asyncio.gather(
        researchers["pro"].run_research(search_seeds_pro, subclaims, market_question),
        researchers["con"].run_research(search_seeds_con, subclaims, market_question),
        researchers["general"].run_research(search_seeds_general, subclaims, market_question),
        return_exceptions=True,
    )

    output: Dict[str, ResearcherOutput] = {}
    for key, result in zip(("pro", "con", "general"), results):
        if isinstance(result, Exception):
            logger.error(f"{key.upper()} autonomous research failed: {result}")
            output[key] = ResearcherOutput(evidence_items=[])
        else:
            output[key] = result
    return output