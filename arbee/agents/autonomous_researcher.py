"""
Autonomous ResearcherAgent with ReAct Pattern
Pilot implementation demonstrating autonomous reasoning + tool use
"""
import json
import re
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import ResearcherOutput, Evidence
from arbee.tools.search import web_search_tool, multi_query_search_tool
from arbee.tools.evidence import extract_evidence_tool, verify_source_tool, ExtractedEvidence
from arbee.tools.memory_search import search_similar_markets_tool

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """
    Normalize query to detect semantic duplicates.

    Removes punctuation, lowercases, sorts tokens alphabetically.
    This allows detection of queries like:
    - "Diplo 5k time" vs "Diplo time 5k" (different order)
    - "Diplo 5k time!" vs "diplo 5k time" (case/punctuation)

    Args:
        query: Search query string

    Returns:
        Normalized query string with sorted tokens
    """
    # Remove punctuation and lowercase
    cleaned = re.sub(r'[^\w\s]', '', query.lower())
    # Sort tokens alphabetically
    tokens = sorted(cleaned.split())
    return ' '.join(tokens)


class AutonomousResearcherAgent(AutonomousReActAgent):
    """
    Autonomous Researcher Agent - Gathers and scores evidence using iterative reasoning

    This agent demonstrates the full ReAct pattern:
    1. Observes what evidence is needed (from subclaims and search seeds)
    2. Thinks about search strategy (which queries to try first)
    3. Acts by calling web search tools
    4. Observes search results quality
    5. Validates if sufficient evidence gathered
    6. Extracts structured evidence from best results
    7. Continues or terminates based on quality/quantity

    Capabilities:
    - Autonomous search strategy (tries different queries if needed)
    - Quality validation (checks if results are sufficient)
    - Source verification (assesses credibility)
    - Evidence extraction (structured parsing with LLR estimation)
    - Learning from similar cases (memory search)
    """

    def __init__(
        self,
        direction: Literal["pro", "con", "general"] = "general",
        min_evidence_items: int = 5,
        max_search_attempts: int = 10,
        **kwargs
    ):
        """
        Initialize Autonomous Researcher Agent

        Args:
            direction: Research direction (pro/con/general)
            min_evidence_items: Minimum evidence items before completion
            max_search_attempts: Maximum search queries to try
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(**kwargs)
        self.direction = direction
        self.min_evidence_items = min_evidence_items
        self.max_search_attempts = max_search_attempts

        self.logger.info(
            f"AutonomousResearcherAgent initialized: direction={direction}, "
            f"min_evidence={min_evidence_items}"
        )

    def get_system_prompt(self) -> str:
        """
        System prompt for autonomous research with tool usage guidelines
        """
        direction_instruction = {
            "pro": "You are seeking evidence that SUPPORTS a YES outcome.",
            "con": "You are seeking evidence that SUPPORTS a NO outcome.",
            "general": "You are seeking neutral contextual evidence."
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

You have access to these tools:

1. **web_search_tool** - Search the web for information
   - Use this to find articles, reports, polls, studies
   - Be specific in your queries for better results
   - Try different phrasings if first search doesn't work well

2. **multi_query_search_tool** - Execute multiple searches in parallel
   - Use this when you have several different search angles
   - More efficient than multiple individual searches

3. **verify_source_tool** - Check source credibility
   - Use this before extracting evidence from a source
   - Helps assess if source is trustworthy

4. **extract_evidence_tool** - Parse search result into structured evidence
   - Use this on your best search results
   - Automatically estimates LLR and quality scores

5. **search_similar_markets_tool** - Find similar past analyses
   - Use this at the start to learn from similar cases
   - Can inform your search strategy

## Task Input Format

Your task input will contain:
- **search_seeds**: List of specific search queries to start with
- **subclaims**: List of specific claims to find evidence for
- **market_question**: The main prediction market question

**CRITICAL**: Always use the exact search_seeds provided in your task input for initial searches.

## Your Reasoning Process

Follow these steps iteratively:

**Step 1: Understand the Task**
- Review market question, subclaims, and search seeds provided in your task input
- Check if similar markets have been analyzed before (use search_similar_markets_tool)
- Plan initial search strategy using the exact search seeds provided

**Step 2: Execute Searches**
- START WITH PROVIDED SEARCH SEEDS: Use the exact search seeds in your task input
- Use web_search_tool or multi_query_search_tool
- If results are poor, try alternative queries based on the same topics
- Always include specific names, dates, and contexts from the market question

**MEMORY AWARENESS (CRITICAL - CHECK BEFORE EVERY SEARCH)**:

Before calling web_search_tool or multi_query_search_tool, YOU MUST check your memory:

1. **Check attempted_queries** in intermediate_results:
   - Before searching, check if query was already attempted
   - Use: intermediate_results.get('attempted_queries', set())
   - If query in attempted_queries → DON'T search again
   - Generate a DIFFERENT query instead

2. **Check last_N_queries** for repetition:
   - Get last queries: intermediate_results.get('last_N_queries', [])
   - If last 3 queries are identical → STOP and use adaptive strategy
   - This means you're stuck - try completely different approach

3. **Example of CORRECT behavior**:
   - Iteration 1: Search "Diplo 5k time 2025" → added to attempted_queries
   - Iteration 2: Check memory, see "Diplo 5k time 2025" was tried
   - Iteration 2: Use DIFFERENT query like "Diplo Run Club performance"
   - WRONG: Searching "Diplo 5k time 2025" again without checking

**If you attempt a query already in memory, you are wasting iterations and will timeout!**

**ADAPTIVE STRATEGY (Important)**:
- If `intermediate_results['force_new_strategy']` is True, you MUST try completely different approaches:
  - Change timeframe: "2025" → "recent", "last 6 months", "2024"
  - Change specificity: "5k time" → "running events", "athletic performance"
  - Change sources: Try different domains, forums, social media
  - Change angle: Instead of direct facts, look for context, commentary, related events
- Examples of adaptive queries:
  - Original: "Diplo 5k running time 2025"
  - Adaptive: "Diplo fitness level", "Diplo Run Club events", "Diplo athletic background"

**Step 3: Extract Evidence with Perspective Tagging**
- Select best search results (relevant, credible, recent)
- Optionally verify sources using verify_source_tool
- Use extract_evidence_tool to parse results into structured evidence
- **CRITICAL**: After getting evidence from extract_evidence_tool, TAG IT with perspective:
  - Analyze the evidence content
  - Determine if it's assertive (strong YES/NO signal), skeptical (concerns/limitations), or neutral (context/background)
  - Add 'perspective' field to evidence item dict
  - Example: evidence_item['perspective'] = 'assertive' or 'skeptical' or 'neutral'
- **IMPORTANT**: Store tagged evidence in intermediate_results['evidence_items']
- Use: intermediate_results['evidence_items'].append(evidence_item) for each new piece of evidence
- Continue until you have {self.min_evidence_items}+ quality items with balanced perspectives

**MEMORY AWARENESS FOR EXTRACTION (CRITICAL)**:

Before calling extract_evidence_tool, YOU MUST check if URL is safe to use:

1. **Check blocked_urls** in intermediate_results:
   - Get blocked URLs: intermediate_results.get('blocked_urls', set())
   - Get URL from search result
   - If URL in blocked_urls → Skip this URL, move to next result
   - Blocked URLs failed multiple times and will fail again

2. **Check attempted_urls** for previous failures:
   - Get attempted URLs: intermediate_results.get('attempted_urls', dict)
   - If URL attempted 2+ times → Skip to avoid wasting iterations
   - These URLs are likely to fail again

3. **If extraction returns null**:
   - URL is automatically blocked after 2 failures
   - Move to NEXT search result immediately
   - DON'T retry the same URL

**If you keep trying blocked URLs, you waste iterations and will timeout!**

**Step 5: Decide Completion**
- Check if you have sufficient evidence ({self.min_evidence_items}+ items) in intermediate_results['evidence_items']
- Check if evidence is diverse (not all from same source)
- **IMPORTANT**: Recognize when to stop - absence of evidence IS meaningful information!
- If yes → task complete, return evidence
- If searches consistently return no relevant results → **STOP** - this tells us something important
- If you have 3-4 quality items after extensive searching → **STOP** - this is likely all available
- If no and haven't hit search limit AND searches are productive → continue searching
- If search limit reached → return what you have

**Recognizing Diminishing Returns:**
- If your last 3-4 searches produced NO usable evidence → **STOP SEARCHING**
- The fact that evidence is scarce or absent is itself valuable information
- Don't loop endlessly hoping for better results - accept the data you have
- Example: If you can't find specific 5k times for someone, that absence is meaningful

## Quality Standards

- **Verifiable**: Primary sources and high-quality journalism preferred
- **Recent**: Prioritize recent information (within last 90 days if possible)
- **Diverse**: Avoid over-relying on single source or echo chamber
- **Specific**: Concrete claims with numbers/dates better than vague statements
- **Relevant**: Must relate to subclaims, not tangential information

## Important Guidelines

- **Think before acting**: Explain your reasoning before each tool call
- **Validate quality**: Don't just collect evidence, ensure it's high quality
- **Try alternatives**: If a search strategy isn't working, try different queries
- **Know when to stop**: Don't endlessly search if you have sufficient evidence
- **Be efficient**: Use parallel searches when possible

## Response Format

**CRITICAL: Evidence Storage**
- **IMMEDIATELY** after calling extract_evidence_tool, store the result
- **DO NOT** summarize evidence in text - store the structured evidence objects
- **ALWAYS** use: intermediate_results['evidence_items'].append(evidence_item)
- **DO NOT** write human-readable summaries of evidence

**Example Correct Usage:**
```
# After tool call returns evidence
evidence = extract_evidence_tool(...)  # This returns an ExtractedEvidence object
if evidence:
    if 'evidence_items' not in intermediate_results:
        intermediate_results['evidence_items'] = []
    intermediate_results['evidence_items'].append(evidence)
    # DO NOT write summaries like "I found evidence that..."
    # DO NOT write human-readable text
    # ONLY store the structured evidence object
```

**CRITICAL RESPONSE FORMAT:**
- If you call extract_evidence_tool and get a result, IMMEDIATELY store it
- Your response should be: "Evidence stored successfully" or similar confirmation
- DO NOT include the evidence details in your response text
- The evidence is stored in intermediate_results['evidence_items']

**When you think you're done:**
- Your final message should only confirm completion
- The extract_final_output function will parse intermediate_results['evidence_items'] into ResearcherOutput format
- Make sure all evidence items are stored in intermediate_results['evidence_items']

Remember: Quality over quantity. {self.min_evidence_items} excellent sources beats 50 weak ones.
"""

    def get_tools(self) -> List[BaseTool]:
        """
        Return research tools available to this agent
        """
        # Core research tools
        tools = [
            web_search_tool,
            multi_query_search_tool,
            extract_evidence_tool,
            verify_source_tool,
        ]

        # Memory tools (if store is configured)
        if self.store:
            tools.append(search_similar_markets_tool)

        return tools

    def _initialize_memory_tracking(self, state: AgentState) -> None:
        """
        Initialize memory tracking structures if not present.

        This ensures all memory tracking dicts/lists/sets exist in intermediate_results.
        Safe to call multiple times (idempotent).

        Args:
            state: Current agent state
        """
        intermediate = state.setdefault('intermediate_results', {})

        # Query deduplication
        intermediate.setdefault('attempted_queries', set())
        intermediate.setdefault('query_results', {})
        intermediate.setdefault('last_N_queries', [])

        # Progress tracking
        intermediate.setdefault('consecutive_failed_searches', 0)
        intermediate.setdefault('extraction_success_rate', 1.0)
        intermediate.setdefault('evidence_growth_history', [])

        # URL blocking
        intermediate.setdefault('blocked_urls', set())

    def handle_tool_message(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_message
    ) -> None:
        """
        Automatically track search usage and persist extracted evidence.
        Enhanced with memory tracking for deduplication and circuit breakers.
        """
        intermediate = state.setdefault('intermediate_results', {})

        # Initialize memory tracking structures
        self._initialize_memory_tracking(state)

        # Track search queries for deduplication
        if tool_name in {"web_search_tool", "multi_query_search_tool"}:
            increment = 1
            artifact = getattr(tool_message, "artifact", None)
            if tool_name == "multi_query_search_tool" and isinstance(artifact, dict):
                increment = max(len(artifact), 1)
            intermediate['search_count'] = intermediate.get('search_count', 0) + increment

            # NEW: Track query for deduplication
            query = tool_args.get('query', '')
            if query:
                normalized = normalize_query(query)
                intermediate['attempted_queries'].add(normalized)
                intermediate['last_N_queries'].append(normalized)
                # Keep only last 10 queries
                intermediate['last_N_queries'] = intermediate['last_N_queries'][-10:]

                # Track query effectiveness (results returned)
                result_count = 0
                if hasattr(tool_message, 'content'):
                    try:
                        content = tool_message.content
                        if isinstance(content, str):
                            import json
                            results = json.loads(content)
                            result_count = len(results) if isinstance(results, list) else 0
                        elif isinstance(content, list):
                            result_count = len(content)
                    except Exception:
                        pass
                intermediate['query_results'][normalized] = result_count

            return

        if tool_name != "extract_evidence_tool":
            return

        # Track extraction attempts (for loop detection)
        extraction_url = tool_args.get('search_result', {}).get('url', 'unknown')
        attempted_urls = intermediate.setdefault('attempted_urls', {})
        attempted_urls[extraction_url] = attempted_urls.get(extraction_url, 0) + 1

        # NEW: Check if URL is blocked before extraction
        blocked_urls = intermediate.get('blocked_urls', set())
        if extraction_url in blocked_urls:
            self.logger.debug(
                f"🚫 Skipping blocked URL: {extraction_url[:80]}"
            )
            return

        evidence = self._coerce_extract_evidence(tool_message)

        # Track failed extractions
        if evidence is None:
            failed_extractions = intermediate.setdefault('failed_extractions', [])
            failed_extractions.append(extraction_url)
            intermediate['consecutive_failed_searches'] += 1

            self.logger.debug(
                f"⚠️  Failed extraction #{len(failed_extractions)} from: {extraction_url[:80]}"
            )

            # NEW: Block URL after 2 failed attempts (not 3)
            if attempted_urls[extraction_url] >= 2:
                self.logger.error(
                    f"🚫 URL BLOCKED after {attempted_urls[extraction_url]} failed attempts: "
                    f"{extraction_url[:80]}"
                )
                blocked_urls.add(extraction_url)

            # Warn if same URL attempted multiple times
            elif attempted_urls[extraction_url] >= 1:
                self.logger.debug(
                    f"🔁 URL attempted {attempted_urls[extraction_url]} times - "
                    "will block after 1 more failure"
                )
            return

        # SUCCESS: Reset failure counter and update success rate
        intermediate['failed_extractions'] = []
        intermediate['consecutive_failed_searches'] = 0

        # Update extraction success rate (exponential moving average)
        current_rate = intermediate.get('extraction_success_rate', 1.0)
        intermediate['extraction_success_rate'] = 0.7 * current_rate + 0.3 * 1.0

        evidence_items = intermediate.setdefault('evidence_items', [])

        # Skip duplicates based on URL
        new_url = getattr(evidence, 'url', '') or ''
        for existing in evidence_items:
            existing_url = ''
            if isinstance(existing, ExtractedEvidence):
                existing_url = existing.url
            elif isinstance(existing, dict):
                existing_url = existing.get('url', '')
            elif hasattr(existing, 'url'):
                existing_url = getattr(existing, 'url')

            if existing_url and new_url and existing_url == new_url:
                self.logger.debug("♻️  Duplicate evidence detected, skipping auto-store")
                return

        evidence_items.append(evidence)
        self.logger.info(
            f"📥 Evidence stored automatically ({len(evidence_items)} total items)"
        )

    def _coerce_extract_evidence(self, tool_message) -> Optional[ExtractedEvidence]:
        """
        Convert a tool message payload into an ExtractedEvidence object.

        Supports LangGraph tool artifacts, dict payloads, and string fallbacks.
        """
        artifact = getattr(tool_message, "artifact", None)
        if artifact is None:
            artifact = getattr(tool_message, "additional_kwargs", {}).get("return_value")

        try:
            if isinstance(artifact, ExtractedEvidence):
                return artifact
            if isinstance(artifact, dict):
                return ExtractedEvidence(**artifact)
            if hasattr(artifact, "model_dump"):
                return ExtractedEvidence(**artifact.model_dump())
        except Exception as exc:
            self.logger.warning(f"Failed to parse artifact as ExtractedEvidence: {exc}")

        # Fallback to parsing the tool message content
        text_payload = self._message_text(tool_message)
        if not text_payload:
            return None

        # Try JSON first
        try:
            data = json.loads(text_payload)
            return ExtractedEvidence(**data)
        except Exception:
            pass

        # Parse key=value pairs (repr-style) as a last resort
        # Match repr-style key=value pairs while allowing quoted strings and trimming delimiters
        kv_pairs = re.findall(
            r'(\w+)=(".*?"|\'.*?\'|[^\s,]+)',
            text_payload
        )
        if not kv_pairs:
            return None

        parsed: Dict[str, Any] = {}
        for key, raw_value in kv_pairs:
            value = raw_value.strip().strip('"\'')
            # Drop trailing delimiters that often appear in repr strings
            if value.endswith((',', ')')):
                value = value.rstrip(',)')
            parsed[key] = value

        for float_key in (
            'verifiability_score',
            'independence_score',
            'recency_score',
            'estimated_LLR'
        ):
            if float_key in parsed:
                try:
                    if isinstance(parsed[float_key], str):
                        parsed[float_key] = parsed[float_key].rstrip(',)')
                    parsed[float_key] = float(parsed[float_key])
                except ValueError:
                    pass

        if 'support' in parsed:
            parsed['support'] = parsed['support'].lower()

        try:
            return ExtractedEvidence(**parsed)
        except Exception as exc:
            self.logger.warning(f"Failed to coerce evidence from text payload: {exc}")
            return None

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Determine if research task is complete with enhanced circuit breakers.

        Enhanced Criteria:
        1. Have we extracted sufficient evidence items? (min_evidence_items)
        2. Have we hit max search attempts?
        3. **NEW**: Query loop detection (3 identical queries → stop)
        4. **NEW**: Zero results detection (3 searches with 0 results → stop)
        5. **NEW**: Low success rate (< 20% after 5 attempts → stop)
        6. **NEW**: Consecutive failures (5+ failed extractions → stop)
        7. **NEW**: URL blocking threshold (5+ URLs blocked → consider stopping)
        8. Diminishing returns (existing check)

        Args:
            state: Current agent state

        Returns:
            True if task is complete
        """
        intermediate = state.get('intermediate_results', {})
        evidence_items = intermediate.get('evidence_items', [])
        evidence_count = len(evidence_items)

        # Check search attempts
        search_count = intermediate.get('search_count', 0)

        # Track evidence growth over iterations
        iteration = state.get('iteration_count', 0)

        # Initialize memory tracking
        self._initialize_memory_tracking(state)

        # DIAGNOSTIC: Log search and extraction stats (enhanced)
        failed_count = len(intermediate.get('failed_extractions', []))
        attempted_urls_count = len(intermediate.get('attempted_urls', {}))
        blocked_urls_count = len(intermediate.get('blocked_urls', set()))
        extraction_success_rate = intermediate.get('extraction_success_rate', 1.0)
        unique_queries = len(intermediate.get('attempted_queries', set()))

        self.logger.info(
            f"📊 Stats: {evidence_count} evidence, {search_count} searches ({unique_queries} unique), "
            f"{failed_count} failed extractions, {attempted_urls_count} URLs attempted, "
            f"{blocked_urls_count} blocked, success_rate={extraction_success_rate:.0%}"
        )

        # CIRCUIT BREAKER 1: Success - enough evidence gathered
        if evidence_count >= self.min_evidence_items:
            self.logger.info(
                f"✅ Task complete: {evidence_count} evidence items gathered "
                f"(minimum: {self.min_evidence_items})"
            )
            return True

        # CIRCUIT BREAKER 2: Query Loop Detection (NEW)
        last_queries = intermediate.get('last_N_queries', [])
        if len(last_queries) >= 3:
            # Check if last 3 queries are identical
            recent_3 = last_queries[-3:]
            if len(set(recent_3)) == 1:  # All identical
                self.logger.error(
                    f"🔁 QUERY LOOP DETECTED: Same query '{recent_3[0][:50]}' repeated 3x"
                )
                self.logger.warning("   → Stopping to prevent infinite loop")
                return True

            # Check if last 3 queries are semantically similar (>80% token overlap)
            if len(recent_3) == 3:
                words_1 = set(recent_3[0].split())
                words_2 = set(recent_3[1].split())
                words_3 = set(recent_3[2].split())
                all_words = words_1 | words_2 | words_3
                if all_words:
                    overlap = len(words_1 & words_2 & words_3) / len(all_words)
                    if overlap > 0.8:
                        self.logger.warning(
                            f"⚠️  QUERY SIMILARITY: Last 3 queries are {overlap:.0%} similar"
                        )
                        self.logger.warning("   → Triggering alternative search strategy")
                        # Set flag for adaptive query generation
                        intermediate['force_new_strategy'] = True

        # CIRCUIT BREAKER 3: Search Effectiveness - Zero Results (NEW)
        query_results = intermediate.get('query_results', {})
        if len(query_results) >= 3:
            recent_queries_list = list(intermediate.get('last_N_queries', []))[-3:]
            recent_result_counts = [query_results.get(q, 0) for q in recent_queries_list]

            if sum(recent_result_counts) == 0:
                self.logger.error(
                    f"🔍 ZERO RESULTS: Last 3 searches returned no relevant results"
                )
                self.logger.info(
                    f"   → Accepting {evidence_count} items (search space exhausted)"
                )
                return True

        # CIRCUIT BREAKER 4: Extraction Success Rate (NEW)
        if attempted_urls_count >= 5 and extraction_success_rate < 0.2:
            self.logger.error(
                f"📉 LOW SUCCESS RATE: Only {extraction_success_rate:.0%} of extractions successful "
                f"after {attempted_urls_count} attempts"
            )
            self.logger.info(
                f"   → Accepting {evidence_count} items (sources not yielding evidence)"
            )
            return True

        # CIRCUIT BREAKER 5: URL Blocking Threshold (NEW)
        if blocked_urls_count >= 5:
            self.logger.warning(
                f"🚫 {blocked_urls_count} URLs blocked due to repeated failures"
            )
            if evidence_count >= 3:
                self.logger.info(
                    f"   → Accepting {evidence_count} items (many sources blocked)"
                )
                return True

        # CIRCUIT BREAKER 6: Consecutive Failed Searches (NEW)
        consecutive_failed = intermediate.get('consecutive_failed_searches', 0)
        if consecutive_failed >= 5:
            self.logger.error(
                f"❌ {consecutive_failed} consecutive failed extractions"
            )
            self.logger.info(
                f"   → Accepting {evidence_count} items (extraction repeatedly failing)"
            )
            return True

        # NEW: If agent has 3+ evidence items AND stopped making tool calls,
        # accept it as complete (agent has decided search is exhausted)
        if evidence_count >= 3:
            # Check if agent made tool calls in last message
            last_message = state.get('messages', [])[-1] if state.get('messages') else None
            has_tool_calls = (last_message and
                            hasattr(last_message, 'tool_calls') and
                            last_message.tool_calls)

            if not has_tool_calls and iteration >= 3:
                self.logger.info(
                    f"✅ Task complete: {evidence_count} evidence items "
                    f"(agent has stopped searching - accepting as sufficient)"
                )
                return True

        # Complete if we've hit max search attempts (even if not enough evidence)
        if search_count >= self.max_search_attempts:
            self.logger.warning(
                f"⚠️  Task complete: Max search attempts ({self.max_search_attempts}) reached "
                f"with only {evidence_count} evidence items"
            )
            return True

        # NEW: Detect diminishing returns - if we've done 4+ searches but only have 0-2 items
        # AND we're past iteration 6, it's unlikely more searches will help
        if search_count >= 4 and evidence_count <= 2 and iteration >= 6:
            self.logger.warning(
                f"⚠️  Task complete: Diminishing returns detected - "
                f"{search_count} searches yielded only {evidence_count} items after {iteration} iterations"
            )
            self.logger.info(
                f"   → Absence of evidence is meaningful - stopping search"
            )
            return True

        # NEW: If we have SOME evidence (3-4 items) and have searched extensively (6+ searches),
        # that's probably sufficient even if below minimum
        if evidence_count >= 3 and search_count >= 6 and iteration >= 8:
            self.logger.info(
                f"✅ Task complete: {evidence_count} evidence items after extensive search "
                f"({search_count} searches, {iteration} iterations)"
            )
            return True

        # NEW: Check for unproductive extraction loops
        failed_extractions = intermediate.get('failed_extractions', [])
        if len(failed_extractions) >= 5:
            self.logger.warning(
                f"⚠️  Task complete: {len(failed_extractions)} consecutive failed extractions - "
                f"search yielding no relevant results"
            )
            self.logger.info(
                f"   → Accepting {evidence_count} items (absence of evidence is meaningful)"
            )
            return True

        # NEW: Track last evidence found - if no new evidence in last 4 iterations, stop
        last_evidence_count = intermediate.get('_last_evidence_count', 0)
        if iteration > 0 and evidence_count == last_evidence_count:
            stagnant_iterations = intermediate.get('_stagnant_iterations', 0) + 1
            intermediate['_stagnant_iterations'] = stagnant_iterations

            if stagnant_iterations >= 3 and iteration >= 5:
                self.logger.warning(
                    f"⚠️  Task complete: No new evidence in {stagnant_iterations} iterations "
                    f"({evidence_count} items total - search exhausted)"
                )
                return True
        else:
            # Reset stagnant counter when evidence grows
            intermediate['_stagnant_iterations'] = 0

        # Update tracking
        intermediate['_last_evidence_count'] = evidence_count

        # CRITICAL: If we're past iteration 15, force completion regardless
        # This prevents infinite loops when safety checks fail
        if iteration >= 15:
            self.logger.error(
                f"🛑 FORCE STOP: Iteration {iteration} reached - "
                f"accepting {evidence_count} items to prevent infinite loop"
            )
            return True

        # Otherwise, continue
        self.logger.info(
            f"🔄 Continue: {evidence_count}/{self.min_evidence_items} evidence, "
            f"{search_count}/{self.max_search_attempts} searches, "
            f"iteration {iteration}"
        )
        return False

    async def extract_final_output(self, state: AgentState) -> ResearcherOutput:
        """
        Extract ResearcherOutput from final agent state

        Args:
            state: Final agent state

        Returns:
            ResearcherOutput with all gathered evidence
        """
        # Get evidence items from intermediate results
        evidence_items = state.get('intermediate_results', {}).get('evidence_items', [])

        # Convert to Evidence objects if they're dicts or ExtractedEvidence objects
        evidence_list = []
        for item in evidence_items:
            if isinstance(item, dict):
                # Parse dict into Evidence model
                evidence_list.append(Evidence(**item))
            elif isinstance(item, Evidence):
                evidence_list.append(item)
            elif isinstance(item, ExtractedEvidence):  # ExtractedEvidence object
                # Convert ExtractedEvidence to Evidence
                try:
                    # Parse date string to date object
                    date_obj = None
                    if hasattr(item, 'published_date') and item.published_date != "unknown":
                        try:
                            from datetime import datetime
                            date_obj = datetime.strptime(item.published_date, "%Y-%m-%d").date()
                        except ValueError:
                            # Use a default date if parsing fails
                            from datetime import date
                            date_obj = date.today()
                    else:
                        # Use today's date as default for unknown dates
                        from datetime import date
                        date_obj = date.today()

                    evidence_dict = {
                        'subclaim_id': item.subclaim_id,
                        'title': item.title,
                        'url': item.url,
                        'published_date': date_obj,
                        'source_type': item.source_type,
                        'claim_summary': item.claim_summary,
                        'support': item.support,
                        'verifiability_score': item.verifiability_score,
                        'independence_score': item.independence_score,
                        'recency_score': item.recency_score,
                        'estimated_LLR': item.estimated_LLR,
                        'extraction_notes': item.extraction_notes
                    }
                    evidence_list.append(Evidence(**evidence_dict))
                except Exception as e:
                    self.logger.warning(f"Failed to convert evidence item: {e}")
                    continue

        pro_count = sum(1 for e in evidence_list if e.support == 'pro')
        con_count = sum(1 for e in evidence_list if e.support == 'con')
        neutral_count = len(evidence_list) - pro_count - con_count

        total_pro_llr = sum(
            e.estimated_LLR for e in evidence_list
            if e.support == 'pro' and e.estimated_LLR > 0
        )
        total_con_llr = sum(
            abs(e.estimated_LLR) for e in evidence_list
            if e.support == 'con' and e.estimated_LLR < 0
        )
        net_llr = sum(e.estimated_LLR for e in evidence_list)

        subclaims_data = state.get('task_input', {}).get('subclaims', [])
        subclaim_direction_map = {
            sc.get('id'): sc.get('direction')
            for sc in subclaims_data
            if sc.get('id') and sc.get('direction')
        }
        directional_items = 0
        aligned_items = 0
        for ev in evidence_list:
            expected_direction = subclaim_direction_map.get(ev.subclaim_id)
            if expected_direction not in {'pro', 'con'}:
                continue
            if ev.support == 'neutral':
                continue
            directional_items += 1
            if ev.support == expected_direction:
                aligned_items += 1

        context_alignment_score = (
            aligned_items / directional_items if directional_items else 0.0
        )

        self.logger.info(
            f"📤 Final output: {len(evidence_list)} evidence items "
            f"(pro={pro_count}, con={con_count}, neutral={neutral_count}), "
            f"net LLR={net_llr:+.2f}, context_alignment={context_alignment_score:.2f}"
        )

        search_strategy = (
            f"{self.direction.upper()} search captured {len(evidence_list)} items "
            f"(pro={pro_count}, con={con_count}, neutral={neutral_count}); "
            f"net LLR {net_llr:+.2f}"
        )

        return ResearcherOutput(
            evidence_items=evidence_list,
            total_pro_count=pro_count,
            total_con_count=con_count,
            total_pro_llr=total_pro_llr,
            total_con_llr=total_con_llr,
            net_llr=net_llr,
            context_alignment_score=context_alignment_score,
            search_strategy=search_strategy
        )

    async def run_research(
        self,
        search_seeds: List[str],
        subclaims: List[Dict[str, Any]],
        market_question: str,
        **kwargs
    ) -> ResearcherOutput:
        """
        Convenience method matching old ResearcherAgent interface

        Args:
            search_seeds: Search queries to execute
            subclaims: Subclaims to find evidence for
            market_question: Main market question

        Returns:
            ResearcherOutput with gathered evidence
        """
        return await self.run(
            task_description=f"Gather {self.direction.upper()} evidence for market question",
            task_input={
                'search_seeds': search_seeds,
                'subclaims': subclaims,
                'market_question': market_question,
                **kwargs
            }
        )


# Convenience function for parallel execution
async def run_parallel_autonomous_research(
    search_seeds_pro: List[str],
    search_seeds_con: List[str],
    search_seeds_general: List[str],
    subclaims: List[Dict[str, Any]],
    market_question: str,
    **kwargs
) -> Dict[str, ResearcherOutput]:
    """
    Run PRO, CON, and GENERAL autonomous researchers in parallel

    Args:
        search_seeds_pro: PRO search queries
        search_seeds_con: CON search queries
        search_seeds_general: GENERAL search queries
        subclaims: Subclaim list
        market_question: Main question
        **kwargs: Additional args for researchers

    Returns:
        Dict with 'pro', 'con', 'general' ResearcherOutput objects
    """
    import asyncio

    # Create autonomous agents
    researcher_pro = AutonomousResearcherAgent(direction="pro", **kwargs)
    researcher_con = AutonomousResearcherAgent(direction="con", **kwargs)
    researcher_general = AutonomousResearcherAgent(direction="general", **kwargs)

    # Execute in parallel
    results = await asyncio.gather(
        researcher_pro.run_research(search_seeds_pro, subclaims, market_question),
        researcher_con.run_research(search_seeds_con, subclaims, market_question),
        researcher_general.run_research(search_seeds_general, subclaims, market_question),
        return_exceptions=True
    )

    # Handle results
    output = {}
    for direction, result in zip(["pro", "con", "general"], results):
        if isinstance(result, Exception):
            logger.error(f"{direction.upper()} autonomous research failed: {result}")
            output[direction] = ResearcherOutput(evidence_items=[])
        else:
            output[direction] = result

    return output
