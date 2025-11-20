"""
Autonomous PlannerAgent with Memory and Validation
Decomposes market questions into a research plan (prior, subclaims, search seeds).
"""
from __future__ import annotations

import json
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool

from agents.schemas import Subclaim, SearchSeeds  # keep public dependency local
from agents.base import AutonomousReActAgent, AgentState
from agents.schemas import PlannerOutput
from tools.memory import search_similar_markets_tool, get_base_rates_tool

logger = logging.getLogger(__name__)


class AutonomousPlannerAgent(AutonomousReActAgent):
    """
    Planner agent that creates a concise, balanced plan:
      • Prior estimate + justification
      • {min_subclaims}–{max_subclaims} researchable subclaims (balanced PRO/CON)
      • Targeted search seeds (pro/con/general)

    Notes:
      - Memory tools are optional and used only if a store is configured.
      - The agent emits a FINAL_PLAN_JSON block which is parsed and validated.
    """

    def __init__(self, min_subclaims: int = 4, max_subclaims: int = 10, **kwargs):
        """
        Args:
            min_subclaims: Minimum subclaims to generate (>=1).
            max_subclaims: Maximum subclaims to generate (>=min_subclaims).
        """
        assert isinstance(min_subclaims, int) and min_subclaims >= 1, "min_subclaims must be >= 1"
        assert isinstance(max_subclaims, int) and max_subclaims >= min_subclaims, "max_subclaims must be >= min_subclaims"
        super().__init__(**kwargs)
        self.min_subclaims = min_subclaims
        self.max_subclaims = max_subclaims

    def get_system_prompt(self) -> str:
        """Return system prompt guiding the planner to produce a fast, balanced plan."""
        now = datetime.now()
        current_date = now.strftime("%B %d, %Y")
        current_year = now.year
        return f"""You are the Planner Agent in POLYSEER. Create a concise research plan.

**IMPORTANT: Today's date is {current_date}. Use ONLY current year ({current_year}) in search queries. Focus on recent 2025 data and trends unless earlier years are directly relevant.**

## Task
Produce:
1) A simple prior probability (p0_prior) with brief justification
2) {self.min_subclaims}-{self.max_subclaims} balanced, researchable subclaims (equal PRO/CON where possible)
3) 3–5 targeted search seeds for each direction: pro, con, general

## Workflow
- Understand the market question and timeframe
- (Optional) get_base_rates_tool(event_category) to inform prior
- (Optional) search_similar_markets_tool(query) for analogous cases
- Draft p0_prior in [0.1, 0.9] with a 1–2 sentence justification
- Generate specific, falsifiable subclaims (balanced)
- Create diverse, high-quality search seeds
  - Use terms like "2025", "recent", "current", "latest"
  - Avoid outdated years unless needed for context

## Output (exact format)
FINAL_PLAN_JSON:
{{
"market_slug": "clean-identifier-from-question",
"market_question": "Original question text",
"p0_prior": 0.XX,
"prior_justification": "1–2 sentence reasoning",
"subclaims": [
{{"id": "sc_1", "text": "Pro claim", "direction": "pro"}},
{{"id": "sc_2", "text": "Con claim", "direction": "con"}}
],
"key_variables": ["Factor A", "Factor B"],
"search_seeds": {{
"pro": ["2025 recent evidence", "latest trend query"],
"con": ["2025 counter-evidence", "current challenges"],
"general": ["2025 context", "recent background"]
}},
"decision_criteria": ["Criterion 1", "Criterion 2"],
"reasoning_trace": "Brief summary"
}}

rust
Copy code

After emitting FINAL_PLAN_JSON, stop.
Quality: Prior reasonable (0.3–0.7 for high uncertainty), subclaims balanced & researchable, seeds targeted to 2025 data.
"""

    def get_tools(self) -> List[BaseTool]:
        """Return optional memory tools if a store is configured."""
        return [t for t in [search_similar_markets_tool, get_base_rates_tool] if self.store]

    async def agent_node(self, state: AgentState) -> AgentState:
        """
        Run parent node once, then extract FINAL_PLAN_JSON from the last LLM message
        into state['intermediate_results'] if present and valid.
        """
        state = await super().agent_node(state)
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        content = getattr(last, "content", None) if last else None
        if not content:
            return state

        match = re.search(r"FINAL_PLAN_JSON:\s*(\{[\s\S]*?\})\s*(?:```|$)", content)
        if not match:
            return state

        plan_json = match.group(1).strip()
        assert plan_json.startswith("{") and plan_json.endswith("}"), "Plan JSON must be valid JSON object"
        plan = json.loads(plan_json)
        assert isinstance(plan, dict), "Plan must be dict"
        assert all(k in plan for k in ("p0_prior", "subclaims", "search_seeds")), "Plan missing required keys"

        results = state.get("intermediate_results", {})
        results.update(plan)
        state["intermediate_results"] = results
        self.logger.info("Planner: extracted FINAL_PLAN_JSON")
        return state

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Completion when:
          - p0_prior in [0.01, 0.99] and prior_justification present
          - >= min_subclaims subclaims and balanced PRO/CON (both >0)
          - search_seeds has >=3 items for pro, con, general
        """
        r = state.get("intermediate_results", {})
        prior = r.get("p0_prior")
        if prior is None or not (0.01 <= float(prior) <= 0.99):
            return False
        if not (isinstance(r.get("prior_justification"), str) and r["prior_justification"].strip()):
            return False

        subclaims = r.get("subclaims") or []
        if len(subclaims) < self.min_subclaims:
            return False
        pro = sum(sc.get("direction") == "pro" for sc in subclaims)
        con = sum(sc.get("direction") == "con" for sc in subclaims)
        if pro == 0 or con == 0:
            return False

        seeds = r.get("search_seeds") or {}
        if not all(isinstance(seeds.get(k), list) and len(seeds[k]) >= 2 for k in ("pro", "con", "general")):
            return False

        self.logger.info("Planner: planning complete")
        return True

    async def extract_final_output(self, state: AgentState) -> PlannerOutput:
        """
        Build PlannerOutput from validated intermediate_results.
        Raises if subclaims are missing the minimum count.
        """

        r = state.get("intermediate_results", {})
        subs = r.get("subclaims", [])
        if len(subs) < self.min_subclaims:
            raise ValueError(f"Planner requires >= {self.min_subclaims} subclaims (got {len(subs)}).")

        subclaims = [
            Subclaim(id=sc.get("id", f"sc_{i}"), text=sc.get("text", ""), direction=sc.get("direction", "pro"))
            for i, sc in enumerate(subs)
        ]
        seeds_dict = r.get("search_seeds", {})
        seeds = SearchSeeds(
            pro=seeds_dict.get("pro", []),
            con=seeds_dict.get("con", []),
            general=seeds_dict.get("general", []),
        )

        trace = r.get("reasoning_trace", "")
        if not trace:
            steps = state.get("reasoning_trace", [])
            trace = "\n".join(f"{i+1}. {s.thought}" for i, s in enumerate(steps)) if steps else "Reasoning trace unavailable."

        return PlannerOutput(
            market_slug=r.get("market_slug", ""),
            market_question=r.get("market_question", ""),
            p0_prior=r.get("p0_prior", 0.5),
            prior_justification=r.get("prior_justification", ""),
            subclaims=subclaims,
            key_variables=r.get("key_variables", []),
            search_seeds=seeds,
            decision_criteria=r.get("decision_criteria", []),
            reasoning_trace=trace,
        )

    async def plan(
        self,
        market_question: str,
        market_url: str = "",
        market_slug: str = "",
        context: Dict[str, Any] = None,
        *,
        max_iterations: Optional[int] = None,
    ) -> PlannerOutput:
        """
        Generate a research plan autonomously.

        Args:
            market_question: Prediction market question (required, non-empty).
            market_url: Optional URL.
            market_slug: Optional slug (defaults to a slugified question).
            context: Additional context dict.
            max_iterations: Optional override for reasoning loop steps.
        """
        assert isinstance(market_question, str) and market_question.strip(), "market_question is required"
        
        # Log memory context if available
        if self.store and self.enable_auto_memory_query:
            self.logger.info("Querying memory for similar markets and base rates before planning")
        
        return await self.run(
            task_description="Create comprehensive research plan for market question",
            task_input={
                "market_question": market_question,
                "market_url": market_url or "unknown",
                "market_slug": market_slug or market_question.lower().replace(" ", "-")[:50],
                **(context or {}),
            },
            max_iterations=max_iterations,
        )

# -------------------------
# Integrity & diagnostics
# -------------------------
def integrity_report(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact integrity report for planner outputs.
    Returns coverage and balance metrics to spot issues early.
    """
    r = (state or {}).get("intermediate_results", {}) if isinstance(state, dict) else {}
    subs = r.get("subclaims") or []
    pro = sum(sc.get("direction") == "pro" for sc in subs)
    con = sum(sc.get("direction") == "con" for sc in subs)
    seeds = r.get("search_seeds") or {}
    return {
        "has_prior": "p0_prior" in r,
        "has_justification": bool(r.get("prior_justification")),
        "subclaims_count": len(subs),
        "balance_ratio": (min(pro, con) / max(pro, con)) if max(pro, con) else 0.0,
        "seeds_coverage": {k: len(v) for k, v in seeds.items() if isinstance(v, list)},
        "passes_minimums": all(
            [
                "p0_prior" in r,
                isinstance(r.get("prior_justification"), str) and r["prior_justification"].strip(),
                len(subs) >= 1,
                pro > 0 and con > 0,
                all(isinstance(seeds.get(k), list) and len(seeds[k]) >= 3 for k in ("pro", "con", "general")),
            ]
        ),
    }