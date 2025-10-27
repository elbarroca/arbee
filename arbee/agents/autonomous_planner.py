"""
Autonomous PlannerAgent with Memory and Validation
First agent in POLYSEER workflow - decomposes market questions with autonomous reasoning
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import PlannerOutput
from arbee.tools.memory_search import search_similar_markets_tool, get_base_rates_tool

logger = logging.getLogger(__name__)


class AutonomousPlannerAgent(AutonomousReActAgent):
    """
    Autonomous Planner Agent - Decomposes market questions into research plans

    Autonomous Capabilities:
    - Searches memory for similar market analyses to inform planning
    - Queries base rates for prior probability estimation
    - Estimates reasonable prior without iterative validation
    - Generates balanced subclaims (pro/con)
    - Creates targeted search seeds for researchers

    Reasoning Flow:
    1. Understand market question
    2. Search for similar past analyses (memory) [optional]
    3. Get historical base rates for reference class [optional]
    4. Draft simple prior (0.1-0.9 range) with justification
    5. Generate subclaims (balanced pro/con)
    6. Create search seeds for each direction
    7. Output plan immediately
    """

    def __init__(self, min_subclaims: int = 4, max_subclaims: int = 10, **kwargs):
        """
        Initialize Autonomous Planner

        Args:
            min_subclaims: Minimum subclaims to generate
            max_subclaims: Maximum subclaims to generate
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(**kwargs)
        self.min_subclaims = min_subclaims
        self.max_subclaims = max_subclaims

    def get_system_prompt(self) -> str:
        """System prompt for autonomous planning - simple and fast"""
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")

        return f"""You are the Planner Agent in POLYSEER. Your job is to quickly create a research plan for prediction markets.

**IMPORTANT: Today's date is {current_date}. Use ONLY current year ({datetime.now().year}) in search queries. NEVER use 2023, 2024, or any past years - only focus on recent 2025 data and current trends!**

## 🎯 Your Task

Create a structured research plan with:
1. A simple prior probability estimate (p0_prior)
2. {self.min_subclaims}-{self.max_subclaims} balanced subclaims (PRO and CON)
3. 3-5 search seeds for each direction (pro, con, general)

## 📋 Workflow

**Step 1: Understand the Question**
- Parse the market question
- Identify key event and timeframe

**Step 2: Estimate Prior with Base Rates (Optional)**
- Optionally call get_base_rates_tool to find historical base rates
  - Pass: event_category (e.g., "celebrity athletic challenges")
  - Tool searches memory for stored base rates
  - Returns: base_rate, confidence, sources
- If base rates available, use them to inform your p0_prior
- If no base rates found, use neutral prior (0.5) or reason from first principles
- Document your reasoning in prior_justification field

**Step 3: Generate Balanced Subclaims**
- Create {self.min_subclaims}-{self.max_subclaims} specific, researchable subclaims
- **Balance is CRITICAL**: Aim for equal PRO and CON subclaims
- Each subclaim should be:
  - Specific and falsifiable
  - Researchable (evidence can be found)
  - Relevant to the market question

**Step 4: Create Search Seeds**
- Generate 3-5 search queries for EACH direction (pro, con, general)
- Make them specific, diverse, and targeted
- Think about what sources would have quality evidence
-- **CRITICAL**: All search queries must focus on 2025 data only ( or previous years if they are relevant to 2025)
-- Use phrases like "2025", "recent", "current", "latest" - never specify past years

**Step 5: Output Your Plan**
Output your complete plan in this EXACT format:

```
FINAL_PLAN_JSON:
{{
  "market_slug": "clean-identifier-from-question",
  "market_question": "Original question text",
  "p0_prior": 0.XX,
  "prior_justification": "1-2 sentence reasoning for your prior estimate",
  "subclaims": [
    {{"id": "sc_1", "text": "Specific falsifiable claim supporting YES", "direction": "pro"}},
    {{"id": "sc_2", "text": "Specific falsifiable claim supporting NO", "direction": "con"}},
    {{"id": "sc_3", "text": "Another pro claim", "direction": "pro"}},
    {{"id": "sc_4", "text": "Another con claim", "direction": "con"}}
  ],
  "key_variables": ["Key Factor 1", "Key Factor 2", "Key Factor 3"],
  "search_seeds": {{
    "pro": ["2025 recent evidence query", "current trend query", "latest development search"],
    "con": ["2025 recent counter-evidence", "current challenges query", "latest opposing trends"],
    "general": ["2025 context query", "recent background research", "current market analysis"]
  }},
  "decision_criteria": ["Criterion 1", "Criterion 2"],
  "reasoning_trace": "Brief summary of your reasoning process"
}}
```

**IMPORTANT**:
- After outputting FINAL_PLAN_JSON, you are DONE - do not continue reasoning
- Focus on QUALITY subclaims and search seeds
- Your plan will be automatically validated after you output it

## 🔍 Available Tools

**search_similar_markets_tool**(query) - Find similar past analyses (optional)
**get_base_rates_tool**(category) - Get historical base rates from memory (optional)

All tools are OPTIONAL - only use if helpful for informing your prior estimate.

## ✅ Quality Standards

- **Prior**: Simple, reasonable (0.3-0.7 for uncertain events)
- **Subclaims**: Specific, balanced (equal PRO/CON), researchable
- **Search seeds**: Targeted, diverse, high-quality sources likely, focused on 2025 data only

Remember: Your job is to set up a GOOD research plan, not to do the research yourself. The researcher agents will do the deep work.
"""

    def get_tools(self) -> List[BaseTool]:
        """Return planning tools (optional memory tools)"""
        tools = []

        # Add memory tools if store configured (both are optional)
        if self.store:
            tools.extend([
                search_similar_markets_tool,
                get_base_rates_tool,
            ])

        # Note: Validation happens automatically in is_task_complete(), not via tool
        return tools

    async def agent_node(self, state: AgentState) -> AgentState:
        """
        Override agent_node to extract plan JSON from LLM responses.

        After calling the parent agent_node, this extracts FINAL_PLAN_JSON
        from the LLM response and populates intermediate_results.
        """
        import json
        import re

        # Call parent agent_node FIRST
        state = await super().agent_node(state)

        # Extract JSON plan from the last LLM response
        if state.get('messages'):
            last_msg = state['messages'][-1]
            if hasattr(last_msg, 'content') and last_msg.content:
                response_text = last_msg.content

                # Look for FINAL_PLAN_JSON marker
                json_match = re.search(
                    r'FINAL_PLAN_JSON:\s*(\{[\s\S]*?\})\s*(?:```|$)',
                    response_text
                )

                if json_match:
                    try:
                        plan_json = json.loads(json_match.group(1))

                        # Validate plan structure
                        required_keys = {'p0_prior', 'subclaims', 'search_seeds'}
                        if all(k in plan_json for k in required_keys):
                            # Extract and populate intermediate_results
                            results = state.get('intermediate_results', {})
                            results.update(plan_json)
                            state['intermediate_results'] = results

                            self.logger.info("✅ Extracted plan JSON from LLM response")
                            self.logger.info(f"   Prior: {plan_json.get('p0_prior')}")
                            self.logger.info(f"   Subclaims: {len(plan_json.get('subclaims', []))}")
                            self.logger.info(f"   Search seeds: {len(plan_json.get('search_seeds', {}).get('pro', []))}/{len(plan_json.get('search_seeds', {}).get('con', []))}/{len(plan_json.get('search_seeds', {}).get('general', []))}")
                    except (json.JSONDecodeError, ValueError) as e:
                        self.logger.warning(f"Failed to parse plan JSON: {e}")

        return state

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Check if planning is complete

        Criteria:
        - Has valid prior (0.01-0.99)
        - Has prior justification
        - Has minimum subclaims
        - Has balanced pro/con subclaims
        - Has search seeds for all directions
        """
        results = state.get('intermediate_results', {})

        # Check if we have a plan
        if 'p0_prior' not in results:
            return False

        # Check if prior justification exists
        if 'prior_justification' not in results:
            self.logger.info("Need prior justification")
            return False

        # Check prior range
        prior = results.get('p0_prior', 0.5)
        if not (0.01 <= prior <= 0.99):
            self.logger.warning(f"Prior {prior} outside valid range")
            return False

        # Check subclaims
        subclaims = results.get('subclaims', [])
        if len(subclaims) < self.min_subclaims:
            self.logger.info(f"Need more subclaims: {len(subclaims)}/{self.min_subclaims}")
            return False

        # Check balance
        pro_count = sum(1 for sc in subclaims if sc.get('direction') == 'pro')
        con_count = sum(1 for sc in subclaims if sc.get('direction') == 'con')

        if pro_count == 0 or con_count == 0:
            self.logger.warning("Subclaims not balanced")
            return False

        # Check search seeds
        search_seeds = results.get('search_seeds', {})
        if not all(key in search_seeds and len(search_seeds[key]) >= 3
                   for key in ['pro', 'con', 'general']):
            self.logger.info("Search seeds incomplete")
            return False

        self.logger.info("✅ Planning complete - all criteria met")
        return True

    async def extract_final_output(self, state: AgentState) -> PlannerOutput:
        """Extract PlannerOutput from final state"""
        results = state.get('intermediate_results', {})

        # Build PlannerOutput from results
        from arbee.agents.schemas import Subclaim, SearchSeeds

        if len(results.get('subclaims', [])) < self.min_subclaims:
            raise ValueError(
                f"Planner intermediate results missing required subclaims "
                f"({len(results.get('subclaims', []))}/{self.min_subclaims})."
            )

        subclaims = [
            Subclaim(
                id=sc.get('id', f"sc_{i}"),
                text=sc.get('text', ''),
                direction=sc.get('direction', 'pro')
            )
            for i, sc in enumerate(results.get('subclaims', []))
        ]

        search_seeds_data = results.get('search_seeds', {})
        search_seeds = SearchSeeds(
            pro=search_seeds_data.get('pro', []),
            con=search_seeds_data.get('con', []),
            general=search_seeds_data.get('general', [])
        )

        reasoning_trace_text = results.get('reasoning_trace', '')
        if not reasoning_trace_text:
            trace_steps = state.get('reasoning_trace', [])
            if trace_steps:
                reasoning_trace_text = "\n".join(
                    f"{idx + 1}. {step.thought}"
                    for idx, step in enumerate(trace_steps)
                )

        if not reasoning_trace_text:
            reasoning_trace_text = "Reasoning trace unavailable."

        output = PlannerOutput(
            market_slug=results.get('market_slug', ''),
            market_question=results.get('market_question', ''),
            p0_prior=results.get('p0_prior', 0.5),
            prior_justification=results.get('prior_justification', ''),
            subclaims=subclaims,
            key_variables=results.get('key_variables', []),
            search_seeds=search_seeds,
            decision_criteria=results.get('decision_criteria', []),
            reasoning_trace=reasoning_trace_text
        )

        self.logger.info(
            f"📤 Plan complete: {len(subclaims)} subclaims, prior={output.p0_prior:.2%}"
        )

        return output

    async def plan(
        self,
        market_question: str,
        market_url: str = "",
        market_slug: str = "",
        context: Dict[str, Any] = None,
        *,
        max_iterations: Optional[int] = None
    ) -> PlannerOutput:
        """
        Generate research plan autonomously

        Args:
            market_question: Prediction market question
            market_url: Optional URL
            market_slug: Optional slug
            context: Additional context
            max_iterations: Optional override for reasoning loop iteration budget

        Returns:
            PlannerOutput with complete research plan
        """
        return await self.run(
            task_description="Create comprehensive research plan for market question",
            task_input={
                'market_question': market_question,
                'market_url': market_url or 'unknown',
                'market_slug': market_slug or market_question.lower().replace(' ', '-')[:50],
                **(context or {})
            },
            max_iterations=max_iterations
        )
