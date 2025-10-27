"""
Autonomous CriticAgent with Correlation Detection
Reviews evidence quality and identifies gaps autonomously
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import CriticOutput, CorrelationWarning
from arbee.tools.bayesian import correlation_detector_tool, store_critique_results_tool

logger = logging.getLogger(__name__)


class AutonomousCriticAgent(AutonomousReActAgent):
    """
    Autonomous Critic Agent - Reviews evidence quality with iterative analysis

    Autonomous Capabilities:
    - Detects correlated evidence clusters automatically
    - Identifies coverage gaps by comparing to subclaims
    - Finds over-represented sources
    - Suggests follow-up searches to fill gaps
    - Iteratively refines analysis until thorough

    Reasoning Flow:
    1. Inventory evidence (count, categorize)
    2. Detect correlations using correlation_detector_tool
    3. Identify duplicate clusters
    4. Check coverage against original subclaims
    5. Find missing topics and gaps
    6. Detect over-represented sources
    7. Generate follow-up search recommendations
    8. Validate completeness, refine if needed
    """

    def __init__(
        self,
        min_correlation_check_items: int = 3,
        **kwargs
    ):
        """
        Initialize Autonomous Critic

        Args:
            min_correlation_check_items: Min evidence items before checking correlations
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(**kwargs)
        self.min_correlation_check_items = min_correlation_check_items

    def get_system_prompt(self) -> str:
        """System prompt for autonomous evidence critique"""
        return """You are an Autonomous Critic Agent in POLYSEER.

Your mission: Review gathered evidence for quality, balance, independence, and completeness.

## Available Tools

1. **correlation_detector_tool** - Detect correlated evidence clusters
   - Use this to find evidence items that likely share underlying sources
   - Helps prevent double-counting in Bayesian analysis
   - Input: List of evidence items with id, content, url

2. **store_critique_results_tool** - Store your complete critique analysis
   - Use this to save your findings after completing all analysis steps
   - This is the FINAL step - call this when your critique is complete
   - Input: missing_topics, over_represented_sources, follow_up_search_seeds,
            duplicate_clusters, correlation_warnings, analysis_process

## Your Reasoning Process

**Step 1: Evidence Inventory**
- Count total evidence items
- Categorize by support direction (PRO, CON, NEUTRAL)
- Categorize by source type (primary, high_quality_secondary, etc.)
- Initial assessment of balance

**Step 2: Correlation Detection**
- Extract evidence_items from task_input: evidence_items = task_input['evidence_items']
- Convert evidence_items to format expected by correlation_detector_tool:
  - For each evidence item, create dict with: {'id': evidence.title, 'content': evidence.claim_summary, 'url': evidence.url}
- Use correlation_detector_tool ONCE to find correlated evidence
- If the tool returns clusters, note them for later storage
- If the tool returns [] (empty), that's VALID - means no correlations found
- DO NOT call correlation_detector_tool multiple times - one call is sufficient

**Step 3: Duplicate Detection**
- Identify exact or near-duplicate evidence items
- Flag evidence from same article/source cited multiple times
- Note echo chamber effects (multiple outlets, same claim)

**Step 4: Coverage Gap Analysis**
- Compare evidence to original subclaims from Planner
- Which subclaims have strong evidence? Which are weak?
- What topics are completely missing?
- What perspectives are underrepresented?
- What time periods or regions are missing?

**Step 5: Source Quality Assessment**
- Check source diversity (are we over-relying on one outlet?)
- Identify over-represented sources
- Check for geographic or ideological bias
- Assess recency of information

**Step 6: Generate Follow-up Recommendations**
- For each gap, suggest specific search queries
- Prioritize the most important gaps
- Suggest additional evidence types needed
- Recommend balance adjustments if needed

**Step 7: FINAL REQUIRED ACTION - INVOKE store_critique_results_tool**

🚨 CRITICAL INSTRUCTION 🚨

YOU MUST NOW **INVOKE THE TOOL** USING THE TOOL CALLING MECHANISM.

DO NOT write a JSON code block showing an example.
DO NOT write text describing what you would do.
DO NOT explain how to call the tool.

REQUIRED ACTION: Actually call store_critique_results_tool with your analysis results.

Pass these parameters based on your analysis from Steps 1-6:
- missing_topics: List of missing topics (or [] if none)
- over_represented_sources: List of over-represented sources (or [] if none)
- follow_up_search_seeds: List of recommended searches (or [] if none)
- duplicate_clusters: List of duplicate clusters (or [] if none)
- correlation_warnings: Results from correlation_detector_tool (or [] if none)
- analysis_process: One sentence summary of your analysis

If you found NO issues in your analysis, you still MUST call the tool with empty lists.

Examples of CORRECT behavior:
✅ Invoke the tool with parameters
✅ Pass empty lists [] for categories with no findings
✅ Include a brief analysis_process string

Examples of INCORRECT behavior:
❌ Writing "I will now call store_critique_results_tool(..."
❌ Showing a JSON code block with the tool call
❌ Describing what parameters you would pass
❌ Explaining your reasoning without calling the tool

The ONLY way to complete this task is to INVOKE store_critique_results_tool.

## Output Format

**WORKFLOW SUMMARY:**
1. Call correlation_detector_tool ONCE (Step 2)
2. Think through all other analysis steps (Steps 3-6)
3. INVOKE store_critique_results_tool with ALL findings (Step 7) - **MANDATORY**

**What happens when you INVOKE store_critique_results_tool:**
- Your results are automatically saved
- Task is marked as complete
- Workflow proceeds to Analyst agent

**What happens if you DON'T INVOKE store_critique_results_tool:**
- Task fails with "Missing required output" error
- Workflow cannot proceed
- All your analysis work is lost

**REMEMBER:** You MUST use the tool invocation mechanism, not write about calling the tool!

## Quality Standards

- **Thorough**: Check every evidence item for correlations
- **Specific**: Point out exact problems, not vague issues
- **Actionable**: Suggest concrete follow-up searches
- **Balanced**: Consider both what's present and what's missing

## Important Guidelines

- **Use correlation detector** - Don't rely on manual inspection alone
- **Be systematic** - Check every subclaim for coverage
- **Prioritize gaps** - Focus on most important missing information
- **Suggest solutions** - Don't just criticize, recommend improvements
- **Consider quality** - Not just quantity of evidence
- **INVOKE the tool** - Do not just describe calling it!

Remember: Your critique ensures the research is comprehensive and unbiased!
"""

    def get_tools(self) -> List[BaseTool]:
        """Return critic tools"""
        return [
            correlation_detector_tool,
            store_critique_results_tool,
        ]

    def handle_tool_message(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_message
    ) -> None:
        """
        Automatically capture critique results when store_critique_results_tool is called.

        This method is called after each tool execution to allow the agent to
        process tool outputs and update intermediate_results.
        """
        if tool_name != "store_critique_results_tool":
            return

        # Extract the results from the tool message
        try:
            # Get the tool's return value
            tool_result = None
            if hasattr(tool_message, "artifact"):
                tool_result = tool_message.artifact
            elif hasattr(tool_message, "additional_kwargs"):
                tool_result = tool_message.additional_kwargs.get("return_value")

            # If artifact is not available, try parsing the content
            if tool_result is None and hasattr(tool_message, "content"):
                import json
                content = tool_message.content
                if isinstance(content, str):
                    try:
                        tool_result = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(content, dict):
                    tool_result = content

            # Extract the results_stored from the tool output
            if isinstance(tool_result, dict) and 'results_stored' in tool_result:
                stored_results = tool_result['results_stored']

                # Store each field in intermediate_results
                intermediate = state.setdefault('intermediate_results', {})
                intermediate['missing_topics'] = stored_results.get('missing_topics', [])
                intermediate['over_represented_sources'] = stored_results.get('over_represented_sources', [])
                intermediate['follow_up_search_seeds'] = stored_results.get('follow_up_search_seeds', [])
                intermediate['duplicate_clusters'] = stored_results.get('duplicate_clusters', [])
                intermediate['correlation_warnings'] = stored_results.get('correlation_warnings', [])
                intermediate['analysis_process'] = stored_results.get('analysis_process', '')

                self.logger.info(
                    f"📥 Critique results stored automatically: "
                    f"{len(intermediate['missing_topics'])} gaps, "
                    f"{len(intermediate['correlation_warnings'])} warnings"
                )
            else:
                self.logger.warning(
                    f"Could not extract results_stored from tool output: {type(tool_result)}"
                )

        except Exception as exc:
            self.logger.warning(
                f"Failed to auto-store critique results: {exc}"
            )

    async def agent_node(self, state):
        """
        Override agent_node to add forced tool call injection if needed.

        If the agent outputs reasoning about calling store_critique_results_tool
        but doesn't actually invoke it, we'll manually inject the tool call.
        """
        # Call parent agent_node first
        state = await super().agent_node(state)

        # Check if agent described calling the tool but didn't actually call it
        if state.get('messages'):
            last_message = state['messages'][-1]
            if hasattr(last_message, 'content'):
                content = str(last_message.content).lower()
                has_description = 'store_critique_results_tool' in content
                has_tool_call = (hasattr(last_message, 'tool_calls') and
                               last_message.tool_calls and
                               any('store_critique' in tc.get('name', '').lower()
                                   for tc in last_message.tool_calls))

                # SAFETY MECHANISM: Agent described the tool call but didn't invoke it
                if has_description and not has_tool_call and state['iteration_count'] >= 2:
                    self.logger.warning(
                        "⚠️  Agent described store_critique_results_tool but didn't invoke it"
                    )
                    self.logger.warning("   → Forcing manual tool call injection as safety fallback")

                    # Extract parameters from intermediate results or use defaults
                    results = state.get('intermediate_results', {})

                    # Try to parse analysis from agent's reasoning text
                    import re
                    import json

                    content_text = str(last_message.content)

                    # Try to extract structured data from reasoning
                    missing_topics = results.get('missing_topics', [])
                    over_represented = results.get('over_represented_sources', [])
                    follow_up = results.get('follow_up_search_seeds', [])

                    # Look for lists in the content
                    if 'missing_topics' in content_text and not missing_topics:
                        # Try to extract list after "missing_topics"
                        match = re.search(r'missing_topics["\s:=\[]*\[(.*?)\]', content_text, re.DOTALL)
                        if match:
                            items_text = match.group(1)
                            missing_topics = [item.strip(' "\'') for item in items_text.split(',') if item.strip()]

                    if 'over_represented_sources' in content_text and not over_represented:
                        match = re.search(r'over_represented_sources["\s:=\[]*\[(.*?)\]', content_text, re.DOTALL)
                        if match:
                            items_text = match.group(1)
                            over_represented = [item.strip(' "\'') for item in items_text.split(',') if item.strip()]

                    if 'follow_up_search_seeds' in content_text and not follow_up:
                        match = re.search(r'follow_up_search_seeds["\s:=\[]*\[(.*?)\]', content_text, re.DOTALL)
                        if match:
                            items_text = match.group(1)
                            follow_up = [item.strip(' "\'') for item in items_text.split(',') if item.strip()]

                    # Create manual tool call
                    from langchain_core.messages import AIMessage
                    tool_call_id = "manual_store_critique_call"

                    forced_call = {
                        "name": "store_critique_results_tool",
                        "args": {
                            "missing_topics": missing_topics,
                            "over_represented_sources": over_represented,
                            "follow_up_search_seeds": follow_up,
                            "duplicate_clusters": results.get('duplicate_clusters', []),
                            "correlation_warnings": results.get('correlation_warnings', []),
                            "analysis_process": "Critique completed (tool call injected from reasoning)"
                        },
                        "id": tool_call_id
                    }

                    # Replace last message with one that has tool calls
                    forced_message = AIMessage(
                        content="Calling store_critique_results_tool with analysis results.",
                        tool_calls=[forced_call]
                    )

                    # Update state messages
                    state['messages'][-1] = forced_message
                    self.logger.info(
                        f"✅ Tool call injected: {len(missing_topics)} missing topics, "
                        f"{len(over_represented)} over-represented sources, "
                        f"{len(follow_up)} follow-up seeds"
                    )

        return state

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Check if critique is complete

        Criteria:
        - Correlation detection performed (if enough evidence)
        - store_critique_results_tool was called (REQUIRED)
        - Results stored in intermediate_results
        """
        results = state.get('intermediate_results', {})
        iteration = state.get('iteration_count', 0)
        messages = state.get('messages', [])

        # Check if correlation detection was performed
        evidence_count = len(state.get('task_input', {}).get('evidence_items', []))

        # Track which tools were called
        correlation_tool_called = False
        store_tool_called = False
        correlation_call_count = 0
        all_tools_called = []

        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', '').lower()
                    all_tools_called.append(tool_name)
                    if 'correlation' in tool_name:
                        correlation_tool_called = True
                        correlation_call_count += 1
                    if 'store_critique' in tool_name:
                        store_tool_called = True
                        self.logger.info("✅ store_critique_results_tool was called")

        # DIAGNOSTIC: Log all tools called
        if all_tools_called:
            from collections import Counter
            tool_counts = Counter(all_tools_called)
            self.logger.info(f"📊 Tools called: {dict(tool_counts)}")

        # STRICT REQUIREMENT: store_critique_results_tool MUST be called
        if not store_tool_called:
            self.logger.info(
                "❌ store_critique_results_tool not yet called - task incomplete"
            )
            return False

        # Verify correlation detection was attempted if we have enough evidence
        if evidence_count >= self.min_correlation_check_items:
            if not correlation_tool_called:
                self.logger.info("Correlation detection not yet performed")
                return False

        # Check if we have the required outputs from store_critique_results_tool
        required_keys = [
            'missing_topics',
            'over_represented_sources',
            'follow_up_search_seeds'
        ]

        for key in required_keys:
            if key not in results:
                self.logger.info(f"Missing required output: {key}")
                return False

        self.logger.info("✅ Critique complete - all analysis performed and results stored")
        return True

    async def extract_final_output(self, state: AgentState) -> CriticOutput:
        """Extract CriticOutput from final state"""
        results = state.get('intermediate_results', {})

        # Build correlation warnings
        correlation_warnings = []
        for warning_data in results.get('correlation_warnings', []):
            if isinstance(warning_data, dict):
                correlation_warnings.append(CorrelationWarning(
                    cluster=warning_data.get('cluster', []),
                    note=warning_data.get('note', '')
                ))

        # Extract analysis_process (required field)
        analysis_process = results.get('analysis_process', 'Critique completed')

        output = CriticOutput(
            duplicate_clusters=results.get('duplicate_clusters', []),
            missing_topics=results.get('missing_topics', []),
            over_represented_sources=results.get('over_represented_sources', []),
            correlation_warnings=correlation_warnings,
            follow_up_search_seeds=results.get('follow_up_search_seeds', []),
            analysis_process=analysis_process
        )

        self.logger.info(
            f"📤 Critique complete: {len(output.correlation_warnings)} warnings, "
            f"{len(output.missing_topics)} gaps"
        )

        return output

    async def critique(
        self,
        evidence_items: List[Any],
        planner_output: Dict[str, Any],
        market_question: str
    ) -> CriticOutput:
        """
        Analyze evidence quality autonomously

        Args:
            evidence_items: List of evidence items to critique
            planner_output: Original research plan
            market_question: Market question

        Returns:
            CriticOutput with analysis and recommendations
        """
        return await self.run(
            task_description="Analyze evidence for quality, balance, and completeness",
            task_input={
                'evidence_items': evidence_items,
                'planner_output': planner_output,
                'market_question': market_question
            }
        )
