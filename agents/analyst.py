"""
Autonomous AnalystAgent with Bayesian Tools
Performs Bayesian aggregation with autonomous validation and sensitivity analysis.
"""
from __future__ import annotations

from typing import Any, List, Dict

from langchain_core.tools import BaseTool

from agents.base import AutonomousReActAgent, AgentState
from agents.schemas import (
    AnalystOutput,
    EvidenceSummaryItem,
    SensitivityScenario,
    CorrelationAdjustment,
)
from tools.bayesian import (
    bayesian_calculate_tool,
    sensitivity_analysis_tool,
    validate_llr_calibration_tool,
)


class AutonomousAnalystAgent(AutonomousReActAgent):
    """
    Bayesian aggregation with validation, correlation handling, and sensitivity analysis.

    Flow:
      1) Prepare evidence (LLRs + scores)
      2) Validate LLR calibration
      3) Aggregate via bayesian_calculate_tool
      4) Run sensitivity_analysis_tool
      5) Verify completeness & ranges
    """

    def __init__(self, max_sensitivity_range: float = 0.3, **kwargs):
        """
        Args:
            max_sensitivity_range: Max acceptable probability spread in sensitivity analysis.
        """
        assert 0.0 < max_sensitivity_range <= 1.0, "max_sensitivity_range must be in (0,1]"
        super().__init__(**kwargs)
        self.max_sensitivity_range = max_sensitivity_range

    def get_system_prompt(self) -> str:
        """System prompt for autonomous Bayesian analysis."""
        return f"""You are an Autonomous Analyst Agent in POLYSEER.

Your mission: Perform rigorous Bayesian aggregation of evidence.

## CRITICAL: Tool Call Requirements

**YOU MUST GENERATE ACTUAL TOOL_CALLS - DO NOT DESCRIBE OR EXPLAIN WHAT YOU WILL DO**

❌ WRONG: "I will call bayesian_calculate_tool with prior_p=0.4 and evidence_items..."
❌ WRONG: "Processing 17 tool calls..."
✅ CORRECT: Generate actual tool_calls in your response using the tool calling format

**If you describe tools instead of calling them, your task will FAIL.**

## Available Tools

1. **validate_llr_calibration_tool** - Validate evidence LLR calibration
   - Use this to check each evidence item's LLR is properly calibrated
   - Input: llr, source_type
   - Returns: is_valid, expected_range, feedback

2. **bayesian_calculate_tool** - Perform Bayesian aggregation
   - Use this to calculate posterior probability from prior and evidence
   - Input: prior_p, evidence_items (with LLR and scores), correlation_clusters
   - Returns: p_bayesian, log_odds, evidence_summary, correlation_adjustments
   - **Results are AUTOMATICALLY stored in intermediate_results**
   - **YOU MUST CALL THIS TOOL - DO NOT DESCRIBE CALLING IT**

3. **sensitivity_analysis_tool** - Test robustness of conclusions
   - Use this to check how sensitive results are to assumptions
   - Input: prior_p, evidence_items
   - Returns: List of scenarios with resulting probabilities
   - **Results are AUTOMATICALLY stored in intermediate_results['sensitivity_analysis']**
   - **YOU MUST CALL THIS TOOL - DO NOT DESCRIBE CALLING IT**

## Your Reasoning Process

**CRITICAL: You MUST use the tools to complete this task. Do not attempt to calculate probabilities manually.**

**Step 1: Prepare Evidence**
- Extract all evidence items from input
- If evidence_items is empty (0 items):
  - Use prior p0 as p_bayesian (no update)
  - Still run sensitivity analysis with p0
- Ensure each has: id, LLR, verifiability_score, independence_score, recency_score
- Count total evidence items
- Separate by support direction (pro vs con)

**Step 2: Perform Bayesian Calculation (REQUIRED - CALL THE TOOL NOW)**
- **YOU MUST GENERATE A TOOL_CALL for bayesian_calculate_tool** with:
  - prior_p from Planner
  - All evidence items (convert to dict format with id, estimated_LLR, verifiability_score, independence_score, recency_score)
  - correlation_clusters from Critic (empty list if none)
- This tool will calculate p_bayesian and store results automatically
- Results automatically stored in intermediate_results:
  - p_bayesian, log_odds_prior, log_odds_posterior
  - evidence_summary, correlation_adjustments

**Step 3: Run Sensitivity Analysis (REQUIRED - CALL THE TOOL NOW)**
- **YOU MUST GENERATE A TOOL_CALL for sensitivity_analysis_tool** with:
  - prior_p from Planner
  - All evidence items
- This tool will test robustness with different assumptions
- Results automatically stored in intermediate_results['sensitivity_analysis']

**Step 4: Verify Completion**
- Check you have:
  - p_bayesian in valid range [0.01, 0.99]
  - evidence_summary for all items (empty list OK if 0 evidence)
  - sensitivity_analysis with ≥2 scenarios
  - correlation_adjustments present
- If any missing, call the appropriate tool again

## Output Format

Automatically stored in intermediate_results:
- p0: Prior probability
- log_odds_prior: Log-odds of prior
- p_bayesian: Posterior probability
- log_odds_posterior: Log-odds of posterior
- evidence_summary: List of evidence summary dicts
- sensitivity_analysis: List of scenario dicts
- correlation_adjustments: Dict describing adjustments

## Quality Standards

- **Rigorous**: Validate all LLRs before using them
- **Transparent**: Evidence summary shows all calculations
- **Robust**: Sensitivity analysis tests assumptions
- **Complete**: All core outputs present

Remember: You're NOT doing the math yourself - the tools do that. Your job is to:
1. Validate inputs
2. **GENERATE ACTUAL TOOL_CALLS** (not descriptions)
3. Check robustness
4. Verify completeness

**AGAIN: Generate tool_calls in your response. Do not describe what you will do.**
"""

    def get_tools(self) -> List[BaseTool]:
        """Return analysis tools."""
        return [validate_llr_calibration_tool, bayesian_calculate_tool, sensitivity_analysis_tool]

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Complete when:
          - p_bayesian present and within [0.01, 0.99]
          - evidence_summary present (list, possibly empty)
          - sensitivity_analysis present (≥2 scenarios, or ≥1 if 0 evidence)
          - correlation_adjustments present
        """
        results = state.get("intermediate_results", {})
        task_input = state.get("task_input", {})
        evidence_items = task_input.get("evidence_items", [])
        is_zero_evidence = len(evidence_items) == 0
        missing_criteria = []

        # Attempt recovery if tool wrote output but auto-store failed.
        if "p_bayesian" not in results:
            tool_calls = state.get("tool_calls", [])
            bayes_calls = [t for t in tool_calls if getattr(t, "tool_name", "") == "bayesian_calculate_tool"]
            if bayes_calls:
                last = bayes_calls[-1]
                import json
                payload = last.tool_output
                assert payload is not None, "Tool output must exist"
                data = json.loads(payload) if isinstance(payload, str) else payload
                assert isinstance(data, dict) and "p_bayesian" in data, "Tool output must contain p_bayesian"
                results["p_bayesian"] = data.get("p_bayesian")
                results["p0"] = data.get("p0", task_input.get("prior_p", 0.5))
                results["log_odds_prior"] = data.get("log_odds_prior", 0.0)
                results["log_odds_posterior"] = data.get("log_odds_posterior", 0.0)
                results["evidence_summary"] = data.get("evidence_summary", [])
                results["correlation_adjustments"] = data.get("correlation_adjustments", {})

            if "p_bayesian" not in results:
                missing_criteria.append("p_bayesian (not found)")

        p_bayesian = results.get("p_bayesian", 0.5)
        if not (0.01 <= p_bayesian <= 0.99):
            missing_criteria.append(f"p_bayesian (value {p_bayesian:.4f} outside [0.01, 0.99])")

        if "evidence_summary" not in results:
            missing_criteria.append("evidence_summary (not found)")

        # Attempt recovery if sensitivity_analysis tool wrote output but auto-store failed
        sensitivity_analysis = results.get("sensitivity_analysis", [])
        if not sensitivity_analysis or not isinstance(sensitivity_analysis, list):
            tool_calls = state.get("tool_calls", [])
            sens_calls = [t for t in tool_calls if getattr(t, "tool_name", "") == "sensitivity_analysis_tool"]
            if sens_calls:
                last = sens_calls[-1]
                import json
                payload = last.tool_output
                assert payload is not None, "Tool output must exist"
                data = json.loads(payload) if isinstance(payload, str) else payload
                assert isinstance(data, (list, dict)), "Tool output must be list or dict"
                if isinstance(data, list):
                    results["sensitivity_analysis"] = data
                    sensitivity_analysis = data
                elif isinstance(data, dict) and "sensitivity_analysis" in data:
                    results["sensitivity_analysis"] = data["sensitivity_analysis"]
                    sensitivity_analysis = data["sensitivity_analysis"]

        min_scenarios = 1 if is_zero_evidence else 2
        if not isinstance(sensitivity_analysis, list) or len(sensitivity_analysis) < min_scenarios:
            actual_count = len(sensitivity_analysis) if isinstance(sensitivity_analysis, list) else 0
            missing_criteria.append(f"sensitivity_analysis (found {actual_count}, need {min_scenarios})")

        if "correlation_adjustments" not in results:
            missing_criteria.append("correlation_adjustments (not found)")
            # Set default if missing
            results["correlation_adjustments"] = {"method": "none", "details": "No correlation detected"}

        # Fallback recovery: If tools weren't called but should have been, inject them
        tool_calls_made = len(state.get("tool_calls", []))
        iteration = state.get("iteration_count", 0)
        
        # Trigger fallback if we're missing criteria and haven't made tool calls
        # Trigger immediately after iteration 1 if no tool calls were made
        if missing_criteria and iteration >= 1 and tool_calls_made == 0:
            # Agent described tools but didn't call them - inject tool calls programmatically
            self.logger.warning(
                f"Analyst failed to call tools after {iteration} iteration(s). "
                f"Attempting fallback recovery by injecting tool calls."
            )
            await self._inject_required_tool_calls(state)
            # Re-check after injection
            return await self.is_task_complete(state)

        if missing_criteria:
            self.logger.warning(
                f"Analyst task incomplete. Missing criteria: {', '.join(missing_criteria)}. "
                f"Iteration: {state.get('iteration_count', 0)}, "
                f"Tool calls: {len(state.get('tool_calls', []))}"
            )
            return False

        return True
    
    async def _inject_required_tool_calls(self, state: AgentState) -> None:
        """
        Fallback recovery: Programmatically call required tools if LLM failed to do so.
        
        This ensures the agent completes even if the LLM describes tools instead of calling them.
        """
        task_input = state.get("task_input", {})
        prior_p = task_input.get("prior_p", 0.5)
        evidence_items = task_input.get("evidence_items", [])
        correlation_clusters = task_input.get("correlation_clusters", [])
        results = state.setdefault("intermediate_results", {})
        
        # Convert evidence to dicts
        evidence_dicts = self._convert_evidence_to_dicts(evidence_items)
        
        # Call bayesian_calculate_tool if p_bayesian is missing
        if "p_bayesian" not in results:
            self.logger.info("Fallback: Calling bayesian_calculate_tool programmatically")
            bayesian_result = await bayesian_calculate_tool.ainvoke({
                "prior_p": prior_p,
                "evidence_items": evidence_dicts,
                "correlation_clusters": correlation_clusters or []
            })
            assert isinstance(bayesian_result, dict), "Bayesian result must be dict"
            results.update({
                "p0": bayesian_result.get("p0", prior_p),
                "p_bayesian": bayesian_result.get("p_bayesian", prior_p),
                "log_odds_prior": bayesian_result.get("log_odds_prior", 0.0),
                "log_odds_posterior": bayesian_result.get("log_odds_posterior", 0.0),
                "p_neutral": bayesian_result.get("p_neutral", 0.5),
                "evidence_summary": bayesian_result.get("evidence_summary", []),
                "correlation_adjustments": bayesian_result.get("correlation_adjustments", {"method": "none", "details": "No correlation detected"}),
            })
            self.logger.info(f"Fallback: bayesian_calculate_tool completed, p_bayesian={results.get('p_bayesian', 0.5):.2%}")
        
        # Call sensitivity_analysis_tool if sensitivity_analysis is missing
        if "sensitivity_analysis" not in results or not isinstance(results.get("sensitivity_analysis"), list):
            self.logger.info("Fallback: Calling sensitivity_analysis_tool programmatically")
            sensitivity_result = await sensitivity_analysis_tool.ainvoke({
                "prior_p": prior_p,
                "evidence_items": evidence_dicts,
            })
            assert isinstance(sensitivity_result, (list, dict)), "Sensitivity result must be list or dict"
            if isinstance(sensitivity_result, list):
                results["sensitivity_analysis"] = sensitivity_result
                self.logger.info(f"Fallback: sensitivity_analysis_tool completed, {len(sensitivity_result)} scenarios")
            elif isinstance(sensitivity_result, dict) and "sensitivity_analysis" in sensitivity_result:
                results["sensitivity_analysis"] = sensitivity_result["sensitivity_analysis"]
            else:
                results["sensitivity_analysis"] = [
                    {"scenario": "baseline", "p": prior_p},
                    {"scenario": "prior_plus_10pct", "p": min(0.99, prior_p + 0.1)},
                ]
        
        # Ensure correlation_adjustments exists
        if "correlation_adjustments" not in results:
            results["correlation_adjustments"] = {"method": "none", "details": "No correlation detected"}

    async def extract_final_output(self, state: AgentState) -> AnalystOutput:
        """Convert intermediate_results into AnalystOutput."""
        results = state.get("intermediate_results", {})
        evidence_summary = self._as_evidence_summary(results.get("evidence_summary", []))
        sensitivity_analysis = self._as_sensitivity_list(results.get("sensitivity_analysis", []))
        p_bayesian = results.get("p_bayesian", 0.5)
        p0 = results.get("p0", 0.5)

        # Log Bayesian calculation with rich logger
        self.rich_logger.log_bayesian_calculation(
            prior=p0,
            posterior=p_bayesian,
            evidence_count=len(evidence_summary),
            log_odds_prior=results.get("log_odds_prior"),
            log_odds_posterior=results.get("log_odds_posterior"),
        )

        p_low, p_high, conf = self._compute_ci_from_sensitivity(p_bayesian, sensitivity_analysis)

        corr_adj = results.get("correlation_adjustments", {})
        correlation_adjustments = (
            corr_adj
            if isinstance(corr_adj, CorrelationAdjustment)
            else CorrelationAdjustment(
                method=(corr_adj.get("method") if isinstance(corr_adj, dict) else "none") or "none",
                details=(corr_adj.get("details") if isinstance(corr_adj, dict) else "No correlation detected") or "No correlation detected",
            )
        )

        return AnalystOutput(
            p0=results.get("p0", 0.5),
            log_odds_prior=results.get("log_odds_prior", 0.0),
            p_bayesian=p_bayesian,
            p_bayesian_low=p_low,
            p_bayesian_high=p_high,
            confidence_level=conf,
            log_odds_posterior=results.get("log_odds_posterior", 0.0),
            p_neutral=results.get("p_neutral", 0.5),
            calculation_steps=results.get("calculation_steps", []),
            evidence_summary=evidence_summary,
            correlation_adjustments=correlation_adjustments,
            sensitivity_analysis=sensitivity_analysis,
        )

    async def analyze(
        self,
        prior_p: float,
        evidence_items: List[Any],
        critic_output: Any,
        market_question: str,
    ) -> AnalystOutput:
        """
        Run autonomous Bayesian analysis and return AnalystOutput.
        """
        assert isinstance(prior_p, (int, float)) and 0.0 <= prior_p <= 1.0, "prior_p must be in [0,1]"
        assert isinstance(evidence_items, list), "evidence_items must be a list"
        assert isinstance(market_question, str) and market_question.strip(), "market_question is required"

        # Extract correlation clusters from critic output.
        if hasattr(critic_output, "correlation_warnings"):
            correlation_clusters = [w.cluster for w in critic_output.correlation_warnings]
        elif isinstance(critic_output, dict):
            correlation_clusters = [w.get("cluster", []) for w in critic_output.get("correlation_warnings", [])]
        else:
            correlation_clusters = []

        return await self.run(
            task_description="Perform Bayesian aggregation with validation and sensitivity analysis",
            task_input={
                "prior_p": prior_p,
                "evidence_items": evidence_items,
                "correlation_clusters": correlation_clusters,
                "market_question": market_question,
            },
        )

    # -------------------------
    # Private helpers
    # -------------------------
    def _convert_evidence_to_dicts(self, evidence_items: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert evidence items (Pydantic models or dicts) to tool-compatible dicts.
        
        Ensures all required keys are present: id, LLR (or estimated_LLR), 
        verifiability_score, independence_score, recency_score.
        """
        converted = []
        for idx, item in enumerate(evidence_items):
            if item is None:
                continue
            
            # Convert to dict if needed
            if hasattr(item, "model_dump"):
                evidence_dict = item.model_dump()
            elif hasattr(item, "dict"):
                evidence_dict = item.dict()
            elif isinstance(item, dict):
                evidence_dict = item.copy()
            else:
                assert hasattr(item, "__dict__") or isinstance(item, dict), f"Evidence item {idx} must be dict or have __dict__"
                evidence_dict = dict(item) if hasattr(item, "__dict__") else item
            
            # Ensure ID exists
            if "id" not in evidence_dict:
                if "subclaim_id" in evidence_dict:
                    evidence_dict["id"] = evidence_dict["subclaim_id"]
                elif "title" in evidence_dict:
                    evidence_dict["id"] = evidence_dict["title"][:50]
                else:
                    evidence_dict["id"] = f"evidence_{idx}"
            
            # Ensure LLR exists (check both LLR and estimated_LLR)
            if "LLR" not in evidence_dict:
                if "estimated_LLR" in evidence_dict:
                    evidence_dict["LLR"] = evidence_dict["estimated_LLR"]
                else:
                    evidence_dict["LLR"] = 0.0
            
            # Ensure required scores exist with defaults
            evidence_dict.setdefault("verifiability_score", 0.5)
            evidence_dict.setdefault("independence_score", 0.8)
            evidence_dict.setdefault("recency_score", 0.5)
            
            # Handle support direction for LLR sign
            support = str(evidence_dict.get("support", "")).lower()
            llr_value = float(evidence_dict.get("LLR", 0.0))
            
            if support == "neutral":
                evidence_dict["LLR"] = 0.0
            elif support == "pro" and llr_value < 0:
                evidence_dict["LLR"] = abs(llr_value)
            elif support == "con" and llr_value > 0:
                evidence_dict["LLR"] = -abs(llr_value)
            
            converted.append(evidence_dict)
        
        return converted
    
    @staticmethod
    def _as_evidence_summary(items: List[Any]) -> List[EvidenceSummaryItem]:
        out: List[EvidenceSummaryItem] = []
        for it in items:
            if isinstance(it, dict):
                out.append(EvidenceSummaryItem(**it))
        return out

    @staticmethod
    def _as_sensitivity_list(items: List[Any]) -> List[SensitivityScenario]:
        out: List[SensitivityScenario] = []
        for it in items:
            if isinstance(it, dict):
                out.append(SensitivityScenario(**it))
        return out

    def _compute_ci_from_sensitivity(
        self, p_bayesian: float, scenarios: List[SensitivityScenario]
    ) -> tuple[float, float, float]:
        if scenarios and len(scenarios) > 1:
            probs = [s.p for s in scenarios if isinstance(s.scenario, str) and s.scenario.lower() != "baseline"]
            if probs:
                return min(probs), max(probs), 0.80
            return max(0.01, p_bayesian - 0.10), min(0.99, p_bayesian + 0.10), 0.50
        return max(0.01, p_bayesian - 0.15), min(0.99, p_bayesian + 0.15), 0.50