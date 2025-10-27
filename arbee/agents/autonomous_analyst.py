"""
Autonomous AnalystAgent with Bayesian Tools
Performs Bayesian aggregation with autonomous validation and sensitivity analysis
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
import logging

from arbee.agents.autonomous_base import AutonomousReActAgent, AgentState
from arbee.agents.schemas import AnalystOutput, EvidenceSummaryItem, SensitivityScenario, CorrelationAdjustment
from arbee.tools.bayesian import (
    bayesian_calculate_tool,
    sensitivity_analysis_tool,
    validate_llr_calibration_tool
)

logger = logging.getLogger(__name__)


class AutonomousAnalystAgent(AutonomousReActAgent):
    """
    Autonomous Analyst Agent - Bayesian aggregation with validation

    Autonomous Capabilities:
    - Validates LLR calibration before aggregation
    - Performs Bayesian calculation with correlation adjustments
    - Runs sensitivity analysis automatically
    - Checks result robustness
    - Iteratively refines if validation fails

    Reasoning Flow:
    1. Prepare evidence items (extract LLRs and scores)
    2. Validate LLR calibration for each evidence item
    3. Flag miscalibrated evidence for review
    4. Perform Bayesian aggregation using bayesian_calculate_tool
    5. Run sensitivity analysis using sensitivity_analysis_tool
    6. Check if results are robust (low sensitivity)
    7. Generate calculation explanation
    8. Validate completeness, refine if needed
    """

    def __init__(
        self,
        max_sensitivity_range: float = 0.3,  # Max acceptable probability range in sensitivity
        **kwargs
    ):
        """
        Initialize Autonomous Analyst

        Args:
            max_sensitivity_range: Max acceptable range in sensitivity analysis
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(**kwargs)
        self.max_sensitivity_range = max_sensitivity_range

    def get_system_prompt(self) -> str:
        """System prompt for autonomous Bayesian analysis"""
        return f"""You are an Autonomous Analyst Agent in POLYSEER.

Your mission: Perform rigorous Bayesian aggregation of evidence.

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

3. **sensitivity_analysis_tool** - Test robustness of conclusions
   - Use this to check how sensitive results are to assumptions
   - Input: prior_p, evidence_items
   - Returns: List of scenarios with resulting probabilities
   - **Results are AUTOMATICALLY stored in intermediate_results['sensitivity_analysis']**

## Your Reasoning Process

**Step 1: Prepare Evidence**
- Extract all evidence items from input
- If evidence_items is empty (0 items):
  - Use prior p0 as p_bayesian (no update)
  - Still run sensitivity analysis with p0
- Ensure each has: id, LLR, verifiability_score, independence_score, recency_score
- Count total evidence items
- Separate by support direction (pro vs con)

**Step 2: Validate LLR Calibration**
- For each evidence item, use validate_llr_calibration_tool
- Check LLR matches source_type calibration ranges:
  - Primary: ±1-3
  - High-quality secondary: ±0.3-1.0
  - Secondary: ±0.1-0.5
  - Weak: ±0.01-0.2
- Flag any miscalibrated evidence
- Decide: Include with warning, or exclude?

**Step 3: Perform Bayesian Calculation**
- Use bayesian_calculate_tool with:
  - prior_p from Planner
  - All validated evidence items
  - correlation_clusters from Critic
- Results automatically stored in intermediate_results:
  - p_bayesian, log_odds_prior, log_odds_posterior
  - evidence_summary, correlation_adjustments

**Step 4: Run Sensitivity Analysis**
- Use sensitivity_analysis_tool
- Check how p_bayesian changes with:
  - ±25% LLR adjustment
  - Prior ±0.1 adjustment
  - Different correlation assumptions
- Assess robustness: Is range < {self.max_sensitivity_range} (30%)?
- Results automatically stored

**Step 5: Verify Completion**
- Check you have:
  - p_bayesian in valid range [0.01, 0.99]
  - evidence_summary for all items (empty list OK if 0 evidence)
  - sensitivity_analysis with ≥2 scenarios
  - correlation_adjustments present
- If any missing, investigate and ensure tool calls succeeded

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
2. Call tools correctly
3. Check robustness
4. Verify completeness
"""

    def get_tools(self) -> List[BaseTool]:
        """Return analysis tools"""
        return [
            validate_llr_calibration_tool,
            bayesian_calculate_tool,
            sensitivity_analysis_tool,
        ]

    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Check if analysis is complete

        Criteria:
        - Bayesian calculation performed (or 0 evidence handled)
        - Sensitivity analysis performed (or 0 evidence handled)
        - p_bayesian in valid range
        - Evidence summary present
        - Correlation adjustments present
        """
        results = state.get('intermediate_results', {})
        task_input = state.get('task_input', {})
        evidence_items = task_input.get('evidence_items', [])
        is_zero_evidence = len(evidence_items) == 0
        iteration = state.get('iteration_count', 0)

        # DIAGNOSTIC: Log what we have so far
        self.logger.info(f"📊 Completion check (iteration {iteration}):")
        self.logger.info(f"   - p_bayesian present: {'p_bayesian' in results}")
        self.logger.info(f"   - evidence_summary: {len(results.get('evidence_summary', []))} items")
        self.logger.info(f"   - sensitivity_analysis: {len(results.get('sensitivity_analysis', []))} scenarios")
        self.logger.info(f"   - correlation_adjustments present: {'correlation_adjustments' in results}")
        self.logger.info(f"   - evidence count: {len(evidence_items)}")

        # Check if Bayesian calculation done (with fallback check)
        if 'p_bayesian' not in results:
            # Fallback: Check if bayesian_calculate_tool was called
            tool_calls = state.get('tool_calls', [])
            bayesian_calls = [t for t in tool_calls if t.tool_name == 'bayesian_calculate_tool']

            if bayesian_calls:
                self.logger.warning(
                    f"⚠️  bayesian_calculate_tool was called ({len(bayesian_calls)}x) "
                    "but p_bayesian not in results - auto-storage may have failed"
                )
                # Try to extract from last call
                last_call = bayesian_calls[-1]
                try:
                    if hasattr(last_call, 'tool_output') and last_call.tool_output:
                        import json
                        output_data = json.loads(last_call.tool_output) if isinstance(last_call.tool_output, str) else last_call.tool_output
                        if isinstance(output_data, dict) and 'p_bayesian' in output_data:
                            self.logger.info("✅ Recovered p_bayesian from tool_calls history")
                            results['p_bayesian'] = output_data.get('p_bayesian')
                            results['p0'] = output_data.get('p0', task_input.get('prior_p', 0.5))
                            results['log_odds_prior'] = output_data.get('log_odds_prior', 0.0)
                            results['log_odds_posterior'] = output_data.get('log_odds_posterior', 0.0)
                except Exception as e:
                    self.logger.warning(f"Failed to recover from tool_calls: {e}")

            if 'p_bayesian' not in results:
                self.logger.info("Bayesian calculation not yet performed")
                return False

        # Check p_bayesian range
        p_bayesian = results.get('p_bayesian', 0.5)
        if not (0.01 <= p_bayesian <= 0.99):
            self.logger.warning(f"p_bayesian {p_bayesian} outside valid range [0.01, 0.99]")
            return False

        # Check evidence summary present (can be empty list for 0 evidence)
        if 'evidence_summary' not in results:
            self.logger.info("Evidence summary not yet generated")
            return False

        # Check sensitivity analysis (relaxed for 0 evidence)
        min_scenarios = 1 if is_zero_evidence else 2
        sensitivity = results.get('sensitivity_analysis', [])
        if not sensitivity or len(sensitivity) < min_scenarios:
            self.logger.info(f"Need at least {min_scenarios} sensitivity scenarios (0 evidence: {is_zero_evidence})")
            return False

        # Check correlation adjustments present
        if 'correlation_adjustments' not in results:
            self.logger.info("Correlation adjustments not yet generated")
            return False

        self.logger.info(f"✅ Analysis complete: p_bayesian={p_bayesian:.2%} (0 evidence: {is_zero_evidence})")
        return True

    async def extract_final_output(self, state: AgentState) -> AnalystOutput:
        """Extract AnalystOutput from final state"""
        results = state.get('intermediate_results', {})

        # Build evidence summary
        evidence_summary = []
        for item_data in results.get('evidence_summary', []):
            if isinstance(item_data, dict):
                evidence_summary.append(EvidenceSummaryItem(**item_data))

        # Build sensitivity analysis
        sensitivity_analysis = []
        for scenario_data in results.get('sensitivity_analysis', []):
            if isinstance(scenario_data, dict):
                sensitivity_analysis.append(SensitivityScenario(**scenario_data))

        # Calculate confidence intervals from sensitivity analysis
        p_bayesian = results.get('p_bayesian', 0.5)
        if sensitivity_analysis and len(sensitivity_analysis) > 1:
            # Extract probabilities from non-baseline scenarios
            sensitivity_probs = [
                s.p for s in sensitivity_analysis
                if s.scenario.lower() != 'baseline'
            ]
            if sensitivity_probs:
                p_bayesian_low = min(sensitivity_probs)
                p_bayesian_high = max(sensitivity_probs)
                # Confidence level depends on the scenarios (±25% LLR ≈ 80% CI)
                confidence_level = 0.80
            else:
                # Fallback: use ±10% of p_bayesian
                p_bayesian_low = max(0.01, p_bayesian - 0.10)
                p_bayesian_high = min(0.99, p_bayesian + 0.10)
                confidence_level = 0.50  # Lower confidence for fallback
        else:
            # No sensitivity analysis: use conservative ±15% range
            p_bayesian_low = max(0.01, p_bayesian - 0.15)
            p_bayesian_high = min(0.99, p_bayesian + 0.15)
            confidence_level = 0.50

        # Create proper CorrelationAdjustment object (required by schema)
        corr_adj_data = results.get('correlation_adjustments', {})
        if isinstance(corr_adj_data, CorrelationAdjustment):
            correlation_adjustments = corr_adj_data
        elif isinstance(corr_adj_data, dict) and corr_adj_data:
            # If dict with data, use it
            correlation_adjustments = CorrelationAdjustment(
                method=corr_adj_data.get('method', 'none'),
                details=corr_adj_data.get('details', 'No correlation adjustments applied')
            )
        else:
            # Default: no correlation detected
            correlation_adjustments = CorrelationAdjustment(
                method='none',
                details='No correlation detected among evidence items'
            )

        output = AnalystOutput(
            p0=results.get('p0', 0.5),
            log_odds_prior=results.get('log_odds_prior', 0.0),
            p_bayesian=p_bayesian,
            p_bayesian_low=p_bayesian_low,
            p_bayesian_high=p_bayesian_high,
            confidence_level=confidence_level,
            log_odds_posterior=results.get('log_odds_posterior', 0.0),
            p_neutral=results.get('p_neutral', 0.5),
            calculation_steps=results.get('calculation_steps', []),
            evidence_summary=evidence_summary,
            correlation_adjustments=correlation_adjustments,
            sensitivity_analysis=sensitivity_analysis
        )

        self.logger.info(
            f"📤 Analysis complete: p={output.p_bayesian:.2%} "
            f"[{output.p_bayesian_low:.2%} - {output.p_bayesian_high:.2%}] "
            f"({output.confidence_level:.0%} CI) "
            f"(prior={output.p0:.2%})"
        )

        return output

    async def analyze(
        self,
        prior_p: float,
        evidence_items: List[Any],
        critic_output: Any,
        market_question: str
    ) -> AnalystOutput:
        """
        Perform autonomous Bayesian analysis

        Args:
            prior_p: Prior probability from Planner
            evidence_items: All evidence from Researchers
            critic_output: Correlation warnings from Critic
            market_question: Market question

        Returns:
            AnalystOutput with p_bayesian and full analysis
        """
        # Extract correlation clusters from critic output
        correlation_clusters = []
        if hasattr(critic_output, 'correlation_warnings'):
            correlation_clusters = [w.cluster for w in critic_output.correlation_warnings]
        elif isinstance(critic_output, dict):
            warnings = critic_output.get('correlation_warnings', [])
            correlation_clusters = [w.get('cluster', []) for w in warnings]

        return await self.run(
            task_description="Perform Bayesian aggregation with validation and sensitivity analysis",
            task_input={
                'prior_p': prior_p,
                'evidence_items': evidence_items,
                'correlation_clusters': correlation_clusters,
                'market_question': market_question
            }
        )
