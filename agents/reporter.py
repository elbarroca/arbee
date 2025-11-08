"""
Autonomous ReporterAgent
Generates final reports with autonomous validation
"""
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
import logging
import re
import json

from agents.base import AutonomousReActAgent, AgentState
from agents.schemas import ReporterOutput, TopDriver

logger = logging.getLogger(__name__)


class AutonomousReporterAgent(AutonomousReActAgent):
    """
    Autonomous Reporter Agent - Generates comprehensive final reports

    Autonomous Capabilities:
    - Validates completeness of all inputs
    - Generates executive summary
    - Creates structured JSON + Markdown outputs
    - Ensures proper disclaimers
    - Iteratively refines if validation fails
    """

    def get_system_prompt(self) -> str:
        """System prompt for report generation"""
        return """You are an Autonomous Reporter Agent in POLYSEER.

Your mission: Generate comprehensive, accurate final reports with full provenance.

## Process

1. **Gather All Outputs**
   - Planner output (prior, subclaims, search seeds)
   - Researcher outputs (evidence items)
   - Critic output (warnings, gaps)
   - Analyst output (p_bayesian, confidence_interval, sensitivity)
   - Arbitrage opportunities
   - Edge signals (information asymmetry, market inefficiency, sentiment edge, base rate violations)

2. **Extract Key Information from Analyst Output**
   - p_bayesian: The Bayesian probability from analyst_output.p_bayesian
   - confidence_interval: [p_bayesian_low, p_bayesian_high] from analyst_output
   - Top PRO drivers: Extract from evidence_summary (items with positive LLR, sorted by strength)
   - Top CON drivers: Extract from evidence_summary (items with negative LLR, sorted by strength)

3. **Generate Executive Summary**
   - Key finding: p_bayesian with confidence interval
   - Top 3 PRO drivers (strongest evidence supporting YES)
   - Top 3 CON drivers (strongest evidence supporting NO)
   - Critical uncertainties and gaps
   - Arbitrage summary (if opportunities found)
   - Edge detection summary (if edge signals detected)
   - MUST be 500-1000 characters (not words!)

4. **Create TL;DR**
   - 1-2 sentence summary of key finding
   - Include p_bayesian and main driver
   - MUST be max 300 characters

5. **Create Arbitrage Summary**
   - Summarize arbitrage opportunities found
   - If none found, state "No arbitrage opportunities detected"
   - Include edge signals if significant
   - **If threshold_probabilities exist in analyst_output**, include threshold-specific analysis:
     * Format: "30%+ threshold: p_bayesian=XX% vs market=YY%, edge=ZZ%"
     * Compare each threshold's p_bayesian vs market price
     * Highlight significant edges (>2%) per threshold

6. **Create Full JSON Package**
   - Include all data from planner_output, researcher_output, critic_output, analyst_output
   - This will be stored in full_json field

## CRITICAL: Output Format

You MUST output your report in JSON format:

```json
{
  "executive_summary": "Your executive summary here (200-600 CHARACTERS, not words!)...",
  "tldr": "1-2 sentence summary (max 300 characters)",
  "top_pro_drivers": [
    {"direction": "pro", "summary": "Driver 1 summary", "strength": "strong"},
    {"direction": "pro", "summary": "Driver 2 summary", "strength": "moderate"},
    {"direction": "pro", "summary": "Driver 3 summary", "strength": "weak"}
  ],
  "top_con_drivers": [
    {"direction": "con", "summary": "Driver 1 summary", "strength": "strong"},
    {"direction": "con", "summary": "Driver 2 summary", "strength": "moderate"},
    {"direction": "con", "summary": "Driver 3 summary", "strength": "weak"}
  ],
  "arbitrage_summary": "Summary of arbitrage opportunities or 'No arbitrage opportunities detected'",
  "next_steps": ["Step 1", "Step 2"]
}
```

IMPORTANT:
- executive_summary MUST be 200-600 CHARACTERS (truncate if longer)
- tldr MUST be max 300 CHARACTERS
- Extract p_bayesian and confidence_interval from analyst_output
- Extract top drivers from evidence_summary (sort by LLR magnitude)
- arbitrage_summary must be a string (not empty object)

Store in intermediate_results:
- executive_summary: string (REQUIRED, 200-600 chars)
- tldr: string (REQUIRED, max 300 chars)
- top_pro_drivers: list of dicts with direction, summary, strength (REQUIRED, 3 items)
- top_con_drivers: list of dicts with direction, summary, strength (REQUIRED, 3 items)
- arbitrage_summary: string (REQUIRED)
- next_steps: list of strings (optional)
- full_json: dict with all data (REQUIRED)

Complete when all sections validated and formatted.
"""

    def get_tools(self) -> List[BaseTool]:
        """Reporter doesn't need special tools currently"""
        return []

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from LLM text response.
        
        Tries multiple strategies:
        1. JSON blocks (```json ... ```)
        2. Markdown headers (## Executive Summary, ## TL;DR, etc.)
        3. Key-value patterns
        """
        extracted = {}
        
        # Strategy 1: Extract JSON blocks (preferred)
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        for json_str in json_matches:
            assert json_str.strip().startswith("{"), "JSON must be object"
            parsed = json.loads(json_str)
            extracted.update(parsed)
        
        # Strategy 2: Extract from markdown headers
        # Executive Summary
        exec_match = re.search(
            r'(?:##\s*)?Executive\s+Summary[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if exec_match and 'executive_summary' not in extracted:
            summary = exec_match.group(1).strip()
            # Truncate to 600 chars if needed
            if len(summary) > 600:
                summary = summary[:597] + "..."
            extracted['executive_summary'] = summary
        
        # TL;DR
        tldr_patterns = [
            r'(?:##\s*)?TL;DR[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
            r'(?:##\s*)?TLDR[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
            r'(?:##\s*)?Summary[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
        ]
        for pattern in tldr_patterns:
            tldr_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if tldr_match and 'tldr' not in extracted:
                tldr = tldr_match.group(1).strip()
                # Truncate to 300 chars if needed
                if len(tldr) > 300:
                    tldr = tldr[:297] + "..."
                extracted['tldr'] = tldr
                break
        
        # Top PRO Drivers
        pro_drivers_match = re.search(
            r'(?:##\s*)?Top\s+PRO\s+Drivers[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if pro_drivers_match and 'top_pro_drivers' not in extracted:
            drivers_text = pro_drivers_match.group(1).strip()
            drivers = self._parse_drivers(drivers_text, "pro")
            if drivers:
                extracted['top_pro_drivers'] = drivers
        
        # Top CON Drivers
        con_drivers_match = re.search(
            r'(?:##\s*)?Top\s+CON\s+Drivers[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if con_drivers_match and 'top_con_drivers' not in extracted:
            drivers_text = con_drivers_match.group(1).strip()
            drivers = self._parse_drivers(drivers_text, "con")
            if drivers:
                extracted['top_con_drivers'] = drivers
        
        # Arbitrage Summary
        arb_match = re.search(
            r'(?:##\s*)?Arbitrage\s+Summary[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if arb_match and 'arbitrage_summary' not in extracted:
            extracted['arbitrage_summary'] = arb_match.group(1).strip()
        
        # Next Steps
        steps_match = re.search(
            r'(?:##\s*)?Next\s+Steps[:\-]?\s*\n(.*?)(?=\n##|\n#|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if steps_match and 'next_steps' not in extracted:
            steps_text = steps_match.group(1).strip()
            items = re.findall(r'[-*•]\s*(.+?)(?=\n[-*•]|\n\n|\Z)', steps_text, re.MULTILINE)
            if not items:
                items = re.findall(r'\d+[\.\)]\s*(.+?)(?=\n\d+|\n\n|\Z)', steps_text, re.MULTILINE)
            if items:
                extracted['next_steps'] = [item.strip() for item in items]
        
        # Strategy 3: Key-value patterns
        if 'executive_summary' not in extracted:
            exec_kv = re.search(
                r'(?:Executive\s+Summary|Summary)[:\-]\s*(.+?)(?=\n(?:TL;DR|Key|##)|\Z)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if exec_kv:
                summary = exec_kv.group(1).strip()
                if len(summary) > 600:
                    summary = summary[:597] + "..."
                extracted['executive_summary'] = summary
        
        if 'tldr' not in extracted:
            tldr_kv = re.search(
                r'TL;DR[:\-]\s*(.+?)(?=\n(?:Executive|Key|##)|\Z)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if tldr_kv:
                tldr = tldr_kv.group(1).strip()
                if len(tldr) > 300:
                    tldr = tldr[:297] + "..."
                extracted['tldr'] = tldr
        
        return extracted
    
    def _parse_drivers(self, text: str, direction: str) -> List[Dict[str, str]]:
        """Parse driver text into list of TopDriver dicts"""
        drivers = []
        # Try to parse numbered or bulleted list
        items = re.findall(r'[-*•]\s*(.+?)(?=\n[-*•]|\n\n|\Z)', text, re.MULTILINE)
        if not items:
            items = re.findall(r'\d+[\.\)]\s*(.+?)(?=\n\d+|\n\n|\Z)', text, re.MULTILINE)
        
        for item in items[:3]:  # Limit to 3
            item = item.strip()
            # Try to extract strength if mentioned
            strength = "moderate"
            if "strong" in item.lower():
                strength = "strong"
            elif "weak" in item.lower():
                strength = "weak"
            
            # Truncate summary to 200 chars
            summary = item[:197] + "..." if len(item) > 200 else item
            
            drivers.append({
                "direction": direction,
                "summary": summary,
                "strength": strength
            })
        
        return drivers

    async def is_task_complete(self, state: AgentState) -> bool:
        """Check if report is complete"""
        # Ensure intermediate_results exists
        if 'intermediate_results' not in state:
            state['intermediate_results'] = {}
        
        results = state['intermediate_results']
        required_keys = ['executive_summary', 'tldr', 'top_pro_drivers', 'top_con_drivers', 'arbitrage_summary']
        
        # If missing keys, try to extract from LLM response
        if not all(key in results for key in required_keys):
            messages = state.get('messages', [])
            if messages:
                # Check all recent AI messages (last 5) for extractable content
                for message in reversed(messages[-5:]):
                    if isinstance(message, AIMessage):
                        text = self._message_text(message)
                        if text:
                            self.logger.info("Extracting structured data from LLM text response")
                            extracted = self._extract_from_text(text)
                            
                            # Update intermediate_results with extracted data
                            if extracted:
                                state['intermediate_results'].update(extracted)
                                results = state['intermediate_results']
                                self.logger.info(
                                    f"Extracted keys: {list(extracted.keys())}"
                                )
                                # Break after first successful extraction
                                break
        
        # Validate required keys exist and have content
        has_all_keys = all(key in results for key in required_keys)
        if has_all_keys:
            # Validate content is not empty and meets length requirements
            exec_summary = results.get('executive_summary', '').strip()
            tldr = results.get('tldr', '').strip()
            top_pro = results.get('top_pro_drivers', [])
            top_con = results.get('top_con_drivers', [])
            arb_summary = results.get('arbitrage_summary', '')
            
            # Truncate executive_summary if too long
            if len(exec_summary) > 600:
                exec_summary = exec_summary[:597] + "..."
                results['executive_summary'] = exec_summary
            
            # Truncate tldr if too long
            if len(tldr) > 300:
                tldr = tldr[:297] + "..."
                results['tldr'] = tldr
            
            has_content = (
                exec_summary and
                len(exec_summary) >= 200 and  # At least 200 chars for executive summary
                len(exec_summary) <= 600 and  # Max 600 chars
                tldr and
                len(tldr) >= 10 and  # At least 10 chars for TL;DR
                len(tldr) <= 300 and  # Max 300 chars
                isinstance(top_pro, list) and
                len(top_pro) >= 3 and
                isinstance(top_con, list) and
                len(top_con) >= 3 and
                arb_summary and
                isinstance(arb_summary, str)
            )
            
            if has_content:
                self.logger.info(
                    f"Report complete: exec_summary={len(exec_summary)} chars, "
                    f"tldr={len(tldr)} chars, pro_drivers={len(top_pro)}, con_drivers={len(top_con)}"
                )
            else:
                self.logger.warning(
                    f"Report incomplete: exec_summary={len(exec_summary)} chars (need 200-600), "
                    f"tldr={len(tldr)} chars (need <=300), "
                    f"pro_drivers={len(top_pro) if isinstance(top_pro, list) else 0} (need 3), "
                    f"con_drivers={len(top_con) if isinstance(top_con, list) else 0} (need 3), "
                    f"arb_summary={'present' if arb_summary else 'missing'}"
                )
            
            return has_content
        
        return False

    async def extract_final_output(self, state: AgentState) -> ReporterOutput:
        """Extract ReporterOutput from state"""
        results = state.get('intermediate_results', {})
        task_input = state.get('task_input', {})
        
        # Get analyst output to extract p_bayesian and confidence_interval
        analyst_output = task_input.get('analyst_output', {})
        p_bayesian = analyst_output.get('p_bayesian', 0.5)
        p_low = analyst_output.get('p_bayesian_low', max(0.0, p_bayesian - 0.05))
        p_high = analyst_output.get('p_bayesian_high', min(1.0, p_bayesian + 0.05))
        confidence_interval = [p_low, p_high]
        
        # Extract top drivers from evidence_summary if not in results
        top_pro_drivers = results.get('top_pro_drivers', [])
        top_con_drivers = results.get('top_con_drivers', [])
        
        if not top_pro_drivers or not top_con_drivers:
            # Extract from evidence_summary
            evidence_summary = analyst_output.get('evidence_summary', [])
            if evidence_summary:
                # Sort by LLR magnitude
                pro_evidence = sorted(
                    [e for e in evidence_summary if e.get('LLR', 0) > 0],
                    key=lambda x: abs(x.get('LLR', 0)),
                    reverse=True
                )[:3]
                con_evidence = sorted(
                    [e for e in evidence_summary if e.get('LLR', 0) < 0],
                    key=lambda x: abs(x.get('LLR', 0)),
                    reverse=True
                )[:3]
                
                if not top_pro_drivers and pro_evidence:
                    top_pro_drivers = [
                        {
                            "direction": "pro",
                            "summary": f"Evidence ID {e.get('id', 'unknown')[:150]}",
                            "strength": "strong" if abs(e.get('LLR', 0)) > 1.0 else "moderate"
                        }
                        for e in pro_evidence[:3]
                    ]
                
                if not top_con_drivers and con_evidence:
                    top_con_drivers = [
                        {
                            "direction": "con",
                            "summary": f"Evidence ID {e.get('id', 'unknown')[:150]}",
                            "strength": "strong" if abs(e.get('LLR', 0)) > 1.0 else "moderate"
                        }
                        for e in con_evidence[:3]
                    ]
        
        # Ensure we have at least 3 drivers each
        while len(top_pro_drivers) < 3:
            top_pro_drivers.append({
                "direction": "pro",
                "summary": "Additional evidence supporting YES outcome",
                "strength": "moderate"
            })
        
        while len(top_con_drivers) < 3:
            top_con_drivers.append({
                "direction": "con",
                "summary": "Additional evidence supporting NO outcome",
                "strength": "moderate"
            })
        
        # Get arbitrage summary
        arbitrage_opportunities = task_input.get('arbitrage_opportunities', [])
        arbitrage_summary = results.get('arbitrage_summary', '')
        
        # Include threshold-specific analysis if available
        threshold_probabilities = analyst_output.get('threshold_probabilities', {})
        if threshold_probabilities and not arbitrage_summary:
            threshold_lines = []
            for threshold, data in sorted(threshold_probabilities.items()):
                p_threshold = data.get('p_bayesian', 0.5)
                market_price_threshold = data.get('market_price', 0.5)
                edge = data.get('edge', 0.0)
                threshold_lines.append(
                    f"{threshold}%+ threshold: p_bayesian={p_threshold:.1%} vs market={market_price_threshold:.1%}, edge={edge:+.1%}"
                )
            
            if threshold_lines:
                arbitrage_summary = f"Threshold analysis: {'; '.join(threshold_lines)}"
        
        if not arbitrage_summary:
            if arbitrage_opportunities:
                arbitrage_summary = f"Found {len(arbitrage_opportunities)} arbitrage opportunities"
            else:
                arbitrage_summary = "No arbitrage opportunities detected"
        
        # Truncate executive_summary to 600 chars
        exec_summary = results.get('executive_summary', '')
        if len(exec_summary) > 600:
            exec_summary = exec_summary[:597] + "..."
        
        # Truncate tldr to 300 chars
        tldr = results.get('tldr', '')
        if len(tldr) > 300:
            tldr = tldr[:297] + "..."
        
        # Create full_json package
        full_json = {
            "planner_output": task_input.get('planner_output', {}),
            "researcher_output": task_input.get('researcher_output', {}),
            "critic_output": task_input.get('critic_output', {}),
            "analyst_output": analyst_output,
            "arbitrage_opportunities": arbitrage_opportunities,
            "threshold_probabilities": threshold_probabilities,  # Include threshold probabilities
        }
        

        
        output = ReporterOutput(
            market_question=task_input.get('market_question', ''),
            p_bayesian=p_bayesian,
            confidence_interval=confidence_interval,
            top_pro_drivers=[TopDriver(**d) for d in top_pro_drivers[:3]],
            top_con_drivers=[TopDriver(**d) for d in top_con_drivers[:3]],
            arbitrage_summary=arbitrage_summary,
            executive_summary=exec_summary,
            tldr=tldr,
            full_json=full_json,
            next_steps=results.get('next_steps', []),
        )

        self.logger.info(f"📤 Report generated: exec_summary={len(output.executive_summary)} chars, p_bayesian={output.p_bayesian:.2%}")
        return output

    async def generate_report(
        self,
        market_question: str,
        planner_output: Dict[str, Any],
        researcher_output: Dict[str, Any],
        critic_output: Dict[str, Any],
        analyst_output: Dict[str, Any],
        arbitrage_opportunities: List[Dict[str, Any]],
        timestamp: str,
        workflow_id: str
    ) -> ReporterOutput:
        """
        Generate comprehensive report autonomously

        Args:
            market_question: Market question
            planner_output: Planner results
            researcher_output: Research results
            critic_output: Critique results
            analyst_output: Analysis results
            arbitrage_opportunities: Arbitrage opportunities
            timestamp: Workflow timestamp
            workflow_id: Workflow ID

        Returns:
            ReporterOutput with complete report
        """
        return await self.run(
            task_description="Generate comprehensive market analysis report",
            task_input={
                'market_question': market_question,
                'planner_output': planner_output,
                'researcher_output': researcher_output,
                'critic_output': critic_output,
                'analyst_output': analyst_output,
                'arbitrage_opportunities': arbitrage_opportunities,
                'timestamp': timestamp,
                'workflow_id': workflow_id
            }
        )
