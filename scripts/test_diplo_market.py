#!/usr/bin/env python3
"""
Test Script: AUTONOMOUS POLYSEER Deep Analysis on Diplo 5k Race Market
Based on market from Polymarket screenshot

Market Details:
- Question: "How fast will Diplo run 5k?"
- Sub-markets:
  - <23 minutes: 90% (Yes: 90¢, No: 11¢) - $6,354 vol
  - <22 minutes: 74% (Yes: 74¢, No: 27¢) - $4,947 vol
  - <21 minutes: 32% (Yes: 32¢, No: 69¢) - $22,995 vol
- Total volume: $34,296
- Event date: October 25, 2025
- Resolution: BibTag System official race results

UPDATED: Now uses autonomous reasoning agents with tool use and memory!
"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbee.workflow.autonomous_graph import run_autonomous_workflow
from config.settings import settings


async def main():
    """Run autonomous deep analysis on Diplo 5k market"""

    # Market from screenshot
    market_question = "Will Diplo run 5k in under 23 minutes?"
    market_slug = "diplo-5k-under-23-minutes"
    market_url = "https://polymarket.com/event/diplo-5k"

    print("\n" + "=" * 80)
    print("🤖 AUTONOMOUS POLYSEER DEEP ANALYSIS: Diplo 5k Race Market")
    print("=" * 80)
    print(f"\nMarket Question: {market_question}")
    print(f"Market URL: {market_url}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("🚀 Running AUTONOMOUS workflow with reasoning agents...")
    print("   Each agent will think, act, and validate iteratively")
    print("   This will take approximately 5-15 minutes\n")
    print("   Steps:")
    print("   [1/6] 🤔 Autonomous Planner: Reasoning about research strategy")
    print("         - Searches memory for similar markets")
    print("         - Validates prior with tools")
    print("         - Generates balanced subclaims")
    print("   [2/6] 🔍 Autonomous Researchers: Iterative evidence gathering")
    print("         - PRO/CON/GENERAL agents reason independently")
    print("         - Each tries multiple search strategies")
    print("         - Validates quality before completing")
    print("   [3/6] 🔬 Autonomous Critic: Correlation detection")
    print("         - Uses tools to find correlated evidence")
    print("         - Identifies coverage gaps")
    print("   [4/6] 🧮 Autonomous Analyst: Bayesian aggregation")
    print("         - Validates LLR calibration")
    print("         - Runs sensitivity analysis")
    print("   [5/6] 💹 Autonomous Arbitrage: Mispricing detection")
    print("         - Fetches market prices")
    print("         - Calculates edges and EVs")
    print("   [6/6] 📝 Autonomous Reporter: Report generation")
    print("         - Validates completeness")
    print("         - Generates final report\n")

    # Run autonomous deep analysis
    try:
        start_time = datetime.now()

        # Optional: Set up memory store for cross-workflow learning
        # from langgraph.store.memory import InMemoryStore
        # store = InMemoryStore()
        store = None  # Or configure Redis/PostgreSQL for production

        result = await run_autonomous_workflow(
            market_question=market_question,
            market_url=market_url,
            market_slug=market_slug,
            providers=["polymarket"],
            bankroll=settings.DEFAULT_BANKROLL,
            max_kelly=settings.MAX_KELLY_FRACTION,
            min_edge_threshold=settings.MIN_EDGE_THRESHOLD,
            store=store  # Optional: for memory/learning
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("✅ AUTONOMOUS ANALYSIS COMPLETE")
        print("=" * 80)

        # Display key results
        print(f"\n📊 KEY RESULTS:")

        # Get analyst output for confidence intervals
        analyst_output = result.get('analyst_output')
        if analyst_output:
            p_bayesian = analyst_output.p_bayesian
            p_low = getattr(analyst_output, 'p_bayesian_low', None)
            p_high = getattr(analyst_output, 'p_bayesian_high', None)
            conf_level = getattr(analyst_output, 'confidence_level', 0.80)

            print(f"   Bayesian Probability: {p_bayesian:.2%}")
            if p_low is not None and p_high is not None:
                print(f"   Confidence Interval ({conf_level:.0%}): [{p_low:.2%} - {p_high:.2%}]")
                range_width = p_high - p_low
                print(f"   Uncertainty Range: ±{range_width/2:.1%}")
        else:
            print(f"   Bayesian Probability (p_bayesian): {result.get('p_bayesian', 0):.2%}")

        print(f"   Execution Time: {execution_time:.1f}s")

        # Display workflow outputs (autonomous agents)
        print(f"\n🤖 AUTONOMOUS AGENT OUTPUTS:")

        # Planner output
        if 'planner_output' in result and result['planner_output']:
            planner = result['planner_output']
            print(f"\n📋 AUTONOMOUS PLANNER:")
            print(f"   Prior (p0): {planner.p0_prior:.2%}")
            print(f"   Justification: {planner.prior_justification[:100]}...")
            print(f"   Subclaims generated: {len(planner.subclaims)}")
            print(f"   Search seeds: PRO={len(planner.search_seeds.pro)}, "
                  f"CON={len(planner.search_seeds.con)}, "
                  f"GENERAL={len(planner.search_seeds.general)}")

            # Show agent reasoning
            if hasattr(planner, 'reasoning_trace') and planner.reasoning_trace:
                print(f"   Reasoning trace: {planner.reasoning_trace[:150]}...")

        # Researcher output
        if 'researcher_output' in result and result['researcher_output']:
            researcher = result['researcher_output']
            total_evidence = sum(len(r.evidence_items) for r in researcher.values())
            print(f"\n🔍 AUTONOMOUS RESEARCHERS (Parallel):")
            print(f"   Total evidence items: {total_evidence}")
            for direction in ['pro', 'con', 'general']:
                if direction in researcher:
                    count = len(researcher[direction].evidence_items)
                    print(f"   {direction.upper()}: {count} items")
                    # Show sample evidence
                    if count > 0:
                        sample = researcher[direction].evidence_items[0]
                        print(f"      Sample: {sample.title[:60]}... (LLR: {sample.estimated_LLR:+.2f})")

        # Critic output
        if 'critic_output' in result and result['critic_output']:
            critic = result['critic_output']
            print(f"\n🔬 AUTONOMOUS CRITIC:")
            print(f"   Correlation warnings: {len(critic.correlation_warnings)}")
            print(f"   Duplicate clusters: {len(critic.duplicate_clusters)}")
            print(f"   Missing topics: {len(critic.missing_topics)}")
            print(f"   Over-represented sources: {len(critic.over_represented_sources)}")
            print(f"   Follow-up seeds suggested: {len(critic.follow_up_search_seeds)}")

        # Analyst output
        if 'analyst_output' in result and result['analyst_output']:
            analyst = result['analyst_output']
            print(f"\n🧮 AUTONOMOUS ANALYST:")
            print(f"   Prior p0: {analyst.p0:.2%}")
            print(f"   Posterior p_bayesian: {analyst.p_bayesian:.2%}")

            # Show confidence interval if available
            if hasattr(analyst, 'p_bayesian_low') and hasattr(analyst, 'p_bayesian_high'):
                print(f"   Confidence Interval ({analyst.confidence_level:.0%}): "
                      f"[{analyst.p_bayesian_low:.2%} - {analyst.p_bayesian_high:.2%}]")

            print(f"   Change from prior: {(analyst.p_bayesian - analyst.p0):+.2%}")
            print(f"   Evidence items analyzed: {len(analyst.evidence_summary)}")
            print(f"   Calculation steps: {len(analyst.calculation_steps)}")
            print(f"   Sensitivity scenarios: {len(analyst.sensitivity_analysis)}")

            # Show sensitivity range
            if analyst.sensitivity_analysis:
                probs = [s.p for s in analyst.sensitivity_analysis]
                print(f"   Sensitivity range: {min(probs):.2%} to {max(probs):.2%}")

        # Arbitrage output
        if 'arbitrage_output' in result and result['arbitrage_output']:
            opportunities = result['arbitrage_output']
            print(f"\n💹 AUTONOMOUS ARBITRAGE:")
            print(f"   Opportunities found: {len(opportunities)}")
            if opportunities:
                for i, opp in enumerate(opportunities[:3], 1):
                    print(f"   [{i}] {opp.provider}: "
                          f"Edge={opp.edge:.2%}, "
                          f"EV={opp.expected_value_per_dollar:.2%}, "
                          f"Kelly={opp.kelly_fraction:.2%}")

        # Reporter output
        if 'reporter_output' in result and result['reporter_output']:
            reporter = result['reporter_output']
            print(f"\n📝 AUTONOMOUS REPORTER:")
            print(f"   Executive summary: {len(reporter.executive_summary)} chars")
            print(f"   TL;DR: {reporter.tldr[:100]}...")
            print(f"   Key findings: {len(reporter.key_findings)}")

        # Show agent traces (reasoning transparency)
        if 'agent_traces' in result:
            print(f"\n🔍 AGENT ACTIVITY LOG:")
            for trace in result['agent_traces'][-5:]:  # Last 5 traces
                agent_name = trace.get('agent_name', 'Unknown')
                action = trace.get('action', 'unknown')
                details = trace.get('details', {})
                print(f"   {agent_name}: {action}")
                if 'iterations' in details:
                    print(f"      Iterations: {details['iterations']:.1f}")

        # Save outputs
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Save JSON (convert Pydantic models to dicts)
        json_result = {}
        for key, value in result.items():
            if hasattr(value, 'model_dump'):
                json_result[key] = value.model_dump()
            elif isinstance(value, list) and value and hasattr(value[0], 'model_dump'):
                json_result[key] = [v.model_dump() for v in value]
            elif isinstance(value, dict):
                json_result[key] = {
                    k: v.model_dump() if hasattr(v, 'model_dump') else v
                    for k, v in value.items()
                }
            else:
                json_result[key] = value

        json_result['execution_time_seconds'] = execution_time
        json_result['autonomous_workflow'] = True  # Flag for autonomous workflow

        json_path = reports_dir / f"{market_slug}_autonomous_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2, default=str)

        print(f"\n💾 AUTONOMOUS ANALYSIS OUTPUTS SAVED:")
        print(f"   JSON Report: {json_path}")

        # Save Markdown if reporter output exists
        if 'reporter_output' in result and result['reporter_output']:
            reporter = result['reporter_output']
            if hasattr(reporter, 'executive_summary') and reporter.executive_summary:
                md_path = reports_dir / f"{market_slug}_autonomous_analysis.md"
                with open(md_path, 'w') as f:
                    f.write(f"# 🤖 AUTONOMOUS POLYSEER Analysis: {market_question}\n\n")
                    f.write(f"**Workflow ID:** {result.get('workflow_id', 'N/A')}\n")
                    f.write(f"**Timestamp:** {result.get('timestamp', 'N/A')}\n")
                    f.write(f"**Execution Time:** {execution_time:.1f}s\n")
                    f.write(f"**Autonomous Agents:** Yes\n\n")

                    f.write(f"## Executive Summary\n\n")
                    f.write(reporter.executive_summary)

                    f.write(f"\n\n## TL;DR\n\n")
                    f.write(reporter.tldr if hasattr(reporter, 'tldr') else 'N/A')

                    f.write(f"\n\n## Key Findings\n\n")
                    if hasattr(reporter, 'key_findings'):
                        for i, finding in enumerate(reporter.key_findings, 1):
                            f.write(f"{i}. {finding}\n")

                    f.write("\n\n---\n\n")
                    f.write("**Generated by Autonomous Reasoning Agents**\n\n")
                    f.write("Each agent in this analysis used iterative reasoning with tool use:\n")
                    f.write("- 🤔 Planner: Searched memory, validated prior\n")
                    f.write("- 🔍 Researchers: Tried multiple search strategies\n")
                    f.write("- 🔬 Critic: Used correlation detection tools\n")
                    f.write("- 🧮 Analyst: Validated LLRs, ran sensitivity analysis\n")
                    f.write("- 💹 Arbitrage: Calculated edges and EVs\n")
                    f.write("- 📝 Reporter: Validated report completeness\n\n")
                    f.write("**NOT FINANCIAL ADVICE.** This is research only.\n")
                print(f"   Markdown Report: {md_path}")

        print(f"\n💡 Tip: Review the JSON file to see full agent reasoning traces and tool calls")

        print("\n" + "=" * 80)
        print("🤖 AUTONOMOUS ANALYSIS FEATURES:")
        print("=" * 80)
        print("✅ Iterative reasoning loops (agents think → act → observe)")
        print("✅ Tool use (15+ tools for search, analysis, validation)")
        print("✅ Memory integration (learns from similar cases)")
        print("✅ Self-validation (checks quality at every step)")
        print("✅ Transparency (full reasoning traces available)")
        print("\n" + "=" * 80)
        print("NOT FINANCIAL ADVICE. This is research only.")
        print("=" * 80 + "\n")

        return result

    except Exception as e:
        print(f"\n❌ ERROR in autonomous workflow: {e}\n")
        import traceback
        traceback.print_exc()
        print("\nNote: If you see tool import errors, ensure all tools are properly installed")
        print("Check: arbee/tools/ directory and dependencies\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
