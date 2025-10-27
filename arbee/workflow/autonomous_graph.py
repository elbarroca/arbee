"""
Autonomous LangGraph Workflow for POLYSEER
Uses autonomous reasoning agents with tool use, memory, and iterative loops
"""
from typing import Dict, Any, List
import logging
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from arbee.models.schemas import WorkflowState
from arbee.agents.autonomous_planner import AutonomousPlannerAgent
from arbee.agents.autonomous_researcher import run_parallel_autonomous_research
from arbee.agents.autonomous_critic import AutonomousCriticAgent
from arbee.agents.autonomous_analyst import AutonomousAnalystAgent
from arbee.agents.autonomous_arbitrage import AutonomousArbitrageAgent
from arbee.agents.autonomous_reporter import AutonomousReporterAgent
from arbee.utils.memory import get_memory_manager, MemoryConfig
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# AUTONOMOUS AGENT NODE FUNCTIONS
async def autonomous_planner_node(
    state: Dict[str, Any],
    store: BaseStore = None
) -> Dict[str, Any]:
    """
    Autonomous Planner Node
    Agent reasons iteratively about research plan with memory and validation
    """
    logger.info("🤔 === AUTONOMOUS PLANNER NODE ===")

    planner_cfg = (state.get('context') or {}).get('planner_agent', {})

    # Enhanced: Better default parameters to prevent infinite loops
    agent = AutonomousPlannerAgent(
        store=store,
        max_iterations=planner_cfg.get('max_iterations', 10),  # Reduced from 15
        min_subclaims=planner_cfg.get('min_subclaims', 4),
        max_subclaims=planner_cfg.get('max_subclaims', 10),
        auto_extend_iterations=planner_cfg.get('auto_extend_iterations', True),
        iteration_extension=planner_cfg.get('iteration_extension', 5),
        max_iteration_cap=planner_cfg.get('max_iteration_cap', 40),  # Reduced from 60
        recursion_limit=planner_cfg.get('recursion_limit', 100)  # Explicit limit with headroom
    )

    try:
        # Run autonomous planning
        result = await agent.plan(
            market_question=state['market_question'],
            market_url=state.get('market_url', ''),
            market_slug=state.get('market_slug', ''),
            context=state.get('context', {}),
            max_iterations=planner_cfg.get('run_max_iterations')
        )

        logger.info(
            f"✅ Planner completed: {len(result.subclaims)} subclaims, "
            f"prior={result.p0_prior:.2%}"
        )

        # Store learnings in memory
        if store:
            memory = get_memory_manager(store=store)
            await memory.store_episode_memory(
                episode_id=state.get('workflow_id', ''),
                market_question=state['market_question'],
                memory_type="planning_strategy",
                content={
                    "prior": result.p0_prior,
                    "subclaim_count": len(result.subclaims),
                    "balance": {
                        "pro": sum(1 for sc in result.subclaims if sc.direction == "pro"),
                        "con": sum(1 for sc in result.subclaims if sc.direction == "con")
                    }
                },
                effectiveness=0.9  # Assume good if validation passed
            )

        # Get agent stats
        stats = agent.get_stats()

        trace_details = {
            'subclaim_count': len(result.subclaims),
            'search_seed_counts': {
                'pro': len(result.search_seeds.pro),
                'con': len(result.search_seeds.con),
                'general': len(result.search_seeds.general)
            },
            'prior': result.p0_prior,
            'iterations': stats.get('average_iterations', 0),
            'tool_calls': stats.get('total_tool_calls', 0)
        }

        return {
            'planner_output': result,
            'p0_prior': result.p0_prior,
            'search_seeds': result.search_seeds,
            'subclaims': result.subclaims,
            'messages': [
                ("assistant", f"Autonomous Planner: {trace_details['subclaim_count']} subclaims, "
                 f"{trace_details['iterations']:.1f} iterations")
            ],
            'agent_traces': [{
                'agent_name': 'AutonomousPlannerAgent',
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'autonomous_planning_completed',
                'details': trace_details
            }]
        }

    except Exception as e:
        logger.error(f"❌ Autonomous Planner failed: {e}")
        raise


async def autonomous_researcher_node(
    state: Dict[str, Any],
    store: BaseStore = None
) -> Dict[str, Any]:
    """
    Autonomous Researcher Node (Parallel Execution)
    Three autonomous agents reason independently about research strategy
    """
    logger.info("🔍 === AUTONOMOUS RESEARCHER NODE (PARALLEL) ===")

    planner_output = state['planner_output']
    search_seeds = planner_output.search_seeds
    subclaims = [sc.model_dump() for sc in planner_output.subclaims]

    try:
        # Run all autonomous researchers in parallel
        results = await run_parallel_autonomous_research(
            search_seeds_pro=search_seeds.pro,
            search_seeds_con=search_seeds.con,
            search_seeds_general=search_seeds.general,
            subclaims=subclaims,
            market_question=state['market_question'],
            store=store,
            min_evidence_items=5,
            max_search_attempts=10,
            max_iterations=15
        )

        # Combine all evidence
        all_evidence = []
        direction_counts = {}
        for direction in ['pro', 'con', 'general']:
            items = results[direction].evidence_items
            all_evidence.extend(items)
            direction_counts[direction] = len(items)

        logger.info(
            f"✅ Autonomous Researchers completed: {len(all_evidence)} total evidence items "
            f"(PRO: {direction_counts['pro']}, "
            f"CON: {direction_counts['con']}, "
            f"GENERAL: {direction_counts['general']})"
        )

        # Store successful research strategies in memory
        if store and len(all_evidence) >= 10:
            memory = get_memory_manager(store=store)
            await memory.store_episode_memory(
                episode_id=state.get('workflow_id', ''),
                market_question=state['market_question'],
                memory_type="research_strategy",
                content={
                    "search_seeds_used": {
                        "pro": search_seeds.pro,
                        "con": search_seeds.con,
                        "general": search_seeds.general
                    },
                    "evidence_counts": direction_counts,
                    "total_evidence": len(all_evidence)
                },
                effectiveness=min(1.0, len(all_evidence) / 15)  # Scale by target
            )

        return {
            'researcher_output': results,
            'all_evidence': all_evidence,
            'messages': [
                ("assistant", f"Autonomous Researchers: {len(all_evidence)} evidence items "
                 f"({direction_counts['pro']}P, {direction_counts['con']}C, "
                 f"{direction_counts['general']}G)")
            ],
            'agent_traces': [{
                'agent_name': 'AutonomousResearcherAgents',
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'autonomous_research_completed',
                'details': {
                    'total_items': len(all_evidence),
                    'direction_counts': direction_counts
                }
            }]
        }

    except Exception as e:
        logger.error(f"❌ Autonomous Researchers failed: {e}")
        raise


async def autonomous_critic_node(
    state: Dict[str, Any],
    store: BaseStore = None
) -> Dict[str, Any]:
    """
    Autonomous Critic Node
    Agent reasons about evidence quality and correlation iteratively
    """
    logger.info("🔬 === AUTONOMOUS CRITIC NODE ===")

    agent = AutonomousCriticAgent(
        store=store,
        max_iterations=15,
        min_correlation_check_items=3
    )

    try:
        result = await agent.critique(
            evidence_items=state['all_evidence'],
            planner_output=state['planner_output'].model_dump(),
            market_question=state['market_question']
        )

        logger.info(
            f"✅ Autonomous Critic completed: {len(result.correlation_warnings)} warnings, "
            f"{len(result.missing_topics)} gaps"
        )

        stats = agent.get_stats()

        trace_payload = {
            'duplicate_clusters': len(result.duplicate_clusters),
            'missing_topics': len(result.missing_topics),
            'over_represented_sources': len(result.over_represented_sources),
            'iterations': stats.get('average_iterations', 0)
        }

        return {
            'critic_output': result,
            'messages': [
                ("assistant", f"Autonomous Critic: {trace_payload['duplicate_clusters']} "
                 f"correlations, {trace_payload['missing_topics']} gaps")
            ],
            'agent_traces': [{
                'agent_name': 'AutonomousCriticAgent',
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'autonomous_critique_completed',
                'details': trace_payload
            }]
        }

    except Exception as e:
        logger.error(f"❌ Autonomous Critic failed: {e}")
        raise


async def autonomous_analyst_node(
    state: Dict[str, Any],
    store: BaseStore = None
) -> Dict[str, Any]:
    """
    Autonomous Analyst Node
    Agent reasons about Bayesian aggregation with validation and sensitivity
    """
    logger.info("🧮 === AUTONOMOUS ANALYST NODE ===")

    agent = AutonomousAnalystAgent(
        store=store,
        max_iterations=15,
        max_sensitivity_range=0.3
    )

    try:
        result = await agent.analyze(
            prior_p=state['p0_prior'],
            evidence_items=state['all_evidence'],
            critic_output=state['critic_output'],
            market_question=state['market_question']
        )

        logger.info(f"✅ Autonomous Analyst completed: p_bayesian={result.p_bayesian:.2%}")

        stats = agent.get_stats()

        trace_payload = {
            'p_bayesian': result.p_bayesian,
            'evidence_items': len(result.evidence_summary),
            'sensitivity_scenarios': len(result.sensitivity_analysis),
            'iterations': stats.get('average_iterations', 0)
        }

        return {
            'analyst_output': result,
            'p_bayesian': result.p_bayesian,
            'messages': [
                ("assistant", f"Autonomous Analyst: p={result.p_bayesian:.2%} "
                 f"(from {result.p0:.2%})")
            ],
            'agent_traces': [{
                'agent_name': 'AutonomousAnalystAgent',
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'autonomous_analysis_completed',
                'details': trace_payload
            }]
        }

    except Exception as e:
        logger.error(f"❌ Autonomous Analyst failed: {e}")
        raise


async def autonomous_arbitrage_node(
    state: Dict[str, Any],
    store: BaseStore = None
) -> Dict[str, Any]:
    """
    Autonomous Arbitrage Node
    Agent reasons about mispricing opportunities
    """
    logger.info("💹 === AUTONOMOUS ARBITRAGE NODE ===")

    agent = AutonomousArbitrageAgent(
        store=store,
        max_iterations=10,
        min_edge_threshold=state.get('min_edge_threshold', settings.MIN_EDGE_THRESHOLD)
    )

    try:
        opportunities = await agent.detect_arbitrage(
            p_bayesian=state['p_bayesian'],
            market_slug=state.get('market_slug', ''),
            market_question=state['market_question'],
            providers=state.get('providers', ['polymarket', 'kalshi']),
            bankroll=state.get('bankroll', settings.DEFAULT_BANKROLL),
            max_kelly=state.get('max_kelly', settings.MAX_KELLY_FRACTION),
            min_edge_threshold=state.get('min_edge_threshold', settings.MIN_EDGE_THRESHOLD)
        )

        logger.info(f"✅ Autonomous Arbitrage completed: {len(opportunities)} opportunities")

        trace_payload = {
            'opportunity_count': len(opportunities),
            'providers_checked': state.get('providers', [])
        }

        return {
            'arbitrage_output': opportunities,
            'messages': [
                ("assistant", f"Autonomous Arbitrage: {len(opportunities)} opportunities")
            ],
            'agent_traces': [{
                'agent_name': 'AutonomousArbitrageAgent',
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'autonomous_arbitrage_completed',
                'details': trace_payload
            }]
        }

    except Exception as e:
        logger.error(f"⚠️ Autonomous Arbitrage failed: {e}")
        logger.warning("Continuing without arbitrage analysis")
        return {'arbitrage_output': []}


async def autonomous_reporter_node(
    state: Dict[str, Any],
    store: BaseStore = None
) -> Dict[str, Any]:
    """
    Autonomous Reporter Node
    Agent reasons about report completeness and quality
    """
    logger.info("📝 === AUTONOMOUS REPORTER NODE ===")

    agent = AutonomousReporterAgent(
        store=store,
        max_iterations=10
    )

    try:
        result = await agent.generate_report(
            market_question=state['market_question'],
            planner_output=state['planner_output'].model_dump(),
            researcher_output={
                k: v.model_dump() for k, v in state['researcher_output'].items()
            },
            critic_output=state['critic_output'].model_dump(),
            analyst_output=state['analyst_output'].model_dump(),
            arbitrage_opportunities=[opp.model_dump() for opp in state['arbitrage_output']],
            timestamp=state.get('timestamp', datetime.now().isoformat()),
            workflow_id=state.get('workflow_id', '')
        )

        logger.info("✅ Autonomous Reporter completed: Final report generated")

        return {
            'reporter_output': result,
            'messages': [
                ("assistant", "Autonomous Reporter: Final report complete")
            ],
            'agent_traces': [{
                'agent_name': 'AutonomousReporterAgent',
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'autonomous_report_completed',
                'details': {
                    'summary_length': len(result.executive_summary or "")
                }
            }]
        }

    except Exception as e:
        logger.error(f"❌ Autonomous Reporter failed: {e}")
        raise


# ============================================================================
# AUTONOMOUS WORKFLOW CONSTRUCTION
# ============================================================================

def create_autonomous_polyseer_workflow(store: BaseStore = None) -> StateGraph:
    """
    Create the POLYSEER workflow with autonomous agents

    Each node now contains an agent that reasons iteratively using tools
    until it determines the task is complete.

    Workflow Structure:
    START → planner → researcher (parallel) → critic → analyst → arbitrage → reporter → END

    Returns:
        Compiled StateGraph with autonomous agents
    """
    logger.info("🚀 Creating AUTONOMOUS POLYSEER workflow")

    # Create graph with WorkflowState
    workflow = StateGraph(WorkflowState)

    # Add autonomous agent nodes (with store for memory)
    # Create async wrapper functions to properly handle async node calls
    async def planner_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_planner_node(state, store)

    async def researcher_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_researcher_node(state, store)

    async def critic_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_critic_node(state, store)

    async def analyst_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_analyst_node(state, store)

    async def arbitrage_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_arbitrage_node(state, store)

    async def reporter_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await autonomous_reporter_node(state, store)

    workflow.add_node("planner", planner_wrapper)
    workflow.add_node("researcher", researcher_wrapper)
    workflow.add_node("critic", critic_wrapper)
    workflow.add_node("analyst", analyst_wrapper)
    workflow.add_node("arbitrage", arbitrage_wrapper)
    workflow.add_node("reporter", reporter_wrapper)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "analyst")
    workflow.add_edge("analyst", "arbitrage")
    workflow.add_edge("arbitrage", "reporter")
    workflow.add_edge("reporter", END)

    # Compile with checkpointing and increased recursion limit
    # The recursion limit needs to be higher than max possible iterations across all agents
    # With 6 agents × max 20 iterations each + buffer = 150 is safe
    checkpointer = MemorySaver()
    app = workflow.compile(
        checkpointer=checkpointer,
        debug=False  # Set to True for detailed graph execution logs
    )

    logger.info("✅ Autonomous workflow compiled successfully with recursion_limit=150")

    return app


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def run_autonomous_workflow(
    market_question: str,
    market_url: str = "",
    market_slug: str = "",
    providers: List[str] = None,
    bankroll: float = None,
    max_kelly: float = None,
    min_edge_threshold: float = None,
    store: BaseStore = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the complete autonomous POLYSEER workflow

    Args:
        market_question: Prediction market question to analyze
        market_url: URL to the market
        market_slug: Market identifier
        providers: List of platforms to check
        bankroll: Available capital
        max_kelly: Maximum Kelly fraction
        min_edge_threshold: Minimum edge to report
        store: LangGraph Store for memory
        **kwargs: Additional context

    Returns:
        Dict with all workflow outputs

    Example:
        >>> result = await run_autonomous_workflow(
        ...     market_question="Will Donald Trump win the 2024 US Presidential Election?",
        ...     providers=["polymarket", "kalshi"],
        ...     bankroll=10000.0
        ... )
        >>> print(result['reporter_output'].executive_summary)
    """
    workflow_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    logger.info(f"🚀 Starting AUTONOMOUS POLYSEER workflow {workflow_id}")
    logger.info(f"📊 Market Question: {market_question}")

    # Create autonomous workflow
    app = create_autonomous_polyseer_workflow(store=store)

    # Prepare initial state
    initial_state = {
        'workflow_id': workflow_id,
        'timestamp': timestamp,
        'market_question': market_question,
        'market_url': market_url,
        'market_slug': market_slug or market_question.lower().replace(' ', '-')[:50],
        'providers': providers or ['polymarket', 'kalshi'],
        'bankroll': bankroll or settings.DEFAULT_BANKROLL,
        'max_kelly': max_kelly or settings.MAX_KELLY_FRACTION,
        'min_edge_threshold': min_edge_threshold or settings.MIN_EDGE_THRESHOLD,
        'context': kwargs
    }

    # Execute workflow with increased recursion limit
    # Set high recursion limit to handle all agent iterations
    # 6 agents × 20 max iterations + overhead = 150
    config = {
        "configurable": {"thread_id": workflow_id},
        "recursion_limit": 150
    }

    try:
        final_state = await app.ainvoke(initial_state, config)

        logger.info(f"✅ Autonomous workflow {workflow_id} completed successfully")

        return final_state

    except Exception as e:
        logger.error(f"❌ Autonomous workflow {workflow_id} failed: {e}")
        raise
