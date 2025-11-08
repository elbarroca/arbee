"""
Rich Logging Module for POLYSEER Agent Workflows.

Provides sleek, detailed logging with rich formatting for agent operations,
reasoning steps, tool usage, memory interactions, and edge detection.
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich import box
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging

console = Console()
std_logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging levels for consistent styling."""
    INFO = "blue"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    DEBUG = "cyan"


class FormatterRegistry:
    """Centralized formatting utilities with lazy evaluation."""

    @staticmethod
    def format_text_preview(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Format text with length limits and preview suffix."""
        assert isinstance(text, str), "Text must be string"
        return text[:max_length] + suffix if len(text) > max_length else text

    @staticmethod
    def create_styled_table(columns: List[str], header_style: str = "bold cyan") -> Table:
        """Create consistently styled table with given columns."""
        table = Table(show_header=True, header_style=header_style, box=box.ROUNDED)
        for col in columns:
            table.add_column(col, style="cyan" if col != columns[-1] else "yellow")
        return table

    @staticmethod
    def create_panel(content: Union[str, Text], title: str, border_color: str = "blue") -> Panel:
        """Create consistently styled information panel."""
        return Panel(
            content,
            title=f"[bold {border_color}]{title}[/bold {border_color}]",
            border_style=border_color,
        )

    @staticmethod
    def create_header_grid() -> Table:
        """Create header grid layout."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(width=3)
        grid.add_column()
        return grid


class LogBuilder:
    """Builder patterns for complex logging operations."""

    def __init__(self, logger: "RichAgentLogger"):
        self.logger = logger

    def build_completion_status(self, complete: bool, reason: str = "") -> Text:
        """Build completion status text with conditional formatting."""
        status_text = Text()
        status_text.append(
            "✅ Task Complete" if complete else "⏳ Task In Progress",
            style="bold green" if complete else "bold yellow",
        )
        if reason:
            status_text.append(f"\n{reason}", style="dim")
        return status_text

    def build_iteration_summary(self, iteration: int, state: Dict[str, Any]) -> Table:
        """Build iteration summary table with dynamic metrics."""
        summary_table = FormatterRegistry.create_styled_table(["Metric", "Count"], "bold blue")
        summary_table.add_row("Iteration", str(iteration))

        metrics = {
            "tool_calls": len(state.get("tool_calls", [])),
            "reasoning_trace": len(state.get("reasoning_trace", [])),
            "memory_accessed": len(state.get("memory_accessed", [])),
            "evidence_items": len(state.get("evidence_items", [])),
            "edge_signals": len(state.get("edge_signals", [])),
            "errors": len(state.get("errors", [])),
        }

        for metric, count in metrics.items():
            if count > 0:
                summary_table.add_row(metric.replace("_", " ").title(), str(count))

        return summary_table


class RichAgentLogger:
    """
    Provides comprehensive, visually appealing logging for agent execution,
    reasoning processes, tool interactions, memory operations, and edge detection.

    Args:
        agent_name: Name of the agent for contextual logging
    """

    def __init__(self, agent_name: str):
        assert agent_name and isinstance(agent_name, str), "Agent name must be non-empty string"
        self.agent_name = agent_name
        self.console = console
        self.formatter = FormatterRegistry()
        self.builder = LogBuilder(self)
        self._execution_stats: Dict[str, Any] = {}
        # Also create standard logger for compatibility
        self.std_logger = logging.getLogger(f".agents.{agent_name}")

    def log_agent_start(
        self,
        task_description: str,
        input_info: Optional[Dict[str, Any]] = None,
        market_question: Optional[str] = None,
    ):
        """
        Log comprehensive agent initialization with task details and input parameters.

        Args:
            task_description: Description of the agent's task
            input_info: Optional dictionary of input parameters
            market_question: Optional market question being analyzed
        """
        assert task_description and isinstance(task_description, str), "Task description must be non-empty string"

        # Create and display header
        header = self.formatter.create_header_grid()
        header.add_row("🤖", f"[bold cyan]{self.agent_name}[/bold cyan]")
        header.add_row("📋", self.formatter.format_text_preview(task_description, 200))

        if market_question:
            header.add_row("📊", f"[green]Market:[/green] {self.formatter.format_text_preview(market_question, 150)}")

        self.console.print(self.formatter.create_panel(header, "Agent Execution", LogLevel.INFO.value))

        # Display input parameters with smart filtering
        if input_info:
            input_table = self.formatter.create_styled_table(["Parameter", "Value"], "bold magenta")
            filtered_params = {
                k: v
                for k, v in input_info.items()
                if k not in ["messages", "tool_calls"] and v is not None
            }

            for key, value in list(filtered_params.items())[:12]:
                input_table.add_row(key, self.formatter.format_text_preview(str(value), 80))

            self.console.print(
                self.formatter.create_panel(input_table, "Input Parameters", LogLevel.INFO.value)
            )

    def log_reasoning_step(
        self,
        iteration: int,
        thought: str,
        reasoning: str = "",
        intermediate_outputs: Optional[Dict[str, Any]] = None,
    ):
        """
        Log detailed reasoning step with thought process and intermediate results.

        Args:
            iteration: Current iteration number (1-based)
            thought: Agent's thought process text
            reasoning: Additional reasoning details
            intermediate_outputs: Optional dictionary of intermediate computation results
        """
        assert isinstance(iteration, int) and iteration > 0, "Iteration must be positive integer"
        assert thought and isinstance(thought, str), "Thought must be non-empty string"
        assert isinstance(reasoning, str), "Reasoning must be string"
        assert intermediate_outputs is None or isinstance(
            intermediate_outputs, dict
        ), "Intermediate outputs must be dict or None"

        step_text = Text()
        step_text.append(f"💭 [bold]Iteration {iteration}[/bold]\n", style="cyan")
        step_text.append(f"[bold]Thought:[/bold]\n{thought}\n", style="white")
        if reasoning:
            step_text.append(f"\n[bold]Reasoning:[/bold]\n{reasoning}\n", style="yellow")

        self.console.print(
            self.formatter.create_panel(step_text, f"Reasoning Step {iteration}", LogLevel.INFO.value)
        )

        if intermediate_outputs:
            output_table = self.formatter.create_styled_table(["Output Key", "Value"], "bold green")
            for key, value in list(intermediate_outputs.items())[:8]:
                output_table.add_row(key, self.formatter.format_text_preview(str(value), 100))
            self.console.print(
                self.formatter.create_panel(output_table, "Intermediate Results", LogLevel.SUCCESS.value)
            )

    def log_tool_call(self, tool_name: str, tool_input: Dict[str, Any], result: Any = None):
        """
        Log comprehensive tool execution with input, status, and detailed results.

        Args:
            tool_name: Name of the tool being executed
            tool_input: Dictionary of input parameters passed to the tool
            result: Optional result object from tool execution
        """
        assert tool_name and isinstance(tool_name, str), "Tool name must be non-empty string"
        assert isinstance(tool_input, dict), "Tool input must be dictionary"

        tool_table = self.formatter.create_styled_table(["Property", "Details"], "bold magenta")
        tool_table.add_row("Tool Name", f"[bold]{tool_name}[/bold]")
        tool_table.add_row(
            "Input",
            self.formatter.format_text_preview(str(tool_input), 200) if tool_input else "[dim]No input[/dim]",
        )

        if result is None:
            tool_table.add_row("Status", "⏳ Executing")
        elif isinstance(result, dict):
            success = result.get("success", True) if "success" in result else True
            tool_table.add_row("Status", "✅ Success" if success else "❌ Failed")

            # Extract common result fields with smart formatting
            result_fields = {
                "p_bayesian": lambda x: f"{x:.2%}",
                "edge": lambda x: f"{x:.2%}",
                "strength": lambda x: f"{x:.2f}",
                "confidence": lambda x: f"{x:.2f}",
                "evidence_count": lambda x: f"{x} items",
                "opportunities": lambda x: f"{len(x)} found" if isinstance(x, list) else str(x),
                "edge_signals": lambda x: f"{len(x)} signals" if isinstance(x, list) else str(x),
                "execution_time": lambda x: f"{x:.2f}s",
            }

            for field, formatter in result_fields.items():
                if field in result:
                    tool_table.add_row(field.replace("_", " ").title(), formatter(result[field]))

            # Show error if present
            if "error" in result:
                tool_table.add_row("Error", f"[red]{result['error']}[/red]")
        else:
            tool_table.add_row("Status", "✅ Success")
            tool_table.add_row("Result", self.formatter.format_text_preview(str(result), 300))

        self.console.print(self.formatter.create_panel(tool_table, "🔧 Tool Execution", LogLevel.INFO.value))

    def log_memory_access(self, operation: str, key: str, found: bool = False, data: Any = None):
        """
        Log memory access operations with enhanced detail.

        Args:
            operation: Type of memory operation (read/write/search)
            key: Memory key being accessed
            found: Whether the key was found in memory
            data: Optional data retrieved from memory
        """
        assert operation and isinstance(operation, str), "Operation must be non-empty string"
        assert key and isinstance(key, str), "Key must be non-empty string"

        memory_text = Text()
        memory_text.append(f"🧠 [bold]{operation.title()}[/bold]\n", style="yellow")
        memory_text.append(f"Key: [cyan]{key}[/cyan]\n", style="white")
        memory_text.append(f"Status: {'✅ Found' if found else '❌ Not Found'}", style="green" if found else "red")

        if found and data:
            if isinstance(data, list):
                memory_text.append(f"\nItems: {len(data)}", style="dim")
            else:
                memory_text.append(f"\nData: {self.formatter.format_text_preview(str(data), 150)}", style="dim")

        self.console.print(
            self.formatter.create_panel(memory_text, "Memory Access", LogLevel.WARNING.value)
        )

    def log_memory_query(self, query: str, results_count: int, namespace: str = "knowledge_base", query_time: Optional[float] = None):
        """
        Log memory query operation with results summary.
        
        Args:
            query: Query string used
            results_count: Number of results found
            namespace: Namespace queried
            query_time: Optional query execution time in seconds
        """
        assert isinstance(query, str), "Query must be string"
        assert isinstance(results_count, int), "Results count must be integer"
        
        query_table = self.formatter.create_styled_table(["Property", "Value"], "bold yellow")
        query_table.add_row("Query", self.formatter.format_text_preview(query, 100))
        query_table.add_row("Namespace", namespace)
        query_table.add_row("Results Found", f"[green]{results_count}[/green]")
        
        if query_time is not None:
            query_table.add_row("Query Time", f"{query_time:.3f}s")
        
        self.console.print(
            self.formatter.create_panel(query_table, "🧠 Memory Query", LogLevel.INFO.value)
        )
        self.std_logger.info(f"Memory query: '{query[:50]}...' found {results_count} results in {namespace}")

    def log_memory_store(self, operation: str, key: str, content_type: str = "unknown", success: bool = True):
        """
        Log memory store operation.
        
        Args:
            operation: Type of store operation (store_knowledge, store_episode_memory, etc.)
            key: Key/ID of stored item
            content_type: Type of content stored
            success: Whether operation succeeded
        """
        assert isinstance(operation, str), "Operation must be string"
        assert isinstance(key, str), "Key must be string"
        
        store_text = Text()
        store_text.append(f"💾 [bold]{operation.replace('_', ' ').title()}[/bold]\n", style="cyan")
        store_text.append(f"Key: [cyan]{key}[/cyan]\n", style="white")
        store_text.append(f"Content Type: [yellow]{content_type}[/yellow]\n", style="white")
        store_text.append(f"Status: {'✅ Success' if success else '❌ Failed'}", style="green" if success else "red")
        
        self.console.print(
            self.formatter.create_panel(store_text, "Memory Store", LogLevel.INFO.value)
        )
        self.std_logger.info(f"Memory store: {operation} key={key} type={content_type} success={success}")

    def log_edge_detection(
        self,
        edge_type: str,
        strength: float,
        confidence: float,
        evidence: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log edge detection signal with visualization.

        Args:
            edge_type: Type of edge detected
            strength: Edge strength (0-1)
            confidence: Confidence in detection (0-1)
            evidence: List of evidence strings
            metadata: Optional additional metadata
        """
        edge_table = self.formatter.create_styled_table(["Property", "Value"], "bold green")
        edge_table.add_row("Edge Type", f"[bold]{edge_type}[/bold]")
        edge_table.add_row("Strength", f"[green]{strength:.2f}[/green]")
        edge_table.add_row("Confidence", f"[cyan]{confidence:.2f}[/cyan]")
        edge_table.add_row("Evidence Count", str(len(evidence)))

        if evidence:
            evidence_text = "\n".join([f"• {e[:80]}" for e in evidence[:5]])
            if len(evidence) > 5:
                evidence_text += f"\n... and {len(evidence) - 5} more"
            edge_table.add_row("Evidence", evidence_text)

        if metadata:
            for key, value in list(metadata.items())[:5]:
                edge_table.add_row(key.replace("_", " ").title(), str(value)[:80])

        self.console.print(
            self.formatter.create_panel(edge_table, f"🔍 Edge Detection: {edge_type}", LogLevel.SUCCESS.value)
        )

    def log_bayesian_calculation(
        self,
        prior: float,
        posterior: float,
        evidence_count: int,
        log_odds_prior: Optional[float] = None,
        log_odds_posterior: Optional[float] = None,
    ):
        """
        Log Bayesian calculation steps.

        Args:
            prior: Prior probability
            posterior: Posterior probability
            evidence_count: Number of evidence items
            log_odds_prior: Optional prior log-odds
            log_odds_posterior: Optional posterior log-odds
        """
        bayes_table = self.formatter.create_styled_table(["Metric", "Value"], "bold blue")
        bayes_table.add_row("Prior (p0)", f"[cyan]{prior:.2%}[/cyan]")
        bayes_table.add_row("Posterior (p_bayesian)", f"[green]{posterior:.2%}[/green]")
        bayes_table.add_row("Change", f"[{'green' if posterior > prior else 'red'}]{posterior - prior:+.2%}[/{'green' if posterior > prior else 'red'}]")
        bayes_table.add_row("Evidence Items", str(evidence_count))

        if log_odds_prior is not None:
            bayes_table.add_row("Log-Odds Prior", f"{log_odds_prior:.3f}")
        if log_odds_posterior is not None:
            bayes_table.add_row("Log-Odds Posterior", f"{log_odds_posterior:.3f}")

        self.console.print(
            self.formatter.create_panel(bayes_table, "🧮 Bayesian Calculation", LogLevel.INFO.value)
        )

    def log_evidence_gathering(
        self,
        direction: str,
        count: int,
        total_llr: float,
        sample_evidence: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Log evidence gathering progress.

        Args:
            direction: Direction of evidence (pro/con/general)
            count: Number of evidence items found
            total_llr: Total LLR contribution
            sample_evidence: Optional sample evidence items
        """
        evidence_table = self.formatter.create_styled_table(["Property", "Value"], "bold yellow")
        evidence_table.add_row("Direction", f"[bold]{direction.upper()}[/bold]")
        evidence_table.add_row("Items Found", str(count))
        evidence_table.add_row("Total LLR", f"{total_llr:+.2f}")

        if sample_evidence:
            evidence_table.add_row("Sample", "")
            for i, ev in enumerate(sample_evidence[:3], 1):
                title = ev.get("title", "Unknown")[:60]
                llr = ev.get("estimated_LLR", 0.0)
                evidence_table.add_row(f"  {i}", f"{title}... (LLR: {llr:+.2f})")

        self.console.print(
            self.formatter.create_panel(evidence_table, f"🔍 Evidence Gathering: {direction}", LogLevel.INFO.value)
        )

    def log_completion_status(self, complete: bool, reason: str = ""):
        """
        Log task completion status with optional reason.

        Args:
            complete: Whether the task completed successfully
            reason: Optional explanation for the status
        """
        assert isinstance(complete, bool), "Complete must be boolean"

        status_text = self.builder.build_completion_status(complete, reason)
        self.console.print(
            self.formatter.create_panel(
                status_text, "Status", LogLevel.SUCCESS.value if complete else LogLevel.WARNING.value
            )
        )

    def log_iteration_summary(self, iteration: int, state: Dict[str, Any]):
        """
        Log summary statistics for an iteration.

        Args:
            iteration: Iteration number
            state: Dictionary containing iteration state information
        """
        assert isinstance(iteration, int) and iteration > 0, "Iteration must be positive integer"
        assert isinstance(state, dict), "State must be dictionary"

        summary_table = self.builder.build_iteration_summary(iteration, state)
        self.console.print(
            self.formatter.create_panel(summary_table, f"Iteration {iteration} Summary", LogLevel.INFO.value)
        )

    def log_final_output(self, output: Dict[str, Any], execution_stats: Optional[Dict[str, Any]] = None):
        """
        Log comprehensive final agent output with execution statistics and summary.

        Args:
            output: Final output dictionary from agent execution
            execution_stats: Optional execution performance statistics
        """
        assert isinstance(output, dict), "Output must be dictionary"
        assert execution_stats is None or isinstance(execution_stats, dict), "Execution stats must be dict or None"

        if execution_stats:
            self._execution_stats.update(execution_stats)

        output_tree = Tree("[bold green]✅ Final Output[/bold green]")
        self._build_output_tree(output_tree, output)

        # Add execution statistics
        if self._execution_stats:
            stats_branch = output_tree.add("[bold blue]📈 Execution Stats[/bold blue]")
            for key, value in self._execution_stats.items():
                stats_branch.add(f"[blue]{key}:[/blue] {value}")

        # Summary metrics
        summary_metrics = self._generate_summary_metrics(output)
        if summary_metrics:
            output_tree.add(f"[bold]📊 Summary:[/bold] {' | '.join(summary_metrics)}")

        self.console.print(
            self.formatter.create_panel(output_tree, "Final Results", LogLevel.SUCCESS.value)
        )

    def _build_output_tree(self, parent: Tree, data: Any, depth: int = 0, max_depth: int = 3, parent_key: str = None):
        """Recursively build tree structure with smart data handling."""
        if depth >= max_depth:
            parent.add("[dim]... (truncated)[/dim]")
            return

        if isinstance(data, dict):
            items_to_show = list(data.items())[:20]
            for key, value in items_to_show:
                if isinstance(value, (dict, list)):
                    branch = parent.add(f"[cyan]{key}[/cyan] [dim]({type(value).__name__})[/dim]")
                    self._build_output_tree(branch, value, depth + 1, max_depth, key)
                else:
                    parent.add(f"[cyan]{key}[/cyan]: {self.formatter.format_text_preview(str(value), 100)}")

            if len(data) > 20:
                parent.add(f"[dim]... {len(data) - 20} more items[/dim]")

        elif isinstance(data, list):
            items_to_show = data[:15]
            for i, item in enumerate(items_to_show):
                item_desc = self.formatter.format_text_preview(str(item), 80)
                parent.add(f"[yellow]#{i+1}:[/yellow] {item_desc}")
            if len(data) > 15:
                parent.add(f"[dim]... {len(data) - 15} more items[/dim]")
        else:
            parent.add(self.formatter.format_text_preview(str(data), 120))

    def _generate_summary_metrics(self, output: Dict[str, Any]) -> List[str]:
        """Generate comprehensive summary metrics."""
        metrics = []

        # Market analysis metrics
        if "p_bayesian" in output:
            metrics.append(f"p_bayesian: {output['p_bayesian']:.2%}")
        if "evidence_items" in output:
            count = len(output["evidence_items"]) if isinstance(output["evidence_items"], list) else 0
            metrics.append(f"Evidence: {count} items")
        if "edge_signals" in output:
            count = len(output["edge_signals"]) if isinstance(output["edge_signals"], list) else 0
            metrics.append(f"Edge Signals: {count}")
        if "opportunities" in output:
            count = len(output["opportunities"]) if isinstance(output["opportunities"], list) else 0
            metrics.append(f"Opportunities: {count}")

        return metrics

    def log_error(self, error: str, details: str = "", exception: Exception = None):
        """
        Log error conditions with enhanced detail and context.

        Args:
            error: Primary error message
            details: Optional additional error context
            exception: Optional exception object for stack trace
        """
        assert error and isinstance(error, str), "Error must be non-empty string"
        assert isinstance(details, str), "Details must be string"
        assert exception is None or isinstance(exception, Exception), "Exception must be Exception or None"

        error_text = Text()
        error_text.append(f"❌ {error}", style="bold red")

        if details:
            error_text.append(f"\n{details}", style="dim red")

        if exception:
            error_text.append(f"\nException: {type(exception).__name__}: {str(exception)}", style="red")
            if hasattr(exception, "__traceback__"):
                import traceback

                tb_str = "".join(
                    traceback.format_exception(type(exception), exception, exception.__traceback__)
                )
                error_text.append(f"\n[dim red]Stack trace:[/dim red]\n{tb_str}", style="dim red")

        self.console.print(self.formatter.create_panel(error_text, "Error", LogLevel.ERROR.value))
        # Also log to standard logger
        self.std_logger.error(f"{error}: {details}", exc_info=exception)

    def add_execution_stat(self, key: str, value: Any):
        """Add execution statistic for final reporting."""
        assert key and isinstance(key, str), "Key must be non-empty string"
        self._execution_stats[key] = value


def setup_rich_logging(agent_name: str) -> RichAgentLogger:
    """
    Create and configure rich logging instance for agent workflows.

    Args:
        agent_name: Name identifier for the logging agent

    Returns:
        Configured RichAgentLogger instance
    """
    return RichAgentLogger(agent_name)


# TOOL LOGGING FUNCTIONS

def log_tool_start(tool_name: str, params: Dict[str, Any]) -> None:
    """
    Log tool invocation start with rich formatting.
    
    Args:
        tool_name: Name of the tool being invoked
        params: Dictionary of input parameters
    """
    tool_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    tool_table.add_column("Parameter", style="cyan")
    tool_table.add_column("Value", style="yellow")
    
    tool_table.add_row("Tool Name", f"[bold]{tool_name}[/bold]")
    
    # Add key parameters (limit to most important)
    key_params = {
        "query": params.get("query", ""),
        "market_question": params.get("market_question", ""),
        "market_slug": params.get("market_slug", ""),
        "max_results": params.get("max_results", ""),
        "p_bayesian": params.get("p_bayesian", ""),
        "prior_p": params.get("prior_p", ""),
    }
    
    for key, value in key_params.items():
        if value:
            display_value = str(value)
            if len(display_value) > 80:
                display_value = display_value[:77] + "..."
            tool_table.add_row(key.replace("_", " ").title(), display_value)
    
    # Add other params if not already shown
    for key, value in list(params.items())[:5]:
        if key not in key_params and value is not None:
            display_value = str(value)
            if len(display_value) > 80:
                display_value = display_value[:77] + "..."
            tool_table.add_row(key.replace("_", " ").title(), display_value)
    
    console.print(Panel(tool_table, title=f"🔧 Tool Start: {tool_name}", border_style="cyan"))
    std_logger.info(f"Tool {tool_name} started with params: {list(params.keys())}")


def log_tool_success(tool_name: str, result_summary: Dict[str, Any]) -> None:
    """
    Log successful tool completion with rich formatting.
    
    Args:
        tool_name: Name of the tool
        result_summary: Summary of results (e.g., {"results_count": 5, "execution_time": 1.2})
    """
    result_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")
    
    result_table.add_row("Status", "[bold green]✅ Success[/bold green]")
    
    # Format common result fields
    if "results_count" in result_summary:
        result_table.add_row("Results Found", str(result_summary["results_count"]))
    if "execution_time" in result_summary:
        result_table.add_row("Execution Time", f"{result_summary['execution_time']:.2f}s")
    if "edge_strength" in result_summary:
        result_table.add_row("Edge Strength", f"{result_summary['edge_strength']:.2f}")
    if "confidence" in result_summary:
        result_table.add_row("Confidence", f"{result_summary['confidence']:.2f}")
    if "p_bayesian" in result_summary:
        result_table.add_row("p_bayesian", f"{result_summary['p_bayesian']:.2%}")
    
    # Add other summary fields
    for key, value in result_summary.items():
        if key not in ["results_count", "execution_time", "edge_strength", "confidence", "p_bayesian"]:
            display_value = str(value)
            if len(display_value) > 60:
                display_value = display_value[:57] + "..."
            result_table.add_row(key.replace("_", " ").title(), display_value)
    
    console.print(Panel(result_table, title=f"✅ Tool Success: {tool_name}", border_style="green"))
    std_logger.info(f"Tool {tool_name} completed successfully")


def log_tool_error(tool_name: str, error: Exception, details: Optional[str] = None) -> None:
    """
    Log tool error with rich formatting.
    
    Args:
        tool_name: Name of the tool
        error: Exception that occurred
        details: Optional additional error details
    """
    error_text = Text()
    error_text.append(f"❌ Tool Error: {tool_name}\n", style="bold red")
    error_text.append(f"Error Type: {type(error).__name__}\n", style="red")
    error_text.append(f"Message: {str(error)}", style="dim red")
    
    if details:
        error_text.append(f"\n\nDetails: {details}", style="dim red")
    
    console.print(Panel(error_text, title="❌ Tool Error", border_style="red"))
    std_logger.error(f"Tool {tool_name} failed: {error}", exc_info=error)


def log_tool_progress(tool_name: str, message: str, progress_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Log intermediate tool progress with rich formatting.
    
    Args:
        tool_name: Name of the tool
        message: Progress message
        progress_data: Optional dictionary with progress metrics
    """
    progress_text = Text()
    progress_text.append(f"⏳ {tool_name}\n", style="bold yellow")
    progress_text.append(message, style="yellow")
    
    if progress_data:
        progress_text.append("\n\nProgress:", style="bold")
        for key, value in list(progress_data.items())[:5]:
            progress_text.append(f"\n  {key.replace('_', ' ').title()}: {value}", style="dim")
    
    console.print(Panel(progress_text, title="⏳ Tool Progress", border_style="yellow"))
    std_logger.info(f"Tool {tool_name} progress: {message}")


def log_search_results(tool_name: str, query: str, results: List[Dict[str, Any]], max_preview: int = 3) -> None:
    """
    Log search tool results with rich formatting.
    
    Args:
        tool_name: Name of the search tool
        query: Search query used
        results: List of search result dictionaries
        max_preview: Maximum number of results to preview
    """
    result_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    result_table.add_column("Property", style="cyan")
    result_table.add_column("Value", style="yellow")
    
    result_table.add_row("Tool", tool_name)
    result_table.add_row("Query", query[:100] + "..." if len(query) > 100 else query)
    result_table.add_row("Results Found", f"[green]{len(results)}[/green]")
    
    if results:
        result_table.add_row("", "")
        result_table.add_row("[bold]Top Results:[/bold]", "")
        for i, result in enumerate(results[:max_preview], 1):
            title = result.get("title", "No title")[:70]
            url = result.get("url", "No URL")[:50]
            result_table.add_row(f"  {i}.", f"{title}...")
            result_table.add_row("", f"    {url}")
    
    console.print(Panel(result_table, title=f"🔍 Search Results: {tool_name}", border_style="cyan"))
    std_logger.info(f"Tool {tool_name} found {len(results)} results for query: {query[:50]}")


def log_edge_detection_result(tool_name: str, edge_type: str, strength: float, confidence: float, evidence: List[str]) -> None:
    """
    Log edge detection tool result with rich formatting.
    
    Args:
        tool_name: Name of the edge detection tool
        edge_type: Type of edge detected
        strength: Edge strength (0-1)
        confidence: Confidence level (0-1)
        evidence: List of evidence strings
    """
    edge_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
    edge_table.add_column("Property", style="cyan")
    edge_table.add_column("Value", style="green")
    
    edge_table.add_row("Tool", tool_name)
    edge_table.add_row("Edge Type", f"[bold]{edge_type}[/bold]")
    edge_table.add_row("Strength", f"[green]{strength:.2f}[/green]")
    edge_table.add_row("Confidence", f"[cyan]{confidence:.2f}[/cyan]")
    edge_table.add_row("Evidence Count", str(len(evidence)))
    
    if evidence:
        evidence_text = "\n".join([f"• {e[:70]}" for e in evidence[:3]])
        if len(evidence) > 3:
            evidence_text += f"\n... and {len(evidence) - 3} more"
        edge_table.add_row("Evidence", evidence_text)
    
    console.print(Panel(edge_table, title=f"🔍 Edge Detection: {edge_type}", border_style="green"))
    std_logger.info(f"Tool {tool_name} detected {edge_type} edge with strength {strength:.2f}")


def log_bayesian_result(tool_name: str, prior: float, posterior: float, evidence_count: int) -> None:
    """
    Log Bayesian calculation result with rich formatting.
    
    Args:
        tool_name: Name of the Bayesian tool
        prior: Prior probability
        posterior: Posterior probability
        evidence_count: Number of evidence items used
    """
    bayes_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
    bayes_table.add_column("Metric", style="cyan")
    bayes_table.add_column("Value", style="blue")
    
    bayes_table.add_row("Tool", tool_name)
    bayes_table.add_row("Prior (p0)", f"[cyan]{prior:.2%}[/cyan]")
    bayes_table.add_row("Posterior (p_bayesian)", f"[green]{posterior:.2%}[/green]")
    change = posterior - prior
    bayes_table.add_row("Change", f"[{'green' if change > 0 else 'red'}]{change:+.2%}[/{'green' if change > 0 else 'red'}]")
    bayes_table.add_row("Evidence Items", str(evidence_count))
    
    console.print(Panel(bayes_table, title=f"🧮 Bayesian Calculation: {tool_name}", border_style="blue"))
    std_logger.info(f"Tool {tool_name} calculated posterior {posterior:.2%} from prior {prior:.2%}")


# ============================================================================
# WORKFLOW VISUALIZATION FUNCTIONS
# ============================================================================

# Workflow agent order
AGENT_ORDER = [
    "START",
    "Planner",
    "Researcher",
    "Critic",
    "Analyst",
    "Arbitrage",
    "Reporter",
    "END",
]


def render_workflow_graph() -> str:
    """
    Render ASCII diagram of agent workflow flow.
    
    Returns:
        ASCII string showing workflow graph
    """
    graph = """
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌──────────┐
    │ Planner  │
    └────┬─────┘
         │
         ▼
    ┌─────────────┐
    │ Researcher  │ (PRO/CON/GENERAL parallel)
    └────┬────────┘
         │
         ▼
    ┌─────────┐
    │ Critic  │
    └────┬────┘
         │
         ▼
    ┌──────────┐
    │ Analyst  │
    └────┬─────┘
         │
         ▼
    ┌─────────────┐
    │ Arbitrage   │
    └────┬────────┘
         │
         ▼
    ┌──────────┐
    │ Reporter │
    └────┬─────┘
         │
         ▼
    ┌───────┐
    │  END  │
    └───────┘
    """
    return graph


def log_workflow_transition(
    from_agent: str,
    to_agent: str,
    step: str,
    state_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log workflow transition with rich formatting.
    
    Args:
        from_agent: Name of agent transitioning from
        to_agent: Name of agent transitioning to
        step: Step indicator (e.g., "1/6")
        state_summary: Optional summary of workflow state
    """
    transition_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    transition_table.add_column("Property", style="cyan")
    transition_table.add_column("Value", style="yellow")
    
    transition_table.add_row("Step", f"[bold]{step}[/bold]")
    transition_table.add_row("From", f"[dim]{from_agent}[/dim]")
    transition_table.add_row("To", f"[bold green]{to_agent}[/bold green]")
    
    if state_summary:
        for key, value in list(state_summary.items())[:5]:
            display_value = str(value)
            if len(display_value) > 60:
                display_value = display_value[:57] + "..."
            transition_table.add_row(key.replace("_", " ").title(), display_value)
    
    console.print(Panel(transition_table, title=f"🔄 Workflow Transition: {from_agent} → {to_agent}", border_style="cyan"))
    std_logger.info(f"Workflow transition: {from_agent} → {to_agent} ({step})")


def log_workflow_progress(
    current_step: int,
    total_steps: int,
    agent_name: str,
    status: str = "running",
) -> None:
    """
    Log workflow progress with progress bar.
    
    Args:
        current_step: Current step number (1-based)
        total_steps: Total number of steps
        agent_name: Name of current agent
        status: Status string (running, completed, error)
    """
    progress_text = Text()
    progress_text.append(f"Step {current_step}/{total_steps}: ", style="bold")
    progress_text.append(f"{agent_name}", style="bold cyan")
    
    if status == "completed":
        progress_text.append(" ✅", style="green")
    elif status == "error":
        progress_text.append(" ❌", style="red")
    else:
        progress_text.append(" ⏳", style="yellow")
    
    # Calculate percentage
    percentage = (current_step / total_steps) * 100
    
    progress_table = Table(show_header=False, box=box.ROUNDED)
    progress_table.add_column(style="cyan")
    progress_table.add_column(style="yellow")
    
    progress_table.add_row("Progress", f"{percentage:.0f}%")
    progress_table.add_row("Current", agent_name)
    progress_table.add_row("Status", status.title())
    
    console.print(Panel(progress_table, title=f"📊 Workflow Progress: {current_step}/{total_steps}", border_style="blue"))
    std_logger.info(f"Workflow progress: {current_step}/{total_steps} - {agent_name} ({status})")


def log_workflow_summary(
    execution_stats: Dict[str, Any],
    agent_times: Optional[Dict[str, float]] = None,
    tool_call_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Log final workflow summary with execution metrics.
    
    Args:
        execution_stats: Dictionary with overall execution statistics
        agent_times: Optional dict mapping agent names to execution times
        tool_call_counts: Optional dict mapping agent names to tool call counts
    """
    summary_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    # Overall stats
    if "total_time" in execution_stats:
        summary_table.add_row("Total Execution Time", f"{execution_stats['total_time']:.2f}s")
    if "total_iterations" in execution_stats:
        summary_table.add_row("Total Iterations", str(execution_stats['total_iterations']))
    if "total_tool_calls" in execution_stats:
        summary_table.add_row("Total Tool Calls", str(execution_stats['total_tool_calls']))
    
    # Agent times
    if agent_times:
        summary_table.add_row("", "")
        summary_table.add_row("[bold]Agent Execution Times:[/bold]", "")
        for agent, time_sec in sorted(agent_times.items(), key=lambda x: x[1], reverse=True)[:6]:
            summary_table.add_row(f"  {agent}", f"{time_sec:.2f}s")
    
    # Tool call counts
    if tool_call_counts:
        summary_table.add_row("", "")
        summary_table.add_row("[bold]Tool Calls per Agent:[/bold]", "")
        for agent, count in sorted(tool_call_counts.items(), key=lambda x: x[1], reverse=True)[:6]:
            summary_table.add_row(f"  {agent}", str(count))
    
    # Results summary
    if "evidence_items" in execution_stats:
        summary_table.add_row("", "")
        summary_table.add_row("Evidence Items Found", str(execution_stats['evidence_items']))
    if "opportunities" in execution_stats:
        summary_table.add_row("Arbitrage Opportunities", str(execution_stats['opportunities']))
    if "p_bayesian" in execution_stats:
        summary_table.add_row("Bayesian Probability", f"{execution_stats['p_bayesian']:.2%}")
    
    console.print(Panel(summary_table, title="📈 Workflow Summary", border_style="green"))
    std_logger.info("Workflow summary displayed")


def log_agent_separator(agent_name: str) -> None:
    """
    Log a visual separator for agent execution boundaries.
    
    Creates a prominent visual separator to distinguish between different
    agent executions in the workflow.
    
    Args:
        agent_name: Name of the agent being separated
    """
    assert agent_name and isinstance(agent_name, str), "Agent name must be non-empty string"
    
    # Create a visually distinct separator
    separator_text = Text()
    separator_text.append("═" * 80, style="bold cyan")
    separator_text.append(f"\n🤖 {agent_name}", style="bold cyan")
    separator_text.append("\n" + "═" * 80, style="bold cyan")
    
    console.print()
    console.print(Panel(separator_text, border_style="cyan", box=box.DOUBLE))
    console.print()
    std_logger.info(f"Agent separator: {agent_name}")


def log_agent_output_full(agent_name: str, output: Any, max_depth: int = 4) -> None:
    """
    Log complete agent output with full details (no truncation for key fields).
    
    Args:
        agent_name: Name of the agent
        output: Agent output (can be Pydantic model, dict, or other)
        max_depth: Maximum depth for nested structures
    """
    output_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    output_table.add_column("Field", style="cyan", width=30)
    output_table.add_column("Value", style="yellow")
    
    output_table.add_row("[bold]Agent[/bold]", f"[bold]{agent_name}[/bold]")
    
    # Convert Pydantic model to dict if needed
    if hasattr(output, 'model_dump'):
        output_dict = output.model_dump()
    elif hasattr(output, 'dict'):
        output_dict = output.dict()
    elif isinstance(output, dict):
        output_dict = output
    else:
        output_dict = {"raw": str(output)}
    
    # Display all fields without truncation for key metrics
    for key, value in output_dict.items():
        display_value = _format_output_value(value, max_depth=max_depth, key=key)
        output_table.add_row(key.replace("_", " ").title(), display_value)
    
    console.print(Panel(output_table, title=f"📊 Complete {agent_name} Output", border_style="green"))
    std_logger.info(f"Complete {agent_name} output logged")


def _format_output_value(value: Any, max_depth: int = 4, depth: int = 0, key: str = "") -> str:
    """Format a value for display, with special handling for key metrics"""
    if depth >= max_depth:
        return "[dim]... (truncated)[/dim]"
    
    # Special handling for key metrics - show full values
    key_metrics = ['p_bayesian', 'p0', 'p0_prior', 'confidence_interval', 'p_bayesian_low', 'p_bayesian_high',
                   'edge', 'expected_value_per_dollar', 'kelly_fraction', 'suggested_stake',
                   'executive_summary', 'tldr', 'arbitrage_summary']
    
    if isinstance(value, float) and key in key_metrics:
        if 0 <= value <= 1:
            return f"[green]{value:.4f}[/green] ({value:.2%})"
        else:
            return f"[green]{value:.4f}[/green]"
    
    if isinstance(value, (int, float)):
        return str(value)
    
    if isinstance(value, str):
        # For key text fields, show more content
        if key in ['executive_summary', 'tldr', 'arbitrage_summary', 'prior_justification', 'reasoning_trace']:
            if len(value) > 500:
                return value[:497] + "..."
            return value
        # For other strings, truncate moderately
        if len(value) > 200:
            return value[:197] + "..."
        return value
    
    if isinstance(value, list):
        if len(value) == 0:
            return "[dim]Empty list[/dim]"
        if len(value) <= 5:
            items = [_format_output_value(item, max_depth, depth + 1) for item in value]
            return "\n".join([f"  • {item}" for item in items])
        else:
            items = [_format_output_value(item, max_depth, depth + 1) for item in value[:3]]
            return "\n".join([f"  • {item}" for item in items]) + f"\n  ... and {len(value) - 3} more"
    
    if isinstance(value, dict):
        if len(value) == 0:
            return "[dim]Empty dict[/dim]"
        items = []
        for k, v in list(value.items())[:5]:
            formatted_v = _format_output_value(v, max_depth, depth + 1, k)
            items.append(f"  {k}: {formatted_v}")
        if len(value) > 5:
            items.append(f"  ... and {len(value) - 5} more fields")
        return "\n".join(items)
    
    return str(value)[:200]

