"""POLYSEER Workflow Module"""
from .workflow.autonomous_graph import (
    create_autonomous_polyseer_workflow,
    run_autonomous_workflow,
)

# Backwards-compatible aliases
create_polyseer_workflow = create_autonomous_polyseer_workflow
run_workflow = run_autonomous_workflow

__all__ = [
    'create_autonomous_polyseer_workflow',
    'run_autonomous_workflow',
    'create_polyseer_workflow',
    'run_workflow',
]
