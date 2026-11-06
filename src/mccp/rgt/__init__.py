"""Rationale-Guided Training (RGT) data construction.

RGT (O'Neill, Partridge, Jayasekara, Baseten Research, Oct 2025) changes what
is supervised, not how: for each training example it keeps the standard pair

    A: prompt -> <empty reasoning block> answer

and adds a strategy-explicit pair

    B: prompt -> <reasoning block> strategy </reasoning block> answer

where the strategy is a compact rule distilled from a grader's rationale for
why the answer is correct (and, when the current model got it wrong, what the
mistake was). At inference the model is served with an empty reasoning block,
so latency is unchanged.

This package holds the task-independent contracts (`Feedback`, `Strategy`,
`Grader`, `Distiller`), the pair builder, and provenance for the audit trail.
Task-specific graders/distillers live in submodules (`invoices`).
"""

from .contracts import Attempt, Distiller, Feedback, Grader, Issue, Strategy
from .pairs import RGTBuild, build_rgt_rows, final_turn_prompt, hinge_rows, render_reasoning_block

__all__ = [
    "Attempt",
    "Distiller",
    "Feedback",
    "Grader",
    "Issue",
    "RGTBuild",
    "Strategy",
    "build_rgt_rows",
    "final_turn_prompt",
    "hinge_rows",
    "render_reasoning_block",
]
