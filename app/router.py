"""Heuristic routing for query classification into retrieval strategies.

Defines:
- Route: Literal type alias of allowed routes.
- RouteDecision: Dataclass carrying chosen route, rationale, and alpha weight.
- classify_query: Heuristic classifier producing a RouteDecision.

Alpha is the vector weight used in a hybrid keyword+vector retriever.
"""
import re
from dataclasses import dataclass
from typing import Literal


Route = Literal["factual", "keyword", "navigational"]


@dataclass
class RouteDecision:
    """Routing decision and blend weight for retrieval strategy.

    Attributes:
        route: One of 'factual', 'keyword', or 'navigational'.
        reason: Short human-readable rationale for the chosen route.
        alpha: Weight for vector score in hybrid blend (0..1); higher favors vector.
    """
    route: Route
    reason: str
    alpha: float  # weight for vector score in hybrid blend (0..1)


QUESTION_WORDS = {"what", "how", "why", "when", "where", "which", "who"}
NAV_TERMS = {
    "overview",
    "introduction",
    "guide",
    "tutorial",
    "quickstart",
    "getting started",
    "reference",
    "api reference",
    "docs",
}
CODEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\(|`[^`]+`|[_A-Z]{2,}[A-Z0-9_]*")
NUMERIC_HEAVY = re.compile(r"\b\d{2,}\b")
SEPARATOR_CHARS = re.compile(r"[:/#._-]")


def classify_query(q: str) -> RouteDecision:
    """Classify a user query into a retrieval route using heuristics.

    Args:
        q: The raw user query string.

    Returns:
        RouteDecision: Selected route, rationale, and alpha (vector weight in hybrid blend).

    Heuristics:
        - 'keyword' if many code-like identifiers, numeric tokens, or separators are present.
        - 'navigational' for guide/overview/doc navigation intents or "where ..." patterns.
        - 'factual' for question-form queries (what/how/why/etc).
        - Short terse queries may bias toward 'keyword'.
    """
    ql = q.strip().lower()

    # Signals
    has_qword = any(ql.startswith(w + " ") or f" {w} " in ql for w in QUESTION_WORDS)
    has_nav = any(t in ql for t in NAV_TERMS)
    code_hits = len(CODEY.findall(q))
    numeric_hits = len(NUMERIC_HEAVY.findall(q))
    sep_hits = len(SEPARATOR_CHARS.findall(q))

    # Keyword-heavy if many code/numeric/separators or shouting IDs
    if code_hits + numeric_hits + sep_hits >= 3:
        return RouteDecision(route="keyword", reason="code/numeric/separators heavy", alpha=0.4)

    # Navigational if mentions guides/overview or "where is/are ...", "docs for ..."
    if has_nav or ql.startswith("where ") or " where " in ql or " docs" in ql or "documentation" in ql:
        return RouteDecision(route="navigational", reason="nav terms present", alpha=0.5)

    # Factual by default (how/what questions)
    if has_qword:
        return RouteDecision(route="factual", reason="question-form query", alpha=0.65)

    # Fallback: choose factual but slightly balance toward keyword if short and terse
    if len(q.split()) <= 5:
        return RouteDecision(route="keyword", reason="short terse query", alpha=0.45)

    return RouteDecision(route="factual", reason="default", alpha=0.6)
