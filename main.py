"""
Company Search Query Agent
Flow: parse → validate → END (valid) | clarify → parse (capped at 2 attempts)
"""

import json, os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


MODEL = "gpt-4.1-mini"

PARSE_SYSTEM = """
You are a Query mapper. Extract a JSON search filter from the user's business directory query.
Return ONLY valid JSON with these exact keys:
  - industry: string or null
  - location: string or null
  - size_range: {min, max} or null
  - required_attributes: list of strings (must-have traits explicitly stated by the user)
  - optional_attributes: list of strings (nice-to-have traits OR conflict flags)
  - reasoning: string — brief explanation of every mapping decision: why each field was set,
    why anything was set to null, and why any CONFLICT was flagged.

NOTE :
- keep reasoning under 50-80 words, not more than that!!!
- Capture ALL identifiers from the query (company names, URLs, domains, person names, ticker symbols, etc.) as-is into required_attributes — do not drop them.

Rules:
- Use null for ANY field that is missing OR too vague to act on.
- location must be a specific, actionable geography (city, state, country, metro area).
  Vague phrases like "near the border", "near a big city" → null.
- industry must be a recognizable business sector. Vague phrases like "some businesses" → null.
- required_attributes must only contain traits explicitly stated by the user — never infer or add domain suggestions.
- If the query has contradictory size signals, add "CONFLICT: <description>" to optional_attributes, set size_range null.
- Never put conflicting size signals in required_attributes.
- If a trait is part of a CONFLICT, do NOT also add it to required_attributes.
"""

CLARIFY_SYSTEM = """You assist a business directory search. The user's query was incomplete or conflicting.
You will be given the parsed query so far and the specific validation issues found.
Ask ONE short follow-up question targeting the most critical issue. Return only the question."""


# ── Models ────────────────────────────────────────────────────────────────────

class SizeRange(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None

class ParsedQuery(BaseModel):
    industry:            Optional[str]       = None
    location:            Optional[str]       = None
    size_range:          Optional[SizeRange] = None
    required_attributes: list[str]           = Field(default_factory=list)
    optional_attributes: list[str]           = Field(default_factory=list)
    reasoning:           Optional[str]       = None

    @field_validator("required_attributes", "optional_attributes", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v):
        return v if v is not None else []

class AgentState(BaseModel):
    raw_input:              str
    parsed_query:           ParsedQuery  = Field(default_factory=ParsedQuery)
    is_valid:               bool         = False
    validation_issues:      list[str]    = Field(default_factory=list)  # reasons validation failed
    clarification_question: Optional[str] = None
    clarification_answer:   Optional[str] = None
    attempts:               int          = 0


# ── LLM setup ─────────────────────────────────────────────────────────────────

llm       = ChatOpenAI(model=MODEL, temperature=0)
parse_llm = llm.with_structured_output(ParsedQuery)  # schema-enforced, no json.loads needed


# ── Nodes ─────────────────────────────────────────────────────────────────────

def parse_node(state: AgentState) -> AgentState:
    user_msg = (
        f"Original query: {state.raw_input}\n"
        f"Clarification — question: '{state.clarification_question}'\n"
        f"Clarification — answer: '{state.clarification_answer}'"
        if state.clarification_question else state.raw_input
    )

    try:
        parsed = parse_llm.invoke([
            {"role": "system", "content": PARSE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ])
    except Exception as e:
        print(f"\n  [!] Parse failed (attempt {state.attempts + 1}): {e}")
        # keep previous parsed_query, just bump attempts so we don't loop forever
        return state.model_copy(update={"attempts": state.attempts + 1})

    return state.model_copy(update={
        "parsed_query":           parsed,
        "clarification_question": None,
        "clarification_answer":   None,
        "attempts":               state.attempts + 1,
    })


def validate_node(state: AgentState) -> AgentState:
    # force-pass after max attempts to avoid infinite loop
    if state.attempts >= 2:
        return state.model_copy(update={"is_valid": True, "validation_issues": []})

    pq     = state.parsed_query
    issues = []

    if any("CONFLICT" in a for a in pq.optional_attributes):
        issues.append("conflicting constraints detected — needs clarification")

    if not pq.industry:
        issues.append("industry is missing or too vague")

    if not pq.location:
        issues.append("location is missing or too vague")

    if pq.size_range:
        mn, mx = pq.size_range.min, pq.size_range.max
        if mn is not None and mx is not None and mn > mx:
            issues.append(f"size_range is invalid: min ({mn}) > max ({mx})")

    return state.model_copy(update={
        "is_valid":         len(issues) == 0,
        "validation_issues": issues,
    })


def clarify_node(state: AgentState) -> AgentState:
    context = (
        f"Parsed so far:\n{state.parsed_query.model_dump_json(indent=2)}\n\n"
        f"Validation issues:\n" + "\n".join(f"- {i}" for i in state.validation_issues)
    )

    try:
        resp     = llm.invoke([
            {"role": "system", "content": CLARIFY_SYSTEM},
            {"role": "user",   "content": context},
        ])
        question = resp.content.strip()
    except Exception as e:
        print(f"\n  [!] Clarify LLM failed: {e}")
        question = "Could you provide more details about what you're looking for?"

    print(f"\n  Clarification: {question}")
    answer = input("  Your answer: ").strip()

    return state.model_copy(update={
        "clarification_question": question,
        "clarification_answer":   answer,
    })


# ── Graph ─────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    return END if state.is_valid else "clarify"

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("parse",    parse_node)
    g.add_node("validate", validate_node)
    g.add_node("clarify",  clarify_node)
    g.set_entry_point("parse")
    g.add_edge("parse",   "validate")
    g.add_edge("clarify", "parse")
    g.add_conditional_edges("validate", route, {END: END, "clarify": "clarify"})
    return g.compile()

_graph = _build_graph()  # compile once, reuse across all run() calls


# ── Run ───────────────────────────────────────────────────────────────────────

RUNS_DIR = os.path.join(os.path.dirname(__file__), ".runs")
os.makedirs(RUNS_DIR, exist_ok=True)

def run(query: str) -> AgentState:
    raw    = _graph.invoke(AgentState(raw_input=query))
    result = AgentState(**raw) if isinstance(raw, dict) else raw

    print(f"\n{'-'*60}")
    print(f"  INPUT   : {query}")
    print(f"  RESULT  : {result.parsed_query.model_dump_json(indent=4)}")
    if result.validation_issues:
        print(f"  ISSUES  : {result.validation_issues}")
    print(f"  valid={result.is_valid}  attempts={result.attempts}")
    print(f"{'-'*60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with open(os.path.join(RUNS_DIR, f"{timestamp}.json"), "w") as f:
        json.dump({
            "input":             query,
            "result":            json.loads(result.parsed_query.model_dump_json()),
            "validation_issues": result.validation_issues,
            "valid":             result.is_valid,
            "attempts":          result.attempts,
        }, f, indent=2)

    return result
