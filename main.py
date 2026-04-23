"""
Company Search Query Agent 
Flow: parse → validate → END (valid) (or) clarify → parse
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

NOTE : keep reasoning under 50-80 words, not more than that!!!

Rules:
- Use null for ANY field that is missing OR too vague to act on.
- location must be a specific, actionable geography (city, state, country, metro area).
  Vague phrases with no clear clarity like "near the border", "near a big city" → null.
- industry must be a recognizable business sector. Vague phrases like "some businesses" → null.
- required_attributes must only contain traits explicitly stated by the user — never infer or add domain suggestions.
- If the query has contradictory size signals, add "CONFLICT: <description>" to optional_attributes, set size_range null.
- Never put conflicting size signals in required_attributes.
- If a trait is part of a CONFLICT, do NOT also add it to required_attributes.
"""
 
CLARIFY_SYSTEM = """You assist a business directory search. The user's query was incomplete or conflicting.
Ask ONE short follow-up question targeting the key feilds... 1-3 max. Return only the question."""

class SizeRange(BaseModel):
    min: Optional[int] = None   # min emp 
    max: Optional[int] = None   # max emp

class ParsedQuery(BaseModel):
    industry:             Optional[str]       = None   # business type, e.g. "plumbing"
    location:             Optional[str]       = None   # geography
    size_range:           Optional[SizeRange] = None   # emp count
    required_attributes:  list[str]           = Field(default_factory=list)  # filters
    optional_attributes:  list[str]           = Field(default_factory=list)  # conflict_flag
    reasoning:            Optional[str]       = None   #reasoning

    @field_validator("required_attributes", "optional_attributes", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v):
        return v if v is not None else []

class AgentState(BaseModel):
    raw_input:              str                    # source query
    parsed_query:           ParsedQuery            = Field(default_factory=ParsedQuery)
    is_valid:               bool                   = False
    clarification_question: Optional[str]          = None
    clarification_answer:   Optional[str]          = None
    attempts:               int                    = 0


llm        = ChatOpenAI(model=MODEL, temperature=0)
parse_llm  = llm.with_structured_output(ParsedQuery)   # returns ParsedQuery directly


def parse_node(state: AgentState) -> AgentState:
    user_msg = (
        f"Original query: {state.raw_input}\n"
        f"Clarification — question: '{state.clarification_question}'\n"
        f"Clarification — answer: '{state.clarification_answer}'"
        if state.clarification_question else state.raw_input
    )

    parsed = parse_llm.invoke([{"role": "system", "content": PARSE_SYSTEM},
                                {"role": "user",   "content": user_msg}])

    return state.model_copy(update={
        "parsed_query":           parsed,
        "clarification_question": None,
        "clarification_answer":   None,
        "attempts":               state.attempts + 1,
    })


def validate_node(state: AgentState) -> AgentState:
    if state.attempts >= 2:
        return state.model_copy(update={"is_valid": True})

    pq          = state.parsed_query
    has_conflict = any("CONFLICT" in a for a in pq.optional_attributes)
    is_valid     = bool(pq.industry and pq.location and not has_conflict)

    return state.model_copy(update={"is_valid": is_valid})




def clarify_node(state: AgentState) -> AgentState:
    context  = f"Parsed so far: {state.parsed_query.model_dump_json(indent=2)}"
    resp     = llm.invoke([{"role": "system", "content": CLARIFY_SYSTEM},
                            {"role": "user",   "content": context}])
    question = resp.content.strip()

    print(f"\n  Clarification: {question}")
    answer = input("  Your answer: ").strip()
    return state.model_copy(update={"clarification_question": question, "clarification_answer": answer})



def route(state: AgentState) -> str:
    return END if state.is_valid else "clarify"

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("parse",    parse_node)
    g.add_node("validate", validate_node)
    g.add_node("clarify",  clarify_node)

    g.set_entry_point("parse")
    g.add_edge("parse",   "validate")
    g.add_edge("clarify", "parse")
    g.add_conditional_edges("validate", route, {END: END, "clarify": "clarify"})

    return g.compile()


RUNS_DIR = os.path.join(os.path.dirname(__file__), ".runs")

def run(query: str):
    raw = build_graph().invoke(AgentState(raw_input=query))
    result = AgentState(**raw) if isinstance(raw, dict) else raw  # LangGraph returns dict; re-wrap into Pydantic

    print(f"\n{'-'*60}")
    print(f"  INPUT   : {query}")
    print(f"  RESULT  : {result.parsed_query.model_dump_json(indent=4)}")
    print(f"  valid={result.is_valid}  attempts={result.attempts}")
    print(f"{'-'*60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_file  = os.path.join(RUNS_DIR, f"{timestamp}.json")
    with open(run_file, "w") as f:
        json.dump({
            "input":    query,
            "result":   json.loads(result.parsed_query.model_dump_json()),
            "valid":    result.is_valid,
            "attempts": result.attempts,
        }, f, indent=2)

    return result



if __name__ == "__main__":

    #fully specified
    run("B2B SaaS companies in the San Francisco Bay Area with between 10 and 50 employees, founded after 2015, that are actively hiring and have a Glassdoor rating above 4.0")

    #vague
    run("profitable businesses near the border that deal with imports")

    # conflicting size signals
    run("enterprise Fortune 500 companies with under 10 employees in rural Montana")