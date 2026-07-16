from __future__ import annotations

from typing import List, Literal, TypedDict

from .config import Settings
from .llm import LLMClient, build_answer_prompt, build_debug_prompt, build_issue_prompt
from .models import SearchHit
from .store import ChromaStore
from .tickets import build_basic_issue, classify_priority


Route = Literal["answer", "debug", "issue", "priority"]


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    contexts: List[SearchHit]
    answer: str


def simple_router(question: str) -> Route:
    q = question.lower()
    if any(word in q for word in ["priority", "severity", "critical", "classify"]):
        return "priority"
    if any(word in q for word in ["create issue", "bug report", "github issue", "draft issue"]):
        return "issue"
    if any(word in q for word in ["error", "bug", "crash", "fail", "exception", "debug", "traceback"]):
        return "debug"
    return "answer"


class RepoPilotAgent:
    def __init__(self, store: ChromaStore, settings: Settings):
        self.store = store
        self.llm = LLMClient(settings)
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        workflow = StateGraph(AgentState)

        def route_node(state: AgentState) -> AgentState:
            return {**state, "route": simple_router(state["question"])}

        def retrieve_node(state: AgentState) -> AgentState:
            hits = self.store.search(state["question"], k=6)
            return {**state, "contexts": hits}

        def answer_node(state: AgentState) -> AgentState:
            prompt = build_answer_prompt(state["question"], state.get("contexts", []))
            answer = self.llm.generate(prompt, state.get("contexts", []))
            return {**state, "answer": answer}

        def debug_node(state: AgentState) -> AgentState:
            prompt = build_debug_prompt(state["question"], state.get("contexts", []))
            answer = self.llm.generate(prompt, state.get("contexts", []))
            return {**state, "answer": answer}

        def issue_node(state: AgentState) -> AgentState:
            contexts = state.get("contexts", [])
            if contexts:
                prompt = build_issue_prompt(state["question"], contexts)
                answer = self.llm.generate(prompt, contexts)
            else:
                answer = build_basic_issue(state["question"])
            return {**state, "answer": answer}

        def priority_node(state: AgentState) -> AgentState:
            result = classify_priority(state["question"])
            answer = f"Priority: **{result.priority}**\n\nReason: {result.reason}\n\nRule score: {result.score}"
            return {**state, "answer": answer}

        workflow.add_node("route", route_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("answer", answer_node)
        workflow.add_node("debug", debug_node)
        workflow.add_node("issue", issue_node)
        workflow.add_node("priority", priority_node)

        workflow.set_entry_point("route")
        workflow.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {
                "answer": "retrieve",
                "debug": "retrieve",
                "issue": "retrieve",
                "priority": "priority",
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            lambda state: state["route"],
            {
                "answer": "answer",
                "debug": "debug",
                "issue": "issue",
                "priority": "priority",
            },
        )
        workflow.add_edge("answer", END)
        workflow.add_edge("debug", END)
        workflow.add_edge("issue", END)
        workflow.add_edge("priority", END)
        return workflow.compile()

    def run(self, question: str) -> AgentState:
        initial: AgentState = {"question": question}
        if self.graph is None:
            # Fallback path if LangGraph is not installed.
            route = simple_router(question)
            contexts: list[SearchHit] = [] if route == "priority" else self.store.search(question, k=6)
            if route == "debug":
                answer = self.llm.generate(build_debug_prompt(question, contexts), contexts)
            elif route == "issue":
                answer = self.llm.generate(build_issue_prompt(question, contexts), contexts) if contexts else build_basic_issue(question)
            elif route == "priority":
                result = classify_priority(question)
                answer = f"Priority: **{result.priority}**\n\nReason: {result.reason}\n\nRule score: {result.score}"
            else:
                answer = self.llm.generate(build_answer_prompt(question, contexts), contexts)
            return {"question": question, "route": route, "contexts": contexts, "answer": answer}
        return self.graph.invoke(initial)
