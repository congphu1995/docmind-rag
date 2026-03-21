"""
LangGraph StateGraph — wires all agent nodes with conditional routing.

Flow:
  query_analyzer → router (conditional)
    ├── factual/analytical/tabular → query_rewriter → retriever → reranker → generator → END
    ├── multi_hop → decomposer → query_rewriter → retriever → reranker → generator → END
    ├── general → direct_llm → END
    └── greeting → direct_response → END
"""

from langgraph.graph import END, StateGraph

from backend.app.agent.nodes.decomposer import decomposer
from backend.app.agent.nodes.generator import (
    direct_llm,
    direct_response,
    generator_node,
)
from backend.app.agent.nodes.query_analyzer import query_analyzer
from backend.app.agent.nodes.query_rewriter import query_rewriter
from backend.app.agent.nodes.reranker import reranker_node
from backend.app.agent.nodes.retriever import retriever_node
from backend.app.agent.state import RAGAgentState


def _route_query(state: RAGAgentState) -> str:
    """Conditional routing after query_analyzer."""
    query_type = state.get("query_type", "factual")

    if query_type == "greeting":
        return "direct_response"
    if query_type == "general":
        return "direct_llm"
    if query_type == "multi_hop":
        return "decomposer"

    # factual, analytical, tabular → query_rewriter
    return "query_rewriter"


def build_graph():
    """Build and compile the RAG agent graph."""
    graph = StateGraph(RAGAgentState)

    # Add nodes
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("decomposer", decomposer)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("generator", generator_node)
    graph.add_node("direct_response", direct_response)
    graph.add_node("direct_llm", direct_llm)

    # Entry point
    graph.set_entry_point("query_analyzer")

    # Conditional routing after analysis
    graph.add_conditional_edges(
        "query_analyzer",
        _route_query,
        {
            "query_rewriter": "query_rewriter",
            "decomposer": "decomposer",
            "direct_response": "direct_response",
            "direct_llm": "direct_llm",
        },
    )

    # Decomposer → query_rewriter
    graph.add_edge("decomposer", "query_rewriter")

    # Main retrieval path
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")

    # Terminal edges
    graph.add_edge("generator", END)
    graph.add_edge("direct_response", END)
    graph.add_edge("direct_llm", END)

    return graph.compile()
