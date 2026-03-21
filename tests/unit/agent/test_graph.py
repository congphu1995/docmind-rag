
from backend.app.agent.graph import build_graph


def test_graph_compiles():
    """Graph should compile without errors."""
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    """Verify all expected nodes are in the graph."""
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "query_analyzer",
        "query_rewriter",
        "decomposer",
        "retriever",
        "reranker",
        "generator",
        "direct_response",
        "direct_llm",
    }
    assert expected.issubset(node_names)
