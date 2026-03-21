"""
First node in the agent pipeline.
Classifies query type, detects language, extracts metadata filters.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from backend.app.agent.llm import get_mini_model
from backend.app.agent.prompts import QUERY_ANALYSIS_PROMPT, QUERY_ANALYSIS_SYSTEM
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger
from backend.app.schemas.chat import QueryAnalysis


async def query_analyzer(state: RAGAgentState, config: RunnableConfig = None) -> dict:
    query = state["original_query"]
    log = logger.bind(node="query_analyzer")

    try:
        llm = get_mini_model()
        structured_llm = llm.with_structured_output(QueryAnalysis)

        analysis = await structured_llm.ainvoke(
            [
                SystemMessage(content=QUERY_ANALYSIS_SYSTEM),
                HumanMessage(content=QUERY_ANALYSIS_PROMPT.format(query=query)),
            ],
            config=config,
        )

        log.info(
            "query_analyzed",
            query_type=analysis.query_type,
            language=analysis.language,
        )

        return {
            "query_type": analysis.query_type,
            "detected_language": analysis.language,
            "sub_questions": analysis.sub_questions,
            "extracted_filters": analysis.filters.model_dump(exclude_defaults=True),
            "agent_trace": [
                f"Query classified as: {analysis.query_type} (lang={analysis.language})"
            ],
        }

    except Exception as e:
        log.warning("query_analysis_failed", error=str(e))
        return {
            "query_type": "factual",
            "detected_language": "en",
            "sub_questions": [],
            "extracted_filters": {},
            "agent_trace": [f"Query analysis failed, defaulting to factual: {e}"],
        }
