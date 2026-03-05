"""
First node in the agent pipeline.
Classifies query type, detects language, extracts metadata filters.
"""
from backend.app.agent.prompts import QUERY_ANALYSIS_PROMPT, QUERY_ANALYSIS_SYSTEM
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory
from backend.app.schemas.chat import QueryAnalysis


async def query_analyzer(state: RAGAgentState) -> dict:
    query = state["original_query"]
    log = logger.bind(node="query_analyzer")

    try:
        llm = LLMFactory.create_mini()
        analysis = await llm.complete_structured(
            messages=[
                {
                    "role": "user",
                    "content": QUERY_ANALYSIS_PROMPT.format(query=query),
                }
            ],
            response_model=QueryAnalysis,
            system=QUERY_ANALYSIS_SYSTEM,
            max_tokens=200,
            temperature=0,
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
            "extracted_filters": analysis.filters,
            "agent_trace": [
                f"Query classified as: {analysis.query_type} "
                f"(lang={analysis.language})"
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
