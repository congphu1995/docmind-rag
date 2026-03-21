"""
Decomposes multi_hop questions into 2-3 simpler sub-questions.
Only invoked when query_type == 'multi_hop'.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.app.agent.llm import get_mini_model
from backend.app.agent.prompts import DECOMPOSE_PROMPT
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger


async def decomposer(state: RAGAgentState, config: RunnableConfig = None) -> dict:
    query = state["original_query"]
    log = logger.bind(node="decomposer")

    llm = get_mini_model().bind(temperature=0, max_tokens=200)
    result = await llm.ainvoke(
        [HumanMessage(content=DECOMPOSE_PROMPT.format(query=query))],
        config=config,
    )

    sub_questions = [
        line.strip()
        for line in result.content.strip().split("\n")
        if line.strip() and len(line.strip()) > 5
    ]

    if not sub_questions:
        sub_questions = [query]

    log.info("decomposed", sub_questions=len(sub_questions))

    return {
        "sub_questions": sub_questions,
        "agent_trace": [
            f"Decomposed into {len(sub_questions)} sub-questions: "
            + "; ".join(sub_questions)
        ],
    }
