"""
Decomposes multi_hop questions into 2-3 simpler sub-questions.
Only invoked when query_type == 'multi_hop'.
"""
from backend.app.agent.prompts import DECOMPOSE_PROMPT
from backend.app.agent.state import RAGAgentState
from backend.app.core.logging import logger
from backend.app.pipeline.llm.factory import LLMFactory


async def decomposer(state: RAGAgentState) -> dict:
    query = state["original_query"]
    log = logger.bind(node="decomposer")

    llm = LLMFactory.create_mini()
    response = await llm.complete(
        messages=[
            {"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}
        ],
        max_tokens=200,
        temperature=0,
    )

    sub_questions = [
        line.strip()
        for line in response.strip().split("\n")
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
