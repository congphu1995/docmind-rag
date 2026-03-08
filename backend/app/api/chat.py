from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from backend.app.api.dependencies import get_current_user
from backend.app.core.logging import logger
from backend.app.schemas.chat import ChatRequest
from backend.app.services.rag import RAGService

router = APIRouter()

_rag_service = RAGService()


@router.post("/")
async def chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    try:
        if request.stream:
            return EventSourceResponse(
                _stream_response(request),
                media_type="text/event-stream",
            )

        result = await _rag_service.query(request)
        return result

    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(request: ChatRequest):
    async for event in _rag_service.stream_query(request):
        yield {"data": event}
