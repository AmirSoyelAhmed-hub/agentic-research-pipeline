from fastapi import FastAPI
from pydantic import BaseModel

from agents.research_agent import research_agent, langfuse_handler

app = FastAPI(title="RL Research Agent API")


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    url: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    result = research_agent.invoke(
        {"question": request.question},
        config={"callbacks": [langfuse_handler]},
    )
    sources = [
        Source(title=c["title"], url=c["url"]) for c in result["retrieved_chunks"]
    ]
    return AnswerResponse(answer=result["answer"], sources=sources)