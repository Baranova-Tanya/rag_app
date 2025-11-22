import uvicorn
import supabase

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pipeline import (retrieve_context,
                      generate_full_answer,
                      save_feedback)

app = FastAPI(
    title="RAG API",
    description="API for retrieving relevant document context and generating OpenAI responses.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str = Field(
        default=...,
        description="User's question or search query",
        example="How to use Scrum?"
    )
    match_count: int = Field(
        default=3,
        description="Number of most similar document chunks to include in context",
        example=3
    )

class QueryResponse(BaseModel):
    answer: str

class FeedbackResponse(BaseModel):
    status: str
    message: str


@app.post("/ask-full", response_model=QueryResponse)
async def ask_full(req: QueryRequest):
    chunks = await retrieve_context(req.query, req.match_count)
    answer = await generate_full_answer(req.query, chunks)
    return QueryResponse(answer=answer)

@app.post("/feedback")
async def save_feedback_thanks(request: Request):
    request = await request.json()
    answer = await save_feedback(request)
    return FeedbackResponse(status=answer["status"],
                            message=answer["message"])

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )