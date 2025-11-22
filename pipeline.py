import numpy as np
import openai
from typing import List, Dict, AsyncGenerator
from openai import AsyncOpenAI
from supabase import create_client
from rag_utils import search_similar_chunks
from supabase_utils import SUPABASE_URL, SUPABASE_KEY

# Initializing clients
client = AsyncOpenAI()
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === 1. Embedding request ===
async def embed_query(query: str) -> np.ndarray:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    embedding_vector = np.array(response.data[0].embedding, dtype=np.float32)
    print('Embedding is over')
    return embedding_vector


# === 2. Search for similar chunks ===
async def retrieve_context(query: str, match_count: int = 5) -> List[Dict]:
    embedding_vector = await embed_query(query)
    results = search_similar_chunks(supabase_client, embedding_vector, match_count)
    print('Context found')
    return results


# === 3a. Generating a response in streaming mode ===
async def generate_streaming_answer(
    user_query: str,
    chunks: List[Dict],
    model: str = "gpt-5-mini"
) -> AsyncGenerator[str, None]:
    context_text = ""
    for chunk in sorted(chunks, key=lambda x: -x.get("similarity", 0)):
        text = chunk.get("paragraph_text", "")
        url = chunk.get("url", "")
        context_text += f"Source: {url}\n{text}\n\n"

    system_prompt = """
You are an AI assistant that provides information strictly based on the internal corporate wiki documents provided in the context below. 
Your task is to answer employee questions using ONLY the given context — do not use any outside knowledge or make assumptions beyond what is explicitly stated. 
If the user asks a question in a particular language, always respond in that same language. 

Requirements for your response:
1. Use ONLY the provided context as your information source.  
2. Provide a clear, concise, and complete answer in a single message — this is not a continuing chat.  
3. Always explicitly cite the sources of your information at the end of your answer using this format:  
   **Sources:** [Title of Wiki Page](URL)  
4. If the answer cannot be found in the provided context, clearly state:  
   "I couldn’t find this information in the available wiki documents."  
5. Do NOT include any disclaimers about being an AI model.  
6. Do NOT generate follow-up questions or ask for clarification — give your best possible single complete answer.  

"""
    user_prompt = f"User’s question goes here: {user_query}\nContext:\n{context_text}"

    async with client.chat.completions.stream(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ) as stream:
        async for event in stream:
            # If there is a delta, stream immediately
            if getattr(event, "type", None) == "message.delta":
                content = getattr(event.delta, "content", "") or ""
                if content.strip():
                    yield f"data: {content}\n\n"
            # If there is completion, but there are no deltas, send the full text
            elif getattr(event, "type", None) == "message.completed":
                final_text = getattr(event, "message", {}).get("content", "")
                if final_text.strip():
                    yield f"data: {final_text}\n\n"
                yield "data: [END]\n\n"
                break

# === 3b. Generating a response without streaming ===
async def generate_full_answer(
    user_query: str,
    chunks: List[Dict],
    model: str = "gpt-5-mini"
) -> str:
    print("🟢 [FULL] Regular OpenAI query is running...")

    context_text = ""
    for chunk in sorted(chunks, key=lambda x: -x["similarity"]):
        text = chunk.get("paragraph_text", "")
        url = chunk.get("url", "")
        context_text += f"Source: {url}\n{text}\n\n"

    system_prompt = """
You are an AI assistant that provides information strictly based on the internal corporate wiki documents provided in the context below. 
Your task is to answer employee questions using ONLY the given context — do not use any outside knowledge or make assumptions beyond what is explicitly stated. 
If the user asks a question in a particular language, always respond in that same language. 

Requirements for your response:
1. Use ONLY the provided context as your information source.  
2. Provide a clear, concise, and complete answer in a single message — this is not a continuing chat.  
3. Always explicitly cite the sources of your information at the end of your answer using this format:  
   **Sources:** [Title of Wiki Page](URL)  
4. If the answer cannot be found in the provided context, clearly state:  
   "I couldn’t find this information in the available wiki documents."  
5. Do NOT include any disclaimers about being an AI model.  
6. Do NOT generate follow-up questions or ask for clarification — give your best possible single complete answer.  

"""

    user_prompt = f"User’s question goes here: {user_query}\nContext:\n{context_text}"

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    full_answer = response.choices[0].message.content
    print("✅ [FULL] Reply received")
    return full_answer

# === 4. Streaming Response Pipeline ===
async def rag_stream_pipeline(user_query: str):
    chunks = await retrieve_context(user_query)
    async for token in generate_streaming_answer(user_query, chunks):
        yield token

# === 5. Saving feedback ===
async def save_feedback(request):

    question = request.get("question")
    answer = request.get("answer")
    is_helpful = request.get("is_helpful")

    if not question or not answer:
        return {"status": "error", "message": "Missing question or answer"}

    try:

        response = supabase_client.table("feedback").insert({
            "query_text": question,
            "answer_text": answer,
            "is_helpful": is_helpful
        }).execute()
        return {"status": "ok", "message": "Thanks for the feedback"}
    except Exception as e:
        return {"status": "error", "message": f"Supabase error: {e}"}