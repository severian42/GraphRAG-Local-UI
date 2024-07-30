import json
from fastapi import FastAPI, HTTPException
import uvicorn
import httpx
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()

OLLAMA_URL = "http://localhost:11434"  # Default Ollama URL

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str

class EmbeddingResponse(BaseModel):
    object: str
    data: List[dict]
    model: str
    usage: dict

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    async with httpx.AsyncClient() as client:
        if isinstance(request.input, str):
            request.input = [request.input]

        ollama_requests = [{"model": request.model, "prompt": text} for text in request.input]

        embeddings = []


        for i, ollama_request in enumerate(ollama_requests):
            response = await client.post(f"{OLLAMA_URL}/api/embeddings", json=ollama_request)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Ollama API error")
            
            result = response.json()
            embeddings.append({
                "object": "embedding",
                "embedding": result["embedding"],
                "index": i
            })
            

        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=request.model,
            
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the embedding proxy server")
    parser.add_argument("--port", type=int, default=11435, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="URL of the Ollama server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    OLLAMA_URL = args.host
    uvicorn.run("embedding_proxy:app", host="0.0.0.0", port=args.port, reload=args.reload)
