#openai_embeddings_llm.py

from typing_extensions import Unpack
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
import aiohttp
import asyncio
import requests
from typing import List, Union, Dict, Any
import ollama

class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self._client = client
        self._configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self._configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        
        if self._configuration.provider.lower() == "ollama":
            embeddings = await asyncio.gather(*[self._get_ollama_embedding(inp) for inp in input])
        else:  # OpenAI compatible
            endpoint = f"{self._configuration.api_base.rstrip('/')}/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._configuration.api_key}" if self._configuration.api_key else ""
            }
            async with aiohttp.ClientSession() as session:
                tasks = [self._get_openai_embedding(session, endpoint, inp, headers) for inp in input]
                embeddings = await asyncio.gather(*tasks)

        return [emb for emb in embeddings if emb is not None]

    async def _get_ollama_embedding(self, text: str) -> Union[List[float], None]:
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, ollama.embeddings, self._configuration.model, text)
            return embedding["embedding"]
        except Exception as e:
            print(f"Error getting Ollama embedding: {str(e)}")
            return None

    async def _get_openai_embedding(self, session: aiohttp.ClientSession, endpoint: str, text: str, headers: Dict[str, str]) -> Union[List[float], None]:
        data = {
            "model": self._configuration.model,
            "input": text
        }
        async with session.post(endpoint, json=data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result["data"][0]["embedding"]
            else:
                print(f"Error getting OpenAI compatible embedding: {await response.text()}")
                return None

    def execute_llm_sync(self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]) -> EmbeddingOutput | None:
        args = {
            "model": self._configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        
        if self._configuration.provider.lower() == "ollama":
            embeddings = [self._get_ollama_embedding_sync(inp) for inp in input]
        else:  # OpenAI compatible
            endpoint = f"{self._configuration.api_base.rstrip('/')}/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._configuration.api_key}" if self._configuration.api_key else ""
            }
            embeddings = [self._get_openai_embedding_sync(endpoint, inp, headers) for inp in input]

        return [emb for emb in embeddings if emb is not None]

    def _get_ollama_embedding_sync(self, text: str) -> Union[List[float], None]:
        try:
            embedding = ollama.embeddings(model=self._configuration.model, prompt=text)
            return embedding["embedding"]
        except Exception as e:
            print(f"Error getting Ollama embedding: {str(e)}")
            return None

    def _get_openai_embedding_sync(self, endpoint: str, text: str, headers: Dict[str, str]) -> Union[List[float], None]:
        data = {
            "model": self._configuration.model,
            "input": text
        }
        response = requests.post(endpoint, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            print(f"Error getting OpenAI compatible embedding: {response.text}")
            return None