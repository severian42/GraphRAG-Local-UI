import logging
import aiohttp
import asyncio
import requests
from typing import List, Union, Dict, Any
import ollama
from typing_extensions import Unpack
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes

class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    def __init__(self, client: OpenAIClientTypes, configuration: Union[OpenAIConfiguration, Dict[str, Any]]):
        self._client = client
        if isinstance(configuration, OpenAIConfiguration):
            self._configuration = configuration
            self._model = configuration.model
        elif isinstance(configuration, dict):
            self._configuration = configuration
            self._model = configuration.get("model")
        else:
            raise TypeError("Configuration must be either OpenAIConfiguration or a dictionary")

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self._model,
            **(kwargs.get("model_parameters") or {}),
        }
        
        try:
            if self._is_ollama_model(args["model"]):
                embeddings = await asyncio.gather(*[self._get_ollama_embedding(inp, args) for inp in input])
            else:  # OpenAI compatible
                embeddings = await self._get_openai_embeddings(input, args)
        except Exception as e:
            logging.error(f"Error getting embeddings from {args['model']}: {str(e)}")
            if not self._is_ollama_model(args["model"]):
                logging.info("Falling back to Ollama embeddings")
                try:
                    embeddings = await asyncio.gather(*[self._get_ollama_embedding(inp, args) for inp in input])
                except Exception as e:
                    logging.error(f"Error getting Ollama embeddings: {str(e)}")
                    return None

        return [emb for emb in embeddings if emb is not None]

    async def _get_ollama_embedding(self, text: str, args: Dict[str, Any]) -> Union[List[float], None]:
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, ollama.embeddings, args["model"], text)
            return embedding["embedding"]
        except Exception as e:
            logging.error(f"Error getting Ollama embedding: {str(e)}")
            return None

    async def _get_openai_embeddings(self, input: List[str], args: Dict[str, Any]) -> List[Union[List[float], None]]:
        endpoint = f"{self.get_api_base().rstrip('/')}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_api_key()}" if self.get_api_key() != "dummy_key" else ""
        }
        async with aiohttp.ClientSession() as session:
            tasks = [self._get_openai_embedding(session, endpoint, inp, headers, args) for inp in input]
            return await asyncio.gather(*tasks)

    async def _get_openai_embedding(self, session: aiohttp.ClientSession, endpoint: str, text: str, headers: Dict[str, str], args: Dict[str, Any]) -> Union[List[float], None]:
        data = {
            "model": args["model"],
            "input": text,
            **{k: v for k, v in args.items() if k != "model"}
        }
        async with session.post(endpoint, json=data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result["data"][0]["embedding"]
            else:
                logging.error(f"Error getting OpenAI compatible embedding: {await response.text()}")
                return None

    def execute_llm_sync(self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]) -> EmbeddingOutput | None:
        args = {
            "model": self._model,
            **(kwargs.get("model_parameters") or {}),
        }
        
        try:
            if self._is_ollama_model(args["model"]):
                embeddings = [self._get_ollama_embedding_sync(inp, args) for inp in input]
            else:  # OpenAI compatible
                embeddings = self._get_openai_embeddings_sync(input, args)
        except Exception as e:
            logging.error(f"Error getting embeddings from {args['model']}: {str(e)}")
            if not self._is_ollama_model(args["model"]):
                logging.info("Falling back to Ollama embeddings")
                try:
                    embeddings = [self._get_ollama_embedding_sync(inp, args) for inp in input]
                except Exception as e:
                    logging.error(f"Error getting Ollama embeddings: {str(e)}")
                    return None

        return [emb for emb in embeddings if emb is not None]

    def _get_ollama_embedding_sync(self, text: str, args: Dict[str, Any]) -> Union[List[float], None]:
        try:
            embedding = ollama.embeddings(model=args["model"], prompt=text)
            return embedding["embedding"]
        except Exception as e:
            logging.error(f"Error getting Ollama embedding: {str(e)}")
            return None

    def _get_openai_embeddings_sync(self, input: List[str], args: Dict[str, Any]) -> List[Union[List[float], None]]:
        endpoint = f"{self.get_api_base().rstrip('/')}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_api_key()}" if self.get_api_key() != "dummy_key" else ""
        }
        return [self._get_openai_embedding_sync(endpoint, inp, headers, args) for inp in input]

    def _get_openai_embedding_sync(self, endpoint: str, text: str, headers: Dict[str, str], args: Dict[str, Any]) -> Union[List[float], None]:
        data = {
            "model": args["model"],
            "input": text,
            **{k: v for k, v in args.items() if k != "model"}
        }
        response = requests.post(endpoint, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            logging.error(f"Error getting OpenAI compatible embedding: {response.text}")
            return None

    def _is_ollama_model(self, model: str) -> bool:
        return model.lower().startswith("ollama:")

    def get_api_base(self) -> str:
        if isinstance(self._configuration, OpenAIConfiguration):
            return self._configuration.api_base
        elif isinstance(self._configuration, dict):
            return self._configuration.get("api_base")
        else:
            raise TypeError("Invalid configuration type")

    def get_api_key(self) -> str:
        if isinstance(self._configuration, OpenAIConfiguration):
            return self._configuration.api_key
        elif isinstance(self._configuration, dict):
            return self._configuration.get("api_key", "dummy_key")
        else:
            raise TypeError("Invalid configuration type")