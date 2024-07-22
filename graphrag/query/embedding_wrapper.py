from graphrag.llm.openai.openai_embeddings_llm import OpenAIEmbeddingsLLM
import logging

class EmbeddingWrapper:
    def __init__(self, embedder: OpenAIEmbeddingsLLM):
        self.embedder = embedder

    def embed(self, text: str):
        result = self.embedder.execute_llm_sync([text])
        if result and len(result) > 0:
            return result[0]
        else:
            logging.error("Failed to generate embedding")
            return None