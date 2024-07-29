# Using Ollama Embeddings with GraphRAG: A Quick Guide

## Problem

GraphRAG is designed to work with OpenAI-compatible APIs for both language models and embeddings and Ollama currently has their own way of doing embeddings.

## Solution: Embeddings Proxy

To bridge this gap, let's use an embeddings proxy. This proxy acts as a middleware between GraphRAG and Ollama, translating Ollama's embedding responses into a format that GraphRAG expects.

## Use the Embeddings Proxy

1. **Set up the proxy:**
   - Save the provided `embedding_proxy.py` script to your project directory.
   - Install required dependencies (not needed if you've already done this in the normal setup): `pip install fastapi uvicorn httpx`

2. **Run the proxy:**
   ```bash
   python embedding_proxy.py --port 11435 --host http://localhost:11434
   ```
   This starts the proxy on port 11435, connecting to Ollama at localhost:11434.

3. **Configure GraphRAG:**
   Update your `settings.yaml` file to use the proxy for embeddings:

   ```yaml
   embeddings:
     llm:
       api_key: ${GRAPHRAG_API_KEY}
       type: openai_embedding
       model: nomic-embed-text:latest
       api_base: http://localhost:11435  # Point to your proxy
   ```

4. **Run GraphRAG:**
   With the proxy running and the configuration updated, you can now run GraphRAG as usual. It will use Ollama for embeddings through the proxy.
