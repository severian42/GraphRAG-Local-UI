# GraphRAG API

This README provides a detailed guide on the `api.py` file, which serves as the API interface for the GraphRAG (Graph Retrieval-Augmented Generation) system. GraphRAG is a powerful tool that combines graph-based knowledge representation with retrieval-augmented generation techniques to provide context-aware responses to queries.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [API Endpoints](#api-endpoints)
4. [Data Models](#data-models)
5. [Core Functionality](#core-functionality)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

## Overview

The `api.py` file implements a FastAPI-based server that provides various endpoints for interacting with the GraphRAG system. It supports different types of queries, including direct chat, GraphRAG-specific queries, DuckDuckGo searches, and a combined full-model search.

Key features:
- Multiple query types (local and global searches)
- Context caching for improved performance
- Background tasks for long-running operations
- Customizable settings through environment variables and config files
- Integration with external services (e.g., Ollama for LLM interactions)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   Create a `.env` file in the `indexing` directory with the following variables:
   ```
   LLM_API_BASE=<your_llm_api_base_url>
   LLM_MODEL=<your_llm_model>
   LLM_PROVIDER=<llm_provider>
   EMBEDDINGS_API_BASE=<your_embeddings_api_base_url>
   EMBEDDINGS_MODEL=<your_embeddings_model>
   EMBEDDINGS_PROVIDER=<embeddings_provider>
   INPUT_DIR=./indexing/output
   ROOT_DIR=indexing
   API_PORT=8012
   ```

3. Run the API server:
   ```
   python api.py --host 0.0.0.0 --port 8012
   ```

## API Endpoints

### `/v1/chat/completions` (POST)
Main endpoint for chat completions. Supports different models:
- `direct-chat`: Direct interaction with the LLM
- `graphrag-local-search:latest`: Local search using GraphRAG
- `graphrag-global-search:latest`: Global search using GraphRAG
- `duckduckgo-search:latest`: Web search using DuckDuckGo
- `full-model:latest`: Combined search using all available models

### `/v1/prompt_tune` (POST)
Initiates prompt tuning process in the background.

### `/v1/prompt_tune_status` (GET)
Retrieves the status and logs of the prompt tuning process.

### `/v1/index` (POST)
Starts the indexing process for GraphRAG in the background.

### `/v1/index_status` (GET)
Retrieves the status and logs of the indexing process.

### `/health` (GET)
Health check endpoint.

### `/v1/models` (GET)
Lists available models.

## Data Models

The API uses several Pydantic models for request and response handling:

- `Message`: Represents a chat message with role and content.
- `QueryOptions`: Options for GraphRAG queries, including query type, preset, and community level.
- `ChatCompletionRequest`: Request model for chat completions.
- `ChatCompletionResponse`: Response model for chat completions.
- `PromptTuneRequest`: Request model for prompt tuning.
- `IndexingRequest`: Request model for indexing.

## Core Functionality

### Context Loading
The `load_context` function loads necessary data for GraphRAG queries, including entities, relationships, reports, text units, and covariates.

### Search Engine Setup
`setup_search_engines` initializes both local and global search engines using the loaded context data.

### Query Execution
Different query types are handled by separate functions:
- `run_direct_chat`: Sends queries directly to the LLM.
- `run_graphrag_query`: Executes GraphRAG queries (local or global).
- `run_duckduckgo_search`: Performs web searches using DuckDuckGo.
- `run_full_model_search`: Combines results from all search types.

### Background Tasks
Long-running tasks like prompt tuning and indexing are executed as background tasks to prevent blocking the API.

## Usage Examples

### Sending a GraphRAG Query
```python
import requests

url = "http://localhost:8012/v1/chat/completions"
payload = {
    "model": "graphrag-local-search:latest",
    "messages": [{"role": "user", "content": "What is GraphRAG?"}],
    "query_options": {
        "query_type": "local-search",
        "selected_folder": "your_indexed_folder",
        "community_level": 2,
        "response_type": "Multiple Paragraphs"
    }
}
response = requests.post(url, json=payload)
print(response.json())
```

### Starting Indexing Process
```python
import requests

url = "http://localhost:8012/v1/index"
payload = {
    "llm_model": "your_llm_model",
    "embed_model": "your_embed_model",
    "root": "./indexing",
    "verbose": True,
    "emit": ["parquet", "csv"]
}
response = requests.post(url, json=payload)
print(response.json())
```

## Configuration

The API can be configured through:
1. Environment variables
2. A `config.yaml` file (path specified by `GRAPHRAG_CONFIG` environment variable)
3. Command-line arguments when starting the server

Key configuration options:
- `llm_model`: The language model to use
- `embedding_model`: The embedding model for vector representations
- `community_level`: Depth of community analysis in GraphRAG
- `token_limit`: Maximum tokens for context
- `api_key`: API key for LLM service
- `api_base`: Base URL for LLM API
- `api_type`: Type of API (e.g., "openai")

## Troubleshooting

1. If you encounter connection errors with Ollama, ensure the service is running and accessible.
2. For "context loading failed" errors, check that the indexed data is present in the specified output folder.
3. If prompt tuning or indexing processes fail, review the logs using the respective status endpoints.
4. For performance issues, consider adjusting the `community_level` and `token_limit` settings.

For more detailed information on GraphRAG's indexing and querying processes, refer to the official GraphRAG documentation.