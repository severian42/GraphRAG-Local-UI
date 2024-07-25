from dotenv import load_dotenv
import os
import asyncio
import tempfile
from collections import deque
import time
import uuid
import json
import re
import pandas as pd
import tiktoken
import logging
import yaml
import shutil
from fastapi import Body
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from web import DuckDuckGoSearchAPIWrapper
from functools import lru_cache
import requests
import subprocess
import argparse

# GraphRAG related imports
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('indexing/.env')
LLM_API_BASE = os.getenv('LLM_API_BASE', '')
LLM_MODEL = os.getenv('LLM_MODEL')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').lower()
EMBEDDINGS_API_BASE = os.getenv('EMBEDDINGS_API_BASE', '')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL')
EMBEDDINGS_PROVIDER = os.getenv('EMBEDDINGS_PROVIDER', 'openai').lower()
INPUT_DIR = os.getenv('INPUT_DIR', './indexing/output')
ROOT_DIR = os.getenv('ROOT_DIR', 'indexing')
PORT = int(os.getenv('API_PORT', 8012))
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2

# Global variables for storing search engines and question generator
local_search_engine = None
global_search_engine = None
question_generator = None

# Data models
class Message(BaseModel):
    role: str
    content: str

class QueryOptions(BaseModel):
    query_type: str
    preset: Optional[str] = None
    community_level: Optional[int] = None
    response_type: Optional[str] = None
    custom_cli_args: Optional[str] = None
    selected_folder: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    query_options: Optional[QueryOptions] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None

def list_output_folders():
    return [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]

def list_folder_contents(folder_name):
    folder_path = os.path.join(INPUT_DIR, folder_name, "artifacts")
    if not os.path.exists(folder_path):
        return []
    return [item for item in os.listdir(folder_path) if item.endswith('.parquet')]

def normalize_api_base(api_base: str) -> str:
    """Normalize the API base URL by removing trailing slashes and /v1 or /api suffixes."""
    api_base = api_base.rstrip('/')
    if api_base.endswith('/v1') or api_base.endswith('/api'):
        api_base = api_base[:-3]
    return api_base

def get_models_endpoint(api_base: str, api_type: str) -> str:
    """Get the appropriate models endpoint based on the API type."""
    normalized_base = normalize_api_base(api_base)
    if api_type.lower() == 'openai':
        return f"{normalized_base}/v1/models"
    elif api_type.lower() == 'azure':
        return f"{normalized_base}/openai/deployments?api-version=2022-12-01"
    else:  # For other API types (e.g., local LLMs)
        return f"{normalized_base}/models"

async def fetch_available_models(settings: Dict[str, Any]) -> List[str]:
    """Fetch available models from the API."""
    api_base = settings['api_base']
    api_type = settings['api_type']
    api_key = settings['api_key']

    models_endpoint = get_models_endpoint(api_base, api_type)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        response = requests.get(models_endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if api_type.lower() == 'openai':
            return [model['id'] for model in data['data']]
        elif api_type.lower() == 'azure':
            return [model['id'] for model in data['value']]
        else:
            # Adjust this based on the actual response format of your local LLM API
            return [model['name'] for model in data['models']]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching models: {str(e)}")
        return []

def load_settings():
    config_path = os.getenv('GRAPHRAG_CONFIG', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    else:
        config = {}

    settings = {
        'llm_model': os.getenv('LLM_MODEL', config.get('llm_model')),
        'embedding_model': os.getenv('EMBEDDINGS_MODEL', config.get('embedding_model')),
        'community_level': int(os.getenv('COMMUNITY_LEVEL', config.get('community_level', 2))),
        'token_limit': int(os.getenv('TOKEN_LIMIT', config.get('token_limit', 4096))),
        'api_key': os.getenv('GRAPHRAG_API_KEY', config.get('api_key')),
        'api_base': os.getenv('LLM_API_BASE', config.get('api_base')),
        'embeddings_api_base': os.getenv('EMBEDDINGS_API_BASE', config.get('embeddings_api_base')),
        'api_type': os.getenv('API_TYPE', config.get('api_type', 'openai')),
    }

    return settings

    return settings

async def setup_llm_and_embedder(settings):
    logger.info("Setting up LLM and embedder")
    try:
        llm = ChatOpenAI(
            api_key=settings['api_key'],
            api_base=f"{settings['api_base']}/v1",
            model=settings['llm_model'],
            api_type=OpenaiApiType[settings['api_type'].capitalize()],
            max_retries=20,
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        text_embedder = OpenAIEmbedding(
            api_key=settings['api_key'],
            api_base=f"{settings['embeddings_api_base']}/v1",
            api_type=OpenaiApiType[settings['api_type'].capitalize()],
            model=settings['embedding_model'],
            deployment_name=settings['embedding_model'],
            max_retries=20,
        )

        logger.info("LLM and embedder setup complete")
        return llm, token_encoder, text_embedder
    except Exception as e:
        logger.error(f"Error setting up LLM and embedder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set up LLM and embedder: {str(e)}")

async def load_context(selected_folder, settings):
    """
    Load context data including entities, relationships, reports, text units, and covariates
    """
    logger.info("Loading context data")
    try:
        input_dir = os.path.join(INPUT_DIR, selected_folder, "artifacts")
        entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")
        entities = read_indexer_entities(entity_df, entity_embedding_df, settings['community_level'])

        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

        relationship_df = pd.read_parquet(f"{input_dir}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)

        report_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

        text_unit_df = pd.read_parquet(f"{input_dir}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)

        covariate_df = pd.read_parquet(f"{input_dir}/{COVARIATE_TABLE}.parquet")
        claims = read_indexer_covariates(covariate_df)
        logger.info(f"Number of claim records: {len(claims)}")
        covariates = {"claims": claims}

        logger.info("Context data loading complete")
        return entities, relationships, reports, text_units, description_embedding_store, covariates
    except Exception as e:
        logger.error(f"Error loading context data: {str(e)}")
        raise

async def setup_search_engines(llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
                               description_embedding_store, covariates):
    """
    Set up local and global search engines
    """
    logger.info("Setting up search engines")

    # Set up local search engine
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    local_llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    local_search_engine = LocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    # Set up global search engine
    global_context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    global_search_engine = GlobalSearch(
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    logger.info("Search engines setup complete")
    return local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params

def format_response(response):
    """
    Format the response by adding appropriate line breaks and paragraph separations.
    """
    paragraphs = re.split(r'\n{2,}', response)

    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # This is a code block
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')

        formatted_paragraphs.append(para.strip())

    return '\n\n'.join(formatted_paragraphs)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings
    try:
        logger.info("Loading settings...")
        settings = load_settings()
        logger.info("Settings loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        raise

    yield

    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Create a cache for loaded contexts
context_cache = {}

@lru_cache()
def get_settings():
    return load_settings()

async def get_context(selected_folder: str, settings: dict = Depends(get_settings)):
    if selected_folder not in context_cache:
        try:
            llm, token_encoder, text_embedder = await setup_llm_and_embedder(settings)
            entities, relationships, reports, text_units, description_embedding_store, covariates = await load_context(selected_folder, settings)
            local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params = await setup_search_engines(
                llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
                description_embedding_store, covariates
            )
            question_generator = LocalQuestionGen(
                llm=llm,
                context_builder=local_context_builder,
                token_encoder=token_encoder,
                llm_params=local_llm_params,
                context_builder_params=local_context_params,
            )
            context_cache[selected_folder] = {
                "local_search_engine": local_search_engine,
                "global_search_engine": global_search_engine,
                "question_generator": question_generator
            }
        except Exception as e:
            logger.error(f"Error loading context for folder {selected_folder}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load context for folder {selected_folder}")
    
    return context_cache[selected_folder]

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        logger.info(f"Received request for model: {request.model}")
        if request.model == "direct-chat":
            logger.info("Routing to direct chat")
            return await run_direct_chat(request)
        elif request.model.startswith("graphrag-"):
            logger.info("Routing to GraphRAG query")
            if not request.query_options or not request.query_options.selected_folder:
                raise HTTPException(status_code=400, detail="Selected folder is required for GraphRAG queries")
            return await run_graphrag_query(request)
        elif request.model == "duckduckgo-search:latest":
            logger.info("Routing to DuckDuckGo search")
            return await run_duckduckgo_search(request)
        elif request.model == "full-model:latest":
            logger.info("Routing to full model search")
            return await run_full_model_search(request)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid model specified: {request.model}")
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def run_direct_chat(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        if not LLM_API_BASE:
            raise ValueError("LLM_API_BASE environment variable is not set")
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": False
        }
        
        # Optional parameters
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        full_url = f"{normalize_api_base(LLM_API_BASE)}/v1/chat/completions"
        
        logger.info(f"Sending request to: {full_url}")
        logger.info(f"Payload: {payload}")
        
        try:
            response = requests.post(full_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_ex:
            logger.error(f"Request to LLM API failed: {str(req_ex)}")
            if isinstance(req_ex, requests.exceptions.ConnectionError):
                raise HTTPException(status_code=503, detail="Unable to connect to LLM API. Please check your API settings.")
            elif isinstance(req_ex, requests.exceptions.Timeout):
                raise HTTPException(status_code=504, detail="Request to LLM API timed out")
            else:
                raise HTTPException(status_code=500, detail=f"Request to LLM API failed: {str(req_ex)}")
        
        result = response.json()
        logger.info(f"Received response: {result}")
        
        content = result['choices'][0]['message']['content']
        
        return ChatCompletionResponse(
            model=LLM_MODEL,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content
                    ),
                    finish_reason=None
                )
            ],
            usage=None
        )
    except HTTPException as he:
        logger.error(f"HTTP Exception in direct chat: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in direct chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during the direct chat: {str(e)}")

def get_embeddings(text: str) -> List[float]:
    settings = load_settings()
    embeddings_api_base = settings['embeddings_api_base']
    
    headers = {"Content-Type": "application/json"}
    
    if EMBEDDINGS_PROVIDER == 'ollama':
        payload = {
            "model": EMBEDDINGS_MODEL,
            "prompt": text
        }
        full_url = f"{embeddings_api_base}/api/embeddings"
    else:  # OpenAI-compatible API
        payload = {
            "model": EMBEDDINGS_MODEL,
            "input": text
        }
        full_url = f"{embeddings_api_base}/v1/embeddings"
    
    try:
        response = requests.post(full_url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as req_ex:
        logger.error(f"Request to Embeddings API failed: {str(req_ex)}")
        raise HTTPException(status_code=500, detail=f"Failed to get embeddings: {str(req_ex)}")
    
    result = response.json()
    
    if EMBEDDINGS_PROVIDER == 'ollama':
        return result['embedding']
    else:
        return result['data'][0]['embedding']
    

async def run_graphrag_query(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        query_options = request.query_options
        query = request.messages[-1].content  # Get the last user message as the query
        
        cmd = ["python", "-m", "graphrag.query"]
        cmd.extend(["--data", f"./indexing/output/{query_options.selected_folder}/artifacts"])
        cmd.extend(["--method", query_options.query_type.split('-')[1]])  # 'global' or 'local'
        
        if query_options.community_level:
            cmd.extend(["--community_level", str(query_options.community_level)])
        if query_options.response_type:
            cmd.extend(["--response_type", query_options.response_type])
        
        # Handle preset CLI args
        if query_options.preset and query_options.preset != "Custom Query":
            preset_args = get_preset_args(query_options.preset)
            cmd.extend(preset_args)
        
        # Handle custom CLI args
        if query_options.custom_cli_args:
            cmd.extend(query_options.custom_cli_args.split())
        
        cmd.append(query)

        logger.info(f"Executing GraphRAG query: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"GraphRAG query failed: {result.stderr}")

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=result.stdout
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
    except Exception as e:
        logger.error(f"Error in GraphRAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the GraphRAG query: {str(e)}")
    

def get_preset_args(preset: str) -> List[str]:
    preset_args = {
        "Default Global Search": ["--community_level", "2", "--response_type", "Multiple Paragraphs"],
        "Default Local Search": ["--community_level", "2", "--response_type", "Multiple Paragraphs"],
        "Detailed Global Analysis": ["--community_level", "3", "--response_type", "Multi-Page Report"],
        "Detailed Local Analysis": ["--community_level", "3", "--response_type", "Multi-Page Report"],
        "Quick Global Summary": ["--community_level", "1", "--response_type", "Single Paragraph"],
        "Quick Local Summary": ["--community_level", "1", "--response_type", "Single Paragraph"],
        "Global Bullet Points": ["--community_level", "2", "--response_type", "List of 3-7 Points"],
        "Local Bullet Points": ["--community_level", "2", "--response_type", "List of 3-7 Points"],
        "Comprehensive Global Report": ["--community_level", "4", "--response_type", "Multi-Page Report"],
        "Comprehensive Local Report": ["--community_level", "4", "--response_type", "Multi-Page Report"],
        "High-Level Global Overview": ["--community_level", "1", "--response_type", "Single Page"],
        "High-Level Local Overview": ["--community_level", "1", "--response_type", "Single Page"],
        "Focused Global Insight": ["--community_level", "3", "--response_type", "Single Paragraph"],
        "Focused Local Insight": ["--community_level", "3", "--response_type", "Single Paragraph"],
    }
    return preset_args.get(preset, [])

ddg_search = DuckDuckGoSearchAPIWrapper(max_results=5)

async def run_duckduckgo_search(request: ChatCompletionRequest) -> ChatCompletionResponse:
    query = request.messages[-1].content
    results = ddg_search.results(query, max_results=5)
    
    if not results:
        content = "No results found for the given query."
    else:
        content = "DuckDuckGo Search Results:\n\n"
        for result in results:
            content += f"Title: {result['title']}\n"
            content += f"Snippet: {result['snippet']}\n"
            content += f"Link: {result['link']}\n"
            if 'date' in result:
                content += f"Date: {result['date']}\n"
            if 'source' in result:
                content += f"Source: {result['source']}\n"
            content += "\n"

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content=content
                ),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )

async def run_full_model_search(request: ChatCompletionRequest) -> ChatCompletionResponse:
    query = request.messages[-1].content
    
    # Run all search types
    graphrag_global = await run_graphrag_query(ChatCompletionRequest(model="graphrag-global-search:latest", messages=request.messages, query_options=request.query_options))
    graphrag_local = await run_graphrag_query(ChatCompletionRequest(model="graphrag-local-search:latest", messages=request.messages, query_options=request.query_options))
    duckduckgo = await run_duckduckgo_search(request)
    
    # Combine results
    combined_content = f"""Full Model Search Results:

Global Search:
{graphrag_global.choices[0].message.content}

Local Search:
{graphrag_local.choices[0].message.content}

DuckDuckGo Search:
{duckduckgo.choices[0].message.content}
"""

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content=combined_content
                ),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models():
    settings = load_settings()
    try:
        api_models = await fetch_available_models(settings)
    except Exception as e:
        logger.error(f"Error fetching API models: {str(e)}")
        api_models = []

    # Include the hardcoded models
    hardcoded_models = [
        {"id": "graphrag-local-search:latest", "object": "model", "owned_by": "graphrag"},
        {"id": "graphrag-global-search:latest", "object": "model", "owned_by": "graphrag"},
        {"id": "duckduckgo-search:latest", "object": "model", "owned_by": "duckduckgo"},
        {"id": "full-model:latest", "object": "model", "owned_by": "combined"},
    ]

    # Combine API models with hardcoded models
    all_models = [{"id": model, "object": "model", "owned_by": "api"} for model in api_models] + hardcoded_models

    return JSONResponse(content={"data": all_models})

class PromptTuneRequest(BaseModel):
    root: str = "./{ROOT_DIR}"
    domain: Optional[str] = None
    method: str = "random"
    limit: int = 15
    language: Optional[str] = None
    max_tokens: int = 2000
    chunk_size: int = 200
    no_entity_types: bool = False
    output: str = "./{ROOT_DIR}/prompts"

class PromptTuneResponse(BaseModel):
    status: str
    message: str

# Global variable to store the latest logs
prompt_tune_logs = deque(maxlen=100) 

async def run_prompt_tuning(request: PromptTuneRequest):
    cmd = ["python", "-m", "graphrag.prompt_tune"]
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_output:
        # Expand environment variables in the root path
        root_path = os.path.expandvars(request.root)
        
        cmd.extend(["--root", root_path])
        cmd.extend(["--method", request.method])
        cmd.extend(["--limit", str(request.limit)])
        
        if request.domain:
            cmd.extend(["--domain", request.domain])
        
        if request.language:
            cmd.extend(["--language", request.language])
        
        cmd.extend(["--max-tokens", str(request.max_tokens)])
        cmd.extend(["--chunk-size", str(request.chunk_size)])
        
        if request.no_entity_types:
            cmd.append("--no-entity-types")
        
        # Use the temporary directory for output
        cmd.extend(["--output", temp_output])

        logger.info(f"Executing prompt tuning command: {' '.join(cmd)}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line = line.decode().strip()
                    prompt_tune_logs.append(line)
                    logger.info(line)

            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr)
            )

            await process.wait()
            
            if process.returncode == 0:
                logger.info("Prompt tuning completed successfully")
                
                # Replace the existing template files with the newly generated prompts
                dest_dir = os.path.join(ROOT_DIR, "prompts")
                
                for filename in os.listdir(temp_output):
                    if filename.endswith(".txt"):
                        source_file = os.path.join(temp_output, filename)
                        dest_file = os.path.join(dest_dir, filename)
                        shutil.move(source_file, dest_file)
                        logger.info(f"Replaced {filename} in {dest_file}")
                
                return PromptTuneResponse(status="success", message="Prompt tuning completed successfully. Existing prompts have been replaced.")
            else:
                logger.error("Prompt tuning failed")
                return PromptTuneResponse(status="error", message="Prompt tuning failed. Check logs for details.")
        except Exception as e:
            logger.error(f"Prompt tuning failed: {str(e)}")
            return PromptTuneResponse(status="error", message=f"Prompt tuning failed: {str(e)}")

@app.post("/v1/prompt_tune")
async def prompt_tune(request: PromptTuneRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_prompt_tuning, request)
    return {"status": "started", "message": "Prompt tuning process has been started in the background"}

@app.get("/v1/prompt_tune_status")
async def prompt_tune_status():
    return {
        "status": "running" if prompt_tune_logs else "idle",
        "logs": list(prompt_tune_logs)
    }

class IndexingRequest(BaseModel):
    llm_model: str
    embed_model: str
    llm_api_base: str
    embed_api_base: str
    root: str
    verbose: bool = False
    nocache: bool = False
    resume: Optional[str] = None
    reporter: str = "rich"
    emit: List[str] = ["parquet"]
    custom_args: Optional[str] = None
    llm_params: Dict[str, Any] = Field(default_factory=dict)
    embed_params: Dict[str, Any] = Field(default_factory=dict)

# Global variable to store the latest indexing logs
indexing_logs = deque(maxlen=100)

async def run_indexing(request: IndexingRequest):
    cmd = ["python", "-m", "graphrag.index"]
    
    cmd.extend(["--root", request.root])
    
    if request.verbose:
        cmd.append("--verbose")
    
    if request.nocache:
        cmd.append("--nocache")
    
    if request.resume:
        cmd.extend(["--resume", request.resume])
    
    cmd.extend(["--reporter", request.reporter])
    cmd.extend(["--emit", ",".join(request.emit)])
    
    # Set environment variables for LLM and embedding models
    env: Dict[str, Any] = os.environ.copy()
    env["GRAPHRAG_LLM_MODEL"] = request.llm_model
    env["GRAPHRAG_EMBED_MODEL"] = request.embed_model
    env["GRAPHRAG_LLM_API_BASE"] = LLM_API_BASE
    env["GRAPHRAG_EMBED_API_BASE"] = EMBEDDINGS_API_BASE
        
    # Set environment variables for LLM parameters
    for key, value in request.llm_params.items():
        env[f"GRAPHRAG_LLM_{key.upper()}"] = str(value)
    
    # Set environment variables for embedding parameters
    for key, value in request.embed_params.items():
        env[f"GRAPHRAG_EMBED_{key.upper()}"] = str(value)
    
    # Add custom CLI arguments
    if request.custom_args:
        cmd.extend(request.custom_args.split())

    logger.info(f"Executing indexing command: {' '.join(cmd)}")
    logger.info(f"Environment variables: {env}")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        async def read_stream(stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line = line.decode().strip()
                indexing_logs.append(line)
                logger.info(line)

        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr)
        )

        await process.wait()
        
        if process.returncode == 0:
            logger.info("Indexing completed successfully")
            return {"status": "success", "message": "Indexing completed successfully"}
        else:
            logger.error("Indexing failed")
            return {"status": "error", "message": "Indexing failed. Check logs for details."}
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        return {"status": "error", "message": f"Indexing failed: {str(e)}"}


@app.post("/v1/index")
async def start_indexing(request: IndexingRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_indexing, request)
    return {"status": "started", "message": "Indexing process has been started in the background"}

@app.get("/v1/index_status")
async def indexing_status():
    return {
        "status": "running" if indexing_logs else "idle",
        "logs": list(indexing_logs)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the GraphRAG API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload mode")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
