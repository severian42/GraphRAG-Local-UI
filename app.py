import gradio as gr
from gradio.helpers import Progress
import asyncio
import subprocess
import yaml
import os
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import lancedb
import random
import io
import shutil
import logging
import queue
import threading
import time
import re
import glob
from datetime import datetime
import json
import requests
import aiohttp
from openai import OpenAI
from openai import AsyncOpenAI
import pyarrow.parquet as pq
import pandas as pd
import sys
import colorsys
from dotenv import load_dotenv, set_key
import argparse
import socket
import tiktoken
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.llm.openai import create_openai_chat_llm
from graphrag.llm.openai.factories import create_openai_embedding_llm
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.llm.openai.openai_configuration import OpenAIConfiguration
from graphrag.llm.openai.openai_embeddings_llm import OpenAIEmbeddingsLLM
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore



# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gradio_client.documentation")


load_dotenv('ragtest/.env')

# Set default values for API-related environment variables
os.environ.setdefault("LLM_API_BASE", os.getenv("LLM_API_BASE"))
os.environ.setdefault("LLM_API_KEY", os.getenv("LLM_API_KEY"))
os.environ.setdefault("LLM_MODEL", os.getenv("LLM_MODEL"))
os.environ.setdefault("EMBEDDINGS_API_BASE", os.getenv("EMBEDDINGS_API_BASE"))
os.environ.setdefault("EMBEDDINGS_API_KEY", os.getenv("EMBEDDINGS_API_KEY"))
os.environ.setdefault("EMBEDDINGS_MODEL", os.getenv("EMBEDDINGS_MODEL"))

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


# Set up logging
log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


llm = None
text_embedder = None

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))
queue_handler = QueueHandler(log_queue)
logging.getLogger().addHandler(queue_handler)

class OllamaLLM:
    def __init__(self, api_base, model, max_retries=20):
        self.api_base = api_base
        self.model = model
        self.max_retries = max_retries

    async def __call__(self, prompt, **kwargs):
        endpoint = f"{self.api_base}/v1/completions"
        data = {
            "model": self.model,
            "prompt": prompt,
            **kwargs
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['response']
                else:
                    raise Exception(f"Error generating response: {await response.text()}")

class VectorStoreWrapper:
    def __init__(self, embedding_function, connection_string, table_name, collection_name):
        self.embedding_function = embedding_function
        self.connection_string = connection_string
        self.table_name = table_name
        self.collection_name = collection_name
        self._vector_store = None

    def get_vector_store(self):
        if self._vector_store is None:
            try:
                from graphrag.vector_stores.lancedb import LanceDBVectorStore
                self._vector_store = LanceDBVectorStore(
                    embedding_function=self.embedding_function,
                    connection_string=self.connection_string,
                    table_name=self.table_name,
                    collection_name=self.collection_name
                )
                logging.info("Vector store initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing vector store: {str(e)}")
                self._vector_store = None
        return self._vector_store

    def __getstate__(self):
        # This method is called when pickling the object
        state = self.__dict__.copy()
        # Don't pickle _vector_store
        state['_vector_store'] = None
        return state

    def __setstate__(self, state):
        # This method is called when unpickling the object
        self.__dict__.update(state)
        # _vector_store will be None, but it will be recreated when needed    

def create_vector_store_wrapper(text_embedder):
    return VectorStoreWrapper(
        embedding_function=text_embedder,
        connection_string="lancedb",
        table_name="documents",
        collection_name="graphrag_collection"
    )


def initialize_models():
    global llm, text_embedder
    
    llm_api_base = os.getenv("LLM_API_BASE")
    llm_api_key = os.getenv("LLM_API_KEY")
    embeddings_api_base = os.getenv("EMBEDDINGS_API_BASE")
    embeddings_api_key = os.getenv("EMBEDDINGS_API_KEY")
    
    llm_service_type = os.getenv("LLM_SERVICE_TYPE")
    embeddings_service_type = os.getenv("EMBEDDINGS_SERVICE_TYPE")
    
    llm_model = os.getenv("LLM_MODEL")
    embeddings_model = os.getenv("EMBEDDINGS_MODEL")
    
    logging.info("Fetching models...")
    models = fetch_models(llm_api_base, llm_api_key, llm_service_type)
    
    # Use the same models list for both LLM and embeddings
    llm_models = models
    embeddings_models = models
    
    # Initialize LLM
    if llm_service_type.lower() == "openai_chat":
        llm = ChatOpenAI(
            api_key=llm_api_key,
            api_base=f"{llm_api_base}/v1",
            model=llm_model,
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )
    elif llm_service_type.lower() == "ollama":
        llm = OllamaLLM(
            api_base=llm_api_base,
            model=llm_model,
            max_retries=20,
        )
    else:
        raise ValueError(f"Unsupported LLM service type: {llm_service_type}")

    # Initialize OpenAI client for embeddings
    openai_client = OpenAI(
        api_key=embeddings_api_key,
        base_url=f"{embeddings_api_base}/v1"
    )

    # Initialize text embedder using OpenAIEmbeddingsLLM
    text_embedder = OpenAIEmbeddingsLLM(
        client=openai_client,
        configuration={
            "model": embeddings_model,
            "api_type": "open_ai",
            "api_base": embeddings_api_base,
            "api_key": embeddings_api_key,
            "provider": embeddings_service_type.lower()
        }
    )
    
    return llm_models, embeddings_models, llm_service_type, embeddings_service_type, llm_api_base, embeddings_api_base, text_embedder

def find_latest_output_folder():
    root_dir = "./ragtest/output"
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    if not folders:
        raise ValueError("No output folders found")
    
    # Sort folders by creation time, most recent first
    sorted_folders = sorted(folders, key=lambda x: os.path.getctime(os.path.join(root_dir, x)), reverse=True)
    
    latest_folder = None
    timestamp = None
    
    for folder in sorted_folders:
        try:
            # Try to parse the folder name as a timestamp
            timestamp = datetime.strptime(folder, "%Y%m%d-%H%M%S")
            latest_folder = folder
            break
        except ValueError:
            # If the folder name is not a valid timestamp, skip it
            continue
    
    if latest_folder is None:
        raise ValueError("No valid timestamp folders found")
    
    latest_path = os.path.join(root_dir, latest_folder)
    artifacts_path = os.path.join(latest_path, "artifacts")
    
    if not os.path.exists(artifacts_path):
        raise ValueError(f"Artifacts folder not found in {latest_path}")
    
    return latest_path, latest_folder

def initialize_data():
    global entity_df, relationship_df, text_unit_df, report_df, covariate_df
    
    tables = {
        "entity_df": "create_final_nodes",
        "relationship_df": "create_final_edges",
        "text_unit_df": "create_final_text_units",
        "report_df": "create_final_reports",
        "covariate_df": "create_final_covariates"
    }
    
    timestamp = None  # Initialize timestamp to None
    
    try:
        latest_output_folder, timestamp = find_latest_output_folder()
        artifacts_folder = os.path.join(latest_output_folder, "artifacts")
        
        for df_name, file_prefix in tables.items():
            file_pattern = os.path.join(artifacts_folder, f"{file_prefix}*.parquet")
            matching_files = glob.glob(file_pattern)
            
            if matching_files:
                latest_file = max(matching_files, key=os.path.getctime)
                df = pd.read_parquet(latest_file)
                globals()[df_name] = df
                logging.info(f"Successfully loaded {df_name} from {latest_file}")
            else:
                logging.warning(f"No matching file found for {df_name} in {artifacts_folder}. Initializing as an empty DataFrame.")
                globals()[df_name] = pd.DataFrame()
    
    except Exception as e:
        logging.error(f"Error initializing data: {str(e)}")
        for df_name in tables.keys():
            globals()[df_name] = pd.DataFrame()

    return timestamp

# Call initialize_data and store the timestamp
current_timestamp = initialize_data()


def find_available_port(start_port, max_attempts=100):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise IOError("No free ports found")

def start_api_server(port):
    subprocess.Popen([sys.executable, "api_server.py", "--port", str(port)])

def wait_for_api_server(port):
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{port}")
            if response.status_code == 200:
                print(f"API server is up and running on port {port}")
                return
            else:
                print(f"Unexpected response from API server: {response.status_code}")
        except requests.ConnectionError:
            time.sleep(1)
    print("Failed to connect to API server")

def load_settings():
    try:
        with open("ragtest/settings.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def update_setting(key, value):
    settings = load_settings()
    try:
        settings[key] = json.loads(value)
    except json.JSONDecodeError:
        settings[key] = value
    
    try:
        with open("ragtest/settings.yaml", "w") as f:
            yaml.dump(settings, f, default_flow_style=False)
        return f"Setting '{key}' updated successfully"
    except Exception as e:
        return f"Error updating setting '{key}': {str(e)}"

def create_setting_component(key, value):
    with gr.Accordion(key, open=False):
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, indent=2)
            lines = value_str.count('\n') + 1
        else:
            value_str = str(value)
            lines = 1
        
        text_area = gr.TextArea(value=value_str, label="Value", lines=lines, max_lines=20)
        update_btn = gr.Button("Update", variant="primary")
        status = gr.Textbox(label="Status", visible=False)
        
        update_btn.click(
            fn=update_setting,
            inputs=[gr.Textbox(value=key, visible=False), text_area],
            outputs=[status]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[status]
        )



def get_openai_client():
    return OpenAI(
        base_url=os.getenv("LLM_API_BASE"),
        api_key=os.getenv("LLM_API_KEY"),
        llm_model = os.getenv("LLM_MODEL")
    )

def chat_with_openai(messages, model, temperature, max_tokens, api_base):
    try:
        logging.info(f"Attempting to use model: {model}")
        client = OpenAI(base_url=api_base, api_key=os.getenv("LLM_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in chat_with_openai: {str(e)}")
        logging.error(f"Attempted with model: {model}, api_base: {api_base}")
        return f"Error: {str(e)}"

def chat_with_llm(query, history, system_message, temperature, max_tokens, model, api_base):
    try:
        messages = [{"role": "system", "content": system_message}]
        for item in history:
            if isinstance(item, tuple) and len(item) == 2:
                human, ai = item
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": ai})
        messages.append({"role": "user", "content": query})

        logging.info(f"Sending chat request to {api_base} with model {model}")
        client = OpenAI(base_url=api_base, api_key=os.getenv("LLM_API_KEY", "dummy-key"))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in chat_with_llm: {str(e)}")
        logging.error(f"Attempted with model: {model}, api_base: {api_base}")
        raise RuntimeError(f"Chat request failed: {str(e)}")

def run_graphrag_query(cli_args):
    try:
        command = ' '.join(cli_args)
        logging.info(f"Executing command: {command}")
        result = subprocess.run(cli_args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running GraphRAG query: {e}")
        logging.error(f"Command output (stdout): {e.stdout}")
        logging.error(f"Command output (stderr): {e.stderr}")
        raise RuntimeError(f"GraphRAG query failed: {e.stderr}")

def send_message(query_type, query, history, system_message, temperature, max_tokens, preset, community_level, response_type, custom_cli_args, selected_folder):
    try:
        if query_type in ["global", "local"]:
            cli_args = construct_cli_args(query_type, preset, community_level, response_type, custom_cli_args, query, selected_folder)
            logging.info(f"Executing {query_type} search with command: {' '.join(cli_args)}")
            result = run_graphrag_query(cli_args)
            logging.info(f"Query result: {result}")
        else:  # Direct chat
            llm_model = os.getenv("LLM_MODEL")
            llm_api_base = os.getenv("LLM_API_BASE")
            logging.info(f"Executing direct chat with model: {llm_model}, API base: {llm_api_base}")
            
            try:
                result = chat_with_llm(query, history, system_message, temperature, max_tokens, llm_model, llm_api_base)
                logging.info(f"Direct chat result: {result[:100]}...")  # Log first 100 chars of result
            except Exception as chat_error:
                logging.error(f"Error in chat_with_llm: {str(chat_error)}")
                raise RuntimeError(f"Direct chat failed: {str(chat_error)}")
        
        history.append((query, result))
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        logging.exception("Exception details:")
        history.append((query, error_message))
    
    return history, gr.update(value=""), update_logs()

def construct_cli_args(query_type, preset, community_level, response_type, custom_cli_args, query, selected_folder):
    if not selected_folder:
        raise ValueError("No folder selected. Please select an output folder before querying.")

    artifacts_folder = os.path.join("./ragtest/output", selected_folder, "artifacts")
    if not os.path.exists(artifacts_folder):
        raise ValueError(f"Artifacts folder not found in {artifacts_folder}")

    base_args = [
        "python", "-m", "graphrag.query",
        "--data", artifacts_folder,
        "--method", query_type,
    ]

    # Apply preset configurations
    if preset.startswith("Default"):
        base_args.extend(["--community_level", "2", "--response_type", "Multiple Paragraphs"])
    elif preset.startswith("Detailed"):
        base_args.extend(["--community_level", "4", "--response_type", "Multi-Page Report"])
    elif preset.startswith("Quick"):
        base_args.extend(["--community_level", "1", "--response_type", "Single Paragraph"])
    elif preset.startswith("Bullet"):
        base_args.extend(["--community_level", "2", "--response_type", "List of 3-7 Points"])
    elif preset.startswith("Comprehensive"):
        base_args.extend(["--community_level", "5", "--response_type", "Multi-Page Report"])
    elif preset.startswith("High-Level"):
        base_args.extend(["--community_level", "1", "--response_type", "Single Page"])
    elif preset.startswith("Focused"):
        base_args.extend(["--community_level", "3", "--response_type", "Multiple Paragraphs"])
    elif preset == "Custom Query":
        base_args.extend([
            "--community_level", str(community_level),
            "--response_type", f'"{response_type}"',
        ])
        if custom_cli_args:
            base_args.extend(custom_cli_args.split())

    # Add the query at the end
    base_args.append(query)
    
    return base_args






def upload_file(file):
    if file is not None:
        input_dir = os.path.join("ragtest", "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # Get the original filename from the uploaded file
        original_filename = file.name
        
        # Create the destination path
        destination_path = os.path.join(input_dir, os.path.basename(original_filename))
        
        # Move the uploaded file to the destination path
        shutil.move(file.name, destination_path)
        
        logging.info(f"File uploaded and moved to: {destination_path}")
        status = f"File uploaded: {os.path.basename(original_filename)}"
    else:
        status = "No file uploaded"

    # Get the updated file list
    updated_file_list = [f["path"] for f in list_input_files()]
    
    return status, gr.update(choices=updated_file_list), update_logs()

def list_input_files():
    input_dir = os.path.join("ragtest", "input")
    files = []
    if os.path.exists(input_dir):
        files = os.listdir(input_dir)
    return [{"name": f, "path": os.path.join(input_dir, f)} for f in files]

def delete_file(file_path):
    try:
        os.remove(file_path)
        logging.info(f"File deleted: {file_path}")
        status = f"File deleted: {os.path.basename(file_path)}"
    except Exception as e:
        logging.error(f"Error deleting file: {str(e)}")
        status = f"Error deleting file: {str(e)}"

    # Get the updated file list
    updated_file_list = [f["path"] for f in list_input_files()]
    
    return status, gr.update(choices=updated_file_list), update_logs()

def read_file_content(file_path):
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            
            # Get basic information about the DataFrame
            info = f"Parquet File: {os.path.basename(file_path)}\n"
            info += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            info += "Column Names:\n" + "\n".join(df.columns) + "\n\n"
            
            # Display first few rows
            info += "First 5 rows:\n"
            info += df.head().to_string() + "\n\n"
            
            # Display basic statistics
            info += "Basic Statistics:\n"
            info += df.describe().to_string()
            
            return info
        else:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
        return content
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return f"Error reading file: {str(e)}"

def save_file_content(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        logging.info(f"File saved: {file_path}")
        status = f"File saved: {os.path.basename(file_path)}"
    except Exception as e:
        logging.error(f"Error saving file: {str(e)}")
        status = f"Error saving file: {str(e)}"
    return status, update_logs()

def manage_data():
    db = lancedb.connect("./ragtest/lancedb")
    tables = db.table_names()
    table_info = ""
    if tables:
        table = db[tables[0]]
        table_info = f"Table: {tables[0]}\nSchema: {table.schema}"
    
    input_files = list_input_files()
    
    return {
        "database_info": f"Tables: {', '.join(tables)}\n\n{table_info}",
        "input_files": input_files
    }


def find_latest_graph_file(root_dir):
    pattern = os.path.join(root_dir, "output", "*", "artifacts", "*.graphml")
    graph_files = glob.glob(pattern)
    if not graph_files:
        # If no files found, try excluding .DS_Store
        output_dir = os.path.join(root_dir, "output")
        run_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d != ".DS_Store"]
        if run_dirs:
            latest_run = max(run_dirs)
            pattern = os.path.join(root_dir, "output", latest_run, "artifacts", "*.graphml")
            graph_files = glob.glob(pattern)
    
    if not graph_files:
        return None
    
    # Sort files by modification time, most recent first
    latest_file = max(graph_files, key=os.path.getmtime)
    return latest_file

def update_visualization(folder_name, file_name, layout_type, node_size, edge_width, node_color_attribute, color_scheme, show_labels, label_size):
    root_dir = "./ragtest"
    if not folder_name or not file_name:
        return None, "Please select a folder and a GraphML file."
    file_name = file_name.split("] ")[1] if "]" in file_name else file_name  # Remove file type prefix
    graph_path = os.path.join(root_dir, "output", folder_name, "artifacts", file_name)
    if not graph_path.endswith('.graphml'):
        return None, "Please select a GraphML file for visualization."
    try:
        # Load the GraphML file
        graph = nx.read_graphml(graph_path)

        # Create layout based on user selection
        if layout_type == "3D Spring":
            pos = nx.spring_layout(graph, dim=3, seed=42, k=0.5)
        elif layout_type == "2D Spring":
            pos = nx.spring_layout(graph, dim=2, seed=42, k=0.5)
        else:  # Circular
            pos = nx.circular_layout(graph)

        # Extract node positions
        if layout_type == "3D Spring":
            x_nodes = [pos[node][0] for node in graph.nodes()]
            y_nodes = [pos[node][1] for node in graph.nodes()]
            z_nodes = [pos[node][2] for node in graph.nodes()]
        else:
            x_nodes = [pos[node][0] for node in graph.nodes()]
            y_nodes = [pos[node][1] for node in graph.nodes()]
            z_nodes = [0] * len(graph.nodes())  # Set all z-coordinates to 0 for 2D layouts

        # Extract edge positions
        x_edges, y_edges, z_edges = [], [], []
        for edge in graph.edges():
            x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
            if layout_type == "3D Spring":
                z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])
            else:
                z_edges.extend([0, 0, None])

        # Generate node colors based on user selection
        if node_color_attribute == "Degree":
            node_colors = [graph.degree(node) for node in graph.nodes()]
        else:  # Random
            node_colors = [random.random() for _ in graph.nodes()]
        node_colors = np.array(node_colors)
        node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())

        # Create the trace for edges
        edge_trace = go.Scatter3d(
            x=x_edges, y=y_edges, z=z_edges,
            mode='lines',
            line=dict(color='lightgray', width=edge_width),
            hoverinfo='none'
        )

        # Create the trace for nodes
        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node_size,
                color=node_colors,
                colorscale=color_scheme,
                colorbar=dict(
                    title='Node Degree' if node_color_attribute == "Degree" else "Random Value",
                    thickness=10,
                    x=1.1,
                    tickvals=[0, 1],
                    ticktext=['Low', 'High']
                ),
                line=dict(width=1)
            ),
            text=[node for node in graph.nodes()],
            textposition="top center",
            textfont=dict(size=label_size, color='black'),
            hoverinfo='text'
        )

        # Create the plot
        fig = go.Figure(data=[edge_trace, node_trace])

        # Update layout for better visualization
        fig.update_layout(
            title=f'{layout_type} Graph Visualization: {os.path.basename(graph_path)}',
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title='')
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            annotations=[
                dict(
                    showarrow=False,
                    text=f"Interactive {layout_type} visualization of GraphML data",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=0
                )
            ],
            autosize=True
        )

        fig.update_layout(autosize=True)
        fig.update_layout(height=600)  # Set a fixed height
        config = {'responsive': True}
        return fig, f"Graph visualization generated successfully. Using file: {graph_path}", config
    except Exception as e:
        return None, f"Error visualizing graph: {str(e)}"

def update_logs():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return "\n".join(logs)



def update_model_choices(base_url, api_key, service_type="openai"):
    if service_type.lower() == "ollama":
        models = fetch_ollama_models(base_url)
    else:  # OpenAI Compatible
        models = fetch_openai_models(base_url, api_key)
    
    if not models:
        logging.warning(f"No models fetched for {service_type}.")

    # Get the current model from settings
    current_model = settings['llm'].get('model')
    
    # If the current model is not in the list, add it
    if current_model and current_model not in models:
        models.append(current_model)
    
    return gr.update(choices=models, value=current_model if current_model in models else (models[0] if models else None))

def fetch_ollama_models(base_url):
    try:
        # Remove '/v1' from the base_url if present
        base_url = base_url.rstrip('/v1')
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        logging.info(f"Raw Ollama API response: {response.text}")
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                models = [model['name'] for model in data['models']]
            else:
                models = [tag['name'] for tag in data]  # The response is a list of tag objects
            
            if not models:
                logging.warning("No models found in Ollama API response")
                return ["No models available"]
            
            logging.info(f"Successfully fetched Ollama models: {models}")
            return models
        else:
            logging.error(f"Error fetching Ollama models. Status code: {response.status_code}, Response: {response.text}")
            return ["Error fetching models"]
    except requests.RequestException as e:
        logging.error(f"Exception while fetching Ollama models: {str(e)}")
        return ["Error: Connection failed"]
    except Exception as e:
        logging.error(f"Unexpected error in fetch_ollama_models: {str(e)}")
        return ["Error: Unexpected issue"]

def fetch_openai_models(base_url, api_key):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{base_url}/models", headers=headers)
        if response.status_code == 200:
            models = [model['id'] for model in response.json()['data']]
            return models
        else:
            print(f"Error fetching OpenAI models: {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching OpenAI models: {str(e)}")
        return []

def fetch_models(base_url, api_key, service_type):
    if service_type.lower() == "ollama":
        return fetch_ollama_models(base_url)
    else:  # OpenAI Compatible
        return fetch_openai_models(base_url, api_key)

def update_embeddings_model_choices(base_url, api_key, service_type):
    if service_type.lower() == "ollama":
        models = fetch_ollama_models(base_url)
    else:  # OpenAI Compatible
        models = fetch_openai_models(base_url, api_key)
    
    if not models:
        logging.warning(f"No models fetched for {service_type}.")

    # Get the current model from settings
    current_model = settings['embeddings']['llm'].get('model')
    
    # If the current model is not in the list, add it
    if current_model and current_model not in models:
        models.append(current_model)
    
    return gr.update(choices=models, value=current_model if current_model in models else (models[0] if models else None))

def update_llm_settings(llm_model, embeddings_model, context_window, system_message, temperature, max_tokens, 
                        llm_api_base, llm_api_key, 
                        embeddings_api_base, embeddings_api_key, embeddings_service_type):
    try:
        # Update settings.yaml
        settings = load_settings()
        settings['llm'].update({
            "type": "openai",  # Always set to "openai" since we removed the radio button
            "model": llm_model,
            "api_base": llm_api_base,
            "api_key": "${GRAPHRAG_API_KEY}",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": "openai_chat"  # Always set to "openai_chat"
        })
        settings['embeddings']['llm'].update({
            "type": "openai_embedding",  # Always use OpenAIEmbeddingsLLM
            "model": embeddings_model,
            "api_base": embeddings_api_base,
            "api_key": "${GRAPHRAG_API_KEY}",
            "provider": embeddings_service_type
        })
        
        with open("ragtest/settings.yaml", 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)
        
        # Update .env file
        update_env_file("LLM_API_BASE", llm_api_base)
        update_env_file("LLM_API_KEY", llm_api_key)
        update_env_file("LLM_MODEL", llm_model)
        update_env_file("EMBEDDINGS_API_BASE", embeddings_api_base)
        update_env_file("EMBEDDINGS_API_KEY", embeddings_api_key)
        update_env_file("EMBEDDINGS_MODEL", embeddings_model)
        update_env_file("CONTEXT_WINDOW", str(context_window))
        update_env_file("SYSTEM_MESSAGE", system_message)
        update_env_file("TEMPERATURE", str(temperature))
        update_env_file("MAX_TOKENS", str(max_tokens))
        update_env_file("LLM_SERVICE_TYPE", "openai_chat")
        update_env_file("EMBEDDINGS_SERVICE_TYPE", embeddings_service_type)
        
        # Reload environment variables
        load_dotenv(override=True)
        
        return "LLM and embeddings settings updated successfully in both settings.yaml and .env files."
    except Exception as e:
        return f"Error updating LLM and embeddings settings: {str(e)}"

def update_env_file(key, value):
    env_path = 'ragtest/.env'
    with open(env_path, 'r') as file:
        lines = file.readlines()
    
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break
    
    if not updated:
        lines.append(f"{key}={value}\n")
    
    with open(env_path, 'w') as file:
        file.writelines(lines)

custom_css = """
html, body {
    margin: 0;
    padding: 0;
    height: 100vh;
    overflow: hidden;
}

.gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    max-width: 100vw !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: auto;
    display: flex;
    flex-direction: column;
}

#main-container {
    flex: 1;
    display: flex;
    overflow: hidden;
}

#left-column, #right-column {
    height: 100%;
    overflow-y: auto;
    padding: 10px;
}

#left-column {
    flex: 1;
}

#right-column {
    flex: 2;
    display: flex;
    flex-direction: column;
}

#chat-container {
    flex: 0 0 auto;  /* Don't allow this to grow */
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid var(--color-accent);
    border-radius: 8px;
    padding: 10px;
    overflow-y: auto;
}

#chatbot {
    overflow-y: hidden;
    height: 100%;
}

#chat-input-row {
    margin-top: 10px;
}

#visualization-plot {
    width: 100%;
    aspect-ratio: 1 / 1;
    max-height: 600px;  /* Adjust this value as needed */
}

#vis-controls-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
}

#vis-controls-row > * {
    flex: 1;
    margin: 0 5px;
}

#vis-status {
    margin-top: 10px;
}

/* Chat input styling */
#chat-input-row {
    display: flex;
    flex-direction: column;
}

#chat-input-row > div {
    width: 100% !important;
}

#chat-input-row input[type="text"] {
    width: 100% !important;
}

/* Adjust padding for all containers */
.gr-box, .gr-form, .gr-panel {
    padding: 10px !important;
}

/* Ensure all textboxes and textareas have full height */
.gr-textbox, .gr-textarea {
    height: auto !important;
    min-height: 100px !important;
}

/* Ensure all dropdowns have full width */
.gr-dropdown {
    width: 100% !important;
}

:root {
    --color-background: #2C3639;
    --color-foreground: #3F4E4F;
    --color-accent: #A27B5C;
    --color-text: #DCD7C9;
}

body, .gradio-container {
    background-color: var(--color-background);
    color: var(--color-text);
}

.gr-button {
    background-color: var(--color-accent);
    color: var(--color-text);
}

.gr-input, .gr-textarea, .gr-dropdown {
    background-color: var(--color-foreground);
    color: var(--color-text);
    border: 1px solid var(--color-accent);
}

.gr-panel {
    background-color: var(--color-foreground);
    border: 1px solid var(--color-accent);
}

.gr-box {
    border-radius: 8px;
    margin-bottom: 10px;
    background-color: var(--color-foreground);
}

.gr-padded {
    padding: 10px;
}

.gr-form {
    background-color: var(--color-foreground);
}

.gr-input-label, .gr-radio-label {
    color: var(--color-text);
}

.gr-checkbox-label {
    color: var(--color-text);
}

.gr-markdown {
    color: var(--color-text);
}

.gr-accordion {
    background-color: var(--color-foreground);
    border: 1px solid var(--color-accent);
}

.gr-accordion-header {
    background-color: var(--color-accent);
    color: var(--color-text);
}

#visualization-container {
    display: flex;
    flex-direction: column;
    border: 2px solid var(--color-accent);
    border-radius: 8px;
    margin-top: 20px;
    padding: 10px;
    background-color: var(--color-foreground);
    height: calc(100vh - 300px);  /* Adjust this value as needed */
}

#visualization-plot {
    width: 100%;
    height: 100%;
}

#vis-controls-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
}

#vis-controls-row > * {
    flex: 1;
    margin: 0 5px;
}

#vis-status {
    margin-top: 10px;
}

#log-container {
    background-color: var(--color-foreground);
    border: 1px solid var(--color-accent);
    border-radius: 8px;
    padding: 10px;
    margin-top: 20px;
    max-height: auto;
    overflow-y: auto;
}

.setting-accordion .label-wrap {
    cursor: pointer;
}

.setting-accordion .icon {
    transition: transform 0.3s ease;
}

.setting-accordion[open] .icon {
    transform: rotate(90deg);
}

.gr-form.gr-box {
    border: none !important;
    background: none !important;
}

.model-params {
    border-top: 1px solid var(--color-accent);
    margin-top: 10px;
    padding-top: 10px;
}
"""

def list_output_files(root_dir):
    output_dir = os.path.join(root_dir, "output")
    files = []
    for root, _, filenames in os.walk(output_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def update_file_list():
    files = list_input_files()
    return gr.update(choices=[f["path"] for f in files])

def update_file_content(file_path):
    if not file_path:
        return ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return f"Error reading file: {str(e)}"

def update_output_folder_list():
    root_dir = "./ragtest"
    folders = list_output_folders(root_dir)
    return gr.update(choices=folders, value=folders[0] if folders else None)

def update_folder_content_list(folder_name):
    root_dir = "./ragtest"
    if not folder_name:
        return gr.update(choices=[])
    contents = list_folder_contents(os.path.join(root_dir, "output", folder_name))
    return gr.update(choices=contents)

def handle_content_selection(folder_name, selected_item):
    root_dir = "./ragtest"
    if isinstance(selected_item, list) and selected_item:
        selected_item = selected_item[0]  # Take the first item if it's a list
    
    if isinstance(selected_item, str) and selected_item.startswith("[DIR]"):
        dir_name = selected_item[6:]  # Remove "[DIR] " prefix
        sub_contents = list_folder_contents(os.path.join(root_dir, "output", folder_name, dir_name))
        return gr.update(choices=sub_contents), "", ""
    elif isinstance(selected_item, str):
        file_name = selected_item.split("] ")[1] if "]" in selected_item else selected_item  # Remove file type prefix if present
        file_path = os.path.join(root_dir, "output", folder_name, "artifacts", file_name)
        file_size = os.path.getsize(file_path)
        file_type = os.path.splitext(file_name)[1]
        file_info = f"File: {file_name}\nSize: {file_size} bytes\nType: {file_type}"
        content = read_file_content(file_path)
        return gr.update(), file_info, content
    else:
        return gr.update(), "", ""

def initialize_selected_folder(folder_name):
    root_dir = "./ragtest"
    if not folder_name:
        return "Please select a folder first.", gr.update(choices=[])
    folder_path = os.path.join(root_dir, "output", folder_name, "artifacts")
    if not os.path.exists(folder_path):
        return f"Artifacts folder not found in '{folder_name}'.", gr.update(choices=[])
    contents = list_folder_contents(folder_path)
    return f"Folder '{folder_name}/artifacts' initialized with {len(contents)} items.", gr.update(choices=contents)

def list_output_folders(root_dir):
    output_dir = os.path.join(root_dir, "output")
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    return sorted(folders, reverse=True)

def list_folder_contents(folder_path):
    contents = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            contents.append(f"[DIR] {item}")
        else:
            _, ext = os.path.splitext(item)
            contents.append(f"[{ext[1:].upper()}] {item}")
    return contents


settings = load_settings()
default_model = settings['llm']['model']
cli_args = gr.State({})
stop_indexing = threading.Event()
indexing_thread = None

def start_indexing(*args):
    global indexing_thread, stop_indexing
    stop_indexing = threading.Event()  # Reset the stop_indexing event
    indexing_thread = threading.Thread(target=run_indexing, args=args)
    indexing_thread.start()
    return gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)

def stop_indexing_process():
    global indexing_thread
    logging.info("Stop indexing requested")
    stop_indexing.set()
    if indexing_thread and indexing_thread.is_alive():
        logging.info("Waiting for indexing thread to finish")
        indexing_thread.join(timeout=10)
        logging.info("Indexing thread finished" if not indexing_thread.is_alive() else "Indexing thread did not finish within timeout")
    indexing_thread = None  # Reset the thread
    return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)

def refresh_indexing():
    global indexing_thread, stop_indexing
    if indexing_thread and indexing_thread.is_alive():
        logging.info("Cannot refresh: Indexing is still running")
        return gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), "Cannot refresh: Indexing is still running"
    else:
        stop_indexing = threading.Event()  # Reset the stop_indexing event
        indexing_thread = None  # Reset the thread
        return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True), "Indexing process refreshed. You can start indexing again."



def run_indexing(root_dir, config_file, verbose, nocache, resume, reporter, emit_formats, progress=gr.Progress()):
    cmd = ["python", "-m", "graphrag.index", "--root", root_dir]
    if config_file:
        cmd.extend(["--config", config_file.name])
    if verbose:
        cmd.append("--verbose")
    if nocache:
        cmd.append("--nocache")
    if resume:
        cmd.extend(["--resume", resume])
    cmd.extend(["--reporter", reporter])
    cmd.extend(["--emit", ",".join(emit_formats)])
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    
    output = []
    progress_value = 0
    iterations_completed = 0
    
    while True:
        if stop_indexing.is_set():
            process.terminate()
            process.wait(timeout=5)
            if process.poll() is None:
                process.kill()
            return ("\n".join(output + ["Indexing stopped by user."]), 
                    "Indexing stopped.", 
                    100, 
                    gr.update(interactive=True), 
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    str(iterations_completed))

        try:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                line = line.strip()
                output.append(line)
                
                # Update progress based on specific output lines
                if "Processing file" in line:
                    progress_value += 1
                    iterations_completed += 1
                elif "Indexing completed" in line:
                    progress_value = 100
                elif "ERROR" in line:
                    # Highlight errors in the output
                    line = f"🚨 ERROR: {line}"
                
                # Use the actual CLI output for progress message
                progress_msg = line
                
                # Yield an update for every line of output
                yield ("\n".join(output), 
                       progress_msg, 
                       progress_value, 
                       gr.update(interactive=False), 
                       gr.update(interactive=True),
                       gr.update(interactive=False),
                       str(iterations_completed))
                time.sleep(0.1)  # Add a small delay to prevent overwhelming the UI
        except Exception as e:
            logging.error(f"Error during indexing: {str(e)}")
            return ("\n".join(output + [f"Error: {str(e)}"]), 
                    "Error occurred during indexing.", 
                    100, 
                    gr.update(interactive=True), 
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    str(iterations_completed))
    
    if process.returncode != 0 and not stop_indexing.is_set():
        final_output = "\n".join(output + [f"Error: Process exited with return code {process.returncode}"])
        final_progress = "Indexing failed. Check output for details."
    else:
        final_output = "\n".join(output)
        final_progress = "Indexing completed successfully!"
    
    return (final_output, 
            final_progress, 
            100, 
            gr.update(interactive=True), 
            gr.update(interactive=False),
            gr.update(interactive=True),
            str(iterations_completed))

global_vector_store_wrapper = None

def create_gradio_interface():
    global global_vector_store_wrapper
    llm_models, embeddings_models, llm_service_type, embeddings_service_type, llm_api_base, embeddings_api_base, text_embedder = initialize_models()
    settings = load_settings()
    
    global_vector_store_wrapper = create_vector_store_wrapper(text_embedder)

    log_output = gr.TextArea(label="Logs", elem_id="log-output", interactive=False, visible=False)

    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        gr.Markdown("# GraphRAG Local UI", elem_id="title")
        
        with gr.Row(elem_id="main-container"):
            with gr.Column(scale=1, elem_id="left-column"):
                with gr.Tabs():
                    with gr.TabItem("Data Management"):
                        with gr.Accordion("File Upload (.txt)", open=True):
                            file_upload = gr.File(label="Upload .txt File", file_types=[".txt"])
                            upload_btn = gr.Button("Upload File", variant="primary")
                            upload_output = gr.Textbox(label="Upload Status", visible=False)
                        
                        with gr.Accordion("File Management", open=True):
                            file_list = gr.Dropdown(label="Select File", choices=[], interactive=True)
                            refresh_btn = gr.Button("Refresh File List", variant="secondary")
                            
                            file_content = gr.TextArea(label="File Content", lines=10)
                            
                            with gr.Row():
                                delete_btn = gr.Button("Delete Selected File", variant="stop")
                                save_btn = gr.Button("Save Changes", variant="primary")
                            
                            operation_status = gr.Textbox(label="Operation Status", visible=False)
                        
                        

                    with gr.TabItem("Indexing"):
                        root_dir = gr.Textbox(label="Root Directory", value="./ragtest")
                        config_file = gr.File(label="Config File (optional)")
                        with gr.Row():
                            verbose = gr.Checkbox(label="Verbose", value=True)
                            nocache = gr.Checkbox(label="No Cache", value=False)
                        with gr.Row():
                            resume = gr.Textbox(label="Resume Timestamp (optional)")
                            reporter = gr.Dropdown(label="Reporter", choices=["rich", "print", "none"], value="rich")
                        with gr.Row():
                            emit_formats = gr.CheckboxGroup(label="Emit Formats", choices=["json", "csv", "parquet"], value=["parquet"])
                        with gr.Row():
                            run_index_button = gr.Button("Run Indexing")
                            stop_index_button = gr.Button("Stop Indexing", variant="stop")
                            refresh_index_button = gr.Button("Refresh Indexing", variant="secondary")
                        
                        index_output = gr.Textbox(label="Indexing Output", lines=20, max_lines=30)
                        index_progress = gr.Textbox(label="Indexing Progress", lines=3)
                        iterations_completed = gr.Textbox(label="Iterations Completed", value="0")
                        refresh_status = gr.Textbox(label="Refresh Status", visible=True)

                        run_index_button.click(
                            fn=start_indexing,
                            inputs=[root_dir, config_file, verbose, nocache, resume, reporter, emit_formats],
                            outputs=[run_index_button, stop_index_button, refresh_index_button]
                        ).then(
                            fn=run_indexing,
                            inputs=[root_dir, config_file, verbose, nocache, resume, reporter, emit_formats],
                            outputs=[index_output, index_progress, run_index_button, stop_index_button, refresh_index_button, iterations_completed]
                        )

                        stop_index_button.click(
                            fn=stop_indexing_process,
                            outputs=[run_index_button, stop_index_button, refresh_index_button]
                        )

                        refresh_index_button.click(
                            fn=refresh_indexing,
                            outputs=[run_index_button, stop_index_button, refresh_index_button, refresh_status]
                        )

                    with gr.TabItem("KG Chat/Outputs"):
                        output_folder_list = gr.Dropdown(label="Select Output Folder", choices=[], interactive=True)
                        refresh_folder_btn = gr.Button("Refresh Folder List", variant="secondary")
                        initialize_folder_btn = gr.Button("Initialize Selected Folder", variant="primary")
                        folder_content_list = gr.Dropdown(label="Select File or Directory", choices=[], interactive=True)
                        file_info = gr.Textbox(label="File Information", interactive=False)
                        output_content = gr.TextArea(label="File Content", lines=20, interactive=False)
                        initialization_status = gr.Textbox(label="Initialization Status")
                    
                    with gr.TabItem("LLM Settings"):
                        llm_base_url = gr.Textbox(label="LLM API Base URL", value=os.getenv("LLM_API_BASE"))
                        llm_api_key = gr.Textbox(label="LLM API Key", value=os.getenv("LLM_API_KEY"), type="password")
                        llm_service_type = gr.Radio(
                            label="LLM Service Type",
                            choices=["openai", "ollama"],
                            value="openai",
                            visible=False  # Hide this if you want to always use OpenAI
                        )

                        llm_model_dropdown = gr.Dropdown(
                            label="LLM Model", 
                            choices=[],  # Start with an empty list
                            value=settings['llm'].get('model'),
                            allow_custom_value=True
                        )
                        refresh_llm_models_btn = gr.Button("Refresh LLM Models", variant="secondary")
                        
                        embeddings_base_url = gr.Textbox(label="Embeddings API Base URL", value=os.getenv("EMBEDDINGS_API_BASE"))
                        embeddings_api_key = gr.Textbox(label="Embeddings API Key", value=os.getenv("EMBEDDINGS_API_KEY"), type="password")
                        embeddings_service_type = gr.Radio(
                            label="Embeddings Service Type",
                            choices=["openai", "ollama"],
                            value=settings['embeddings']['llm'].get('type', 'openai'),
                        )

                        embeddings_model_dropdown = gr.Dropdown(
                            label="Embeddings Model",
                            choices=[],
                            value=settings['embeddings']['llm'].get('model'),
                            allow_custom_value=True
                        )
                        refresh_embeddings_models_btn = gr.Button("Refresh Embedding Models", variant="secondary")
                        system_message = gr.Textbox(
                            lines=5,
                            label="System Message",
                            value=os.getenv("SYSTEM_MESSAGE", "You are a helpful AI assistant.")
                        )
                        context_window = gr.Slider(
                            label="Context Window",
                            minimum=512,
                            maximum=32768,
                            step=512,
                            value=int(os.getenv("CONTEXT_WINDOW", 4096))
                        )                        
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            value=float(settings['llm'].get('TEMPERATURE', 0.5))
                        )
                        max_tokens = gr.Slider(
                            label="Max Tokens",
                            minimum=1,
                            maximum=8192,
                            step=1,
                            value=int(settings['llm'].get('MAX_TOKENS', 1024))
                        )
                        update_settings_btn = gr.Button("Update LLM Settings", variant="primary")
                        llm_settings_status = gr.Textbox(label="Status", interactive=False)

                        llm_base_url.change(
                            fn=update_model_choices,
                            inputs=[llm_base_url, llm_api_key],
                            outputs=llm_model_dropdown
                        )

                        # Update Embeddings model choices when service type or base URL changes
                        embeddings_service_type.change(
                            fn=update_embeddings_model_choices,
                            inputs=[embeddings_base_url, embeddings_api_key, embeddings_service_type],
                            outputs=embeddings_model_dropdown
                        )

                        embeddings_base_url.change(
                            fn=update_embeddings_model_choices,
                            inputs=[embeddings_base_url, embeddings_api_key, embeddings_service_type],
                            outputs=embeddings_model_dropdown
                        )

                        update_settings_btn.click(
                            fn=update_llm_settings,
                            inputs=[
                                llm_model_dropdown,
                                embeddings_model_dropdown,
                                context_window,
                                system_message,
                                temperature,
                                max_tokens,
                                llm_base_url, 
                                llm_api_key,
                                embeddings_base_url,
                                embeddings_api_key,
                                embeddings_service_type
                            ],
                            outputs=[llm_settings_status]
                        )


                        refresh_llm_models_btn.click(
                            fn=update_model_choices,
                            inputs=[llm_base_url, llm_api_key, llm_service_type],
                            outputs=[llm_model_dropdown]
                        ).then(
                            fn=update_logs,
                            outputs=[log_output]
                        )


                        refresh_embeddings_models_btn.click(
                            fn=update_model_choices,
                            inputs=[embeddings_base_url, embeddings_api_key, embeddings_service_type],
                            outputs=[embeddings_model_dropdown]
                        ).then(
                            fn=update_logs,
                            outputs=[log_output]
                        )

                    with gr.TabItem("YAML Settings"):
                        settings = load_settings()
                        with gr.Group():
                            for key, value in settings.items():
                                if key != 'llm':
                                    create_setting_component(key, value)
                
                with gr.Group(elem_id="log-container"):
                    gr.Markdown("### Logs")
                    log_output = gr.TextArea(label="Logs", elem_id="log-output", interactive=False)

            with gr.Column(scale=2, elem_id="right-column"):
                with gr.Group(elem_id="chat-container"):
                    chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot")
                    with gr.Row(elem_id="chat-input-row"):
                        with gr.Column(scale=1):
                            query_input = gr.Textbox(
                                label="Input",
                                placeholder="Enter your query here...",
                                elem_id="query-input"
                            )
                            query_btn = gr.Button("Send Query", variant="primary")
                        
                    with gr.Accordion("Query Parameters", open=True):
                        query_type = gr.Radio(
                            ["global", "local", "direct"],
                            label="Query Type",
                            value="global",
                            info="Global: community-based search, Local: entity-based search, Direct: LLM chat"
                        )
                        preset_dropdown = gr.Dropdown(
                            label="Preset Query Options",
                            choices=[
                                "Default Global Search",
                                "Default Local Search",
                                "Detailed Global Analysis",
                                "Detailed Local Analysis",
                                "Quick Global Summary",
                                "Quick Local Summary",
                                "Global Bullet Points",
                                "Local Bullet Points",
                                "Comprehensive Global Report",
                                "Comprehensive Local Report",
                                "High-Level Global Overview",
                                "High-Level Local Overview",
                                "Focused Global Insight",
                                "Focused Local Insight",
                                "Custom Query"
                            ],
                            value="Default Global Search",
                            info="Select a preset or choose 'Custom Query' for manual configuration"
                        )
                        selected_folder = gr.Dropdown(
                            label="Select Output Folder",
                            choices=list_output_folders("./ragtest"),
                            value=None,
                            interactive=True
                        )
                        
                        with gr.Group(visible=False) as custom_options:
                            community_level = gr.Slider(
                                label="Community Level",
                                minimum=1,
                                maximum=10,
                                value=2,
                                step=1,
                                info="Higher values use reports on smaller communities"
                            )
                            response_type = gr.Dropdown(
                                label="Response Type",
                                choices=[
                                    "Multiple Paragraphs",
                                    "Single Paragraph",
                                    "Single Sentence",
                                    "List of 3-7 Points",
                                    "Single Page",
                                    "Multi-Page Report"
                                ],
                                value="Multiple Paragraphs",
                                info="Specify the desired format of the response"
                            )
                            custom_cli_args = gr.Textbox(
                                label="Custom CLI Arguments",
                                placeholder="--arg1 value1 --arg2 value2",
                                info="Additional CLI arguments for advanced users"
                            )

                    def update_custom_options(preset):
                        if preset == "Custom Query":
                            return gr.update(visible=True)
                        else:
                            return gr.update(visible=False)

                    preset_dropdown.change(fn=update_custom_options, inputs=[preset_dropdown], outputs=[custom_options])

                
                    

                    with gr.Group(elem_id="visualization-container"):
                        vis_output = gr.Plot(label="Graph Visualization", elem_id="visualization-plot")
                        with gr.Row(elem_id="vis-controls-row"):
                            vis_btn = gr.Button("Visualize Graph", variant="secondary")
                        
                        # Add new controls for customization
                        with gr.Accordion("Visualization Settings", open=False):
                            layout_type = gr.Dropdown(["3D Spring", "2D Spring", "Circular"], label="Layout Type", value="3D Spring")
                            node_size = gr.Slider(1, 20, 7, label="Node Size", step=1)
                            edge_width = gr.Slider(0.1, 5, 0.5, label="Edge Width", step=0.1)
                            node_color_attribute = gr.Dropdown(["Degree", "Random"], label="Node Color Attribute", value="Degree")
                            color_scheme = gr.Dropdown(["Viridis", "Plasma", "Inferno", "Magma", "Cividis"], label="Color Scheme", value="Viridis")
                            show_labels = gr.Checkbox(label="Show Node Labels", value=True)
                            label_size = gr.Slider(5, 20, 10, label="Label Size", step=1)
                        

        # Event handlers
        upload_btn.click(fn=upload_file, inputs=[file_upload], outputs=[upload_output, file_list, log_output])
        refresh_btn.click(fn=update_file_list, outputs=[file_list]).then(
            fn=update_logs,
            outputs=[log_output]
        )
        file_list.change(fn=update_file_content, inputs=[file_list], outputs=[file_content]).then(
            fn=update_logs,
            outputs=[log_output]
        )
        delete_btn.click(fn=delete_file, inputs=[file_list], outputs=[operation_status, file_list, log_output])
        save_btn.click(fn=save_file_content, inputs=[file_list, file_content], outputs=[operation_status, log_output])

        refresh_folder_btn.click(fn=update_output_folder_list, outputs=[output_folder_list]).then(
            fn=update_logs,
            outputs=[log_output]
        )
        output_folder_list.change(
            fn=update_folder_content_list,
            inputs=[output_folder_list],
            outputs=[folder_content_list]
        ).then(
            fn=update_logs,
            outputs=[log_output]
        )
        folder_content_list.change(
            fn=handle_content_selection,
            inputs=[output_folder_list, folder_content_list],
            outputs=[folder_content_list, file_info, output_content]
        ).then(
            fn=update_logs,
            outputs=[log_output]
        )
        initialize_folder_btn.click(
            fn=initialize_selected_folder,
            inputs=[output_folder_list],
            outputs=[initialization_status, folder_content_list]
        ).then(
            fn=update_logs,
            outputs=[log_output]
        )
        vis_btn.click(
            fn=update_visualization,
            inputs=[
                output_folder_list,
                folder_content_list,
                layout_type,
                node_size,
                edge_width,
                node_color_attribute,
                color_scheme,
                show_labels,
                label_size
            ],
            outputs=[vis_output]
        ).then(
            fn=update_logs,
            outputs=[log_output]
        )

        query_btn.click(
            fn=send_message,
            inputs=[
                query_type,
                query_input,
                chatbot,
                system_message,
                temperature,
                max_tokens,
                preset_dropdown,
                community_level,
                response_type,
                custom_cli_args,
                selected_folder
            ],
            outputs=[chatbot, query_input, log_output]
        )

        query_input.submit(
            fn=send_message,
            inputs=[
                query_type,
                query_input,
                chatbot,
                system_message,
                temperature,
                max_tokens,
                preset_dropdown,
                community_level,
                response_type,
                custom_cli_args,
                selected_folder
            ],
            outputs=[chatbot, query_input, log_output]
        )
        refresh_llm_models_btn.click(
            fn=update_model_choices,
            inputs=[llm_base_url, llm_api_key, llm_service_type],
            outputs=[llm_model_dropdown]
        ).then(
            fn=update_logs,
            outputs=[log_output]
        )

        # Update Embeddings model choices
        refresh_embeddings_models_btn.click(
            fn=update_model_choices,
            inputs=[embeddings_base_url, embeddings_api_key, embeddings_service_type],
            outputs=[embeddings_model_dropdown]
        ).then(
            fn=update_logs,
            outputs=[log_output]
        )
        
        # Add this JavaScript to enable Shift+Enter functionality
        demo.load(js="""
        function addShiftEnterListener() {
            const queryInput = document.getElementById('query-input');
            if (queryInput) {
                queryInput.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' && event.shiftKey) {
                        event.preventDefault();
                        const submitButton = queryInput.closest('.gradio-container').querySelector('button.primary');
                        if (submitButton) {
                            submitButton.click();
                        }
                    }
                });
            }
        }
        document.addEventListener('DOMContentLoaded', addShiftEnterListener);
        """)

    return demo.queue()

def main():
    api_port = 8088
    gradio_port = 7860

    print(f"Starting API server on port {api_port}")
    start_api_server(api_port)

    # Wait for the API server to start in a separate thread
    threading.Thread(target=wait_for_api_server, args=(api_port,)).start()

    # Create the Gradio app
    demo = create_gradio_interface()

    print(f"Starting Gradio app on port {gradio_port}")
    # Launch the Gradio app
    demo.launch(server_port=gradio_port, share=True)


demo = create_gradio_interface()
app = demo.app

if __name__ == "__main__":
    initialize_data()
    demo.launch(server_port=7860, share=True)
