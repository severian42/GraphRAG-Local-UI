import gradio as gr
import requests
import logging
import os
import json
import shutil
import glob
import queue
import lancedb
from datetime import datetime
from dotenv import load_dotenv, set_key
import yaml
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel

# Set up logging
log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv('indexing/.env')

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8012')
LLM_API_BASE = os.getenv('LLM_API_BASE', 'http://localhost:11434')
EMBEDDINGS_API_BASE = os.getenv('EMBEDDINGS_API_BASE', 'http://localhost:11434')
ROOT_DIR = os.getenv('ROOT_DIR', 'indexing')  

# Data models
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

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))
queue_handler = QueueHandler(log_queue)
logging.getLogger().addHandler(queue_handler)


def update_logs():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return "\n".join(logs)

##########SETTINGS################
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


#######FILE_MANAGEMENT##############
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

def list_output_folders():
    output_dir = os.path.join(ROOT_DIR, "output")
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    return sorted(folders, reverse=True)

def update_output_folder_list():
    folders = list_output_folders()
    return gr.update(choices=folders, value=folders[0] if folders else None)

def list_folder_contents(folder_name):
    folder_path = os.path.join(ROOT_DIR, "output", folder_name, "artifacts")
    contents = []
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                contents.append(f"[DIR] {item}")
            else:
                _, ext = os.path.splitext(item)
                contents.append(f"[{ext[1:].upper()}] {item}")
    return contents

def update_folder_content_list(folder_name):
    if isinstance(folder_name, list) and folder_name:
        folder_name = folder_name[0]  
    elif not folder_name:
        return gr.update(choices=[])  
    
    contents = list_folder_contents(folder_name)
    return gr.update(choices=contents)

def handle_content_selection(folder_name, selected_item):
    if isinstance(selected_item, list) and selected_item:
        selected_item = selected_item[0]  # Take the first item if it's a list
    
    if isinstance(selected_item, str) and selected_item.startswith("[DIR]"):
        dir_name = selected_item[6:]  # Remove "[DIR] " prefix
        sub_contents = list_folder_contents(os.path.join(ROOT_DIR, "output", folder_name, dir_name))
        return gr.update(choices=sub_contents), "", ""
    elif isinstance(selected_item, str):
        file_name = selected_item.split("] ")[1] if "]" in selected_item else selected_item  # Remove file type prefix if present
        file_path = os.path.join(ROOT_DIR, "output", folder_name, "artifacts", file_name)
        file_size = os.path.getsize(file_path)
        file_type = os.path.splitext(file_name)[1]
        file_info = f"File: {file_name}\nSize: {file_size} bytes\nType: {file_type}"
        content = read_file_content(file_path)
        return gr.update(), file_info, content
    else:
        return gr.update(), "", ""

def initialize_selected_folder(folder_name):
    if not folder_name:
        return "Please select a folder first.", gr.update(choices=[])
    folder_path = os.path.join(ROOT_DIR, "output", folder_name, "artifacts")
    if not os.path.exists(folder_path):
        return f"Artifacts folder not found in '{folder_name}'.", gr.update(choices=[])
    contents = list_folder_contents(folder_path)
    return f"Folder '{folder_name}/artifacts' initialized with {len(contents)} items.", gr.update(choices=contents)

def upload_file(file):
    if file is not None:
        input_dir = os.path.join(ROOT_DIR, 'input')
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
    input_dir = os.path.join(ROOT_DIR, 'input')
    files = []
    if os.path.exists(input_dir):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
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
    db = lancedb.connect(f"{ROOT_DIR}/lancedb")
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

def find_latest_output_folder():
    root_dir =f"{ROOT_DIR}/output"
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


###########MODELS##################
def normalize_api_base(api_base: str) -> str:
    """Normalize the API base URL by removing trailing slashes and /v1 or /api suffixes."""
    api_base = api_base.rstrip('/')
    if api_base.endswith('/v1') or api_base.endswith('/api'):
        api_base = api_base[:-3]
    return api_base

def is_ollama_api(base_url: str) -> bool:
    """Check if the given base URL is for Ollama API."""
    try:
        response = requests.get(f"{normalize_api_base(base_url)}/api/tags")
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_ollama_models(base_url: str) -> List[str]:
    """Fetch available models from Ollama API."""
    try:
        response = requests.get(f"{normalize_api_base(base_url)}/api/tags")
        response.raise_for_status()
        models = response.json().get('models', [])
        return [model['name'] for model in models]
    except requests.RequestException as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
        return []

def get_openai_compatible_models(base_url: str) -> List[str]:
    """Fetch available models from OpenAI-compatible API."""
    try:
        response = requests.get(f"{normalize_api_base(base_url)}/v1/models")
        response.raise_for_status()
        models = response.json().get('data', [])
        return [model['id'] for model in models]
    except requests.RequestException as e:
        logger.error(f"Error fetching OpenAI-compatible models: {str(e)}")
        return []

def get_local_models(base_url: str) -> List[str]:
    """Get available models based on the API type."""
    if is_ollama_api(base_url):
        return get_ollama_models(base_url)
    else:
        return get_openai_compatible_models(base_url)

def get_model_params(base_url: str, model_name: str) -> dict:
    """Get model parameters for Ollama models."""
    if is_ollama_api(base_url):
        try:
            response = requests.post(f"{normalize_api_base(base_url)}/api/show", json={"name": model_name})
            response.raise_for_status()
            model_info = response.json()
            return model_info.get('parameters', {})
        except requests.RequestException as e:
            logger.error(f"Error fetching Ollama model parameters: {str(e)}")
    return {}








#########API###########
def start_indexing(request: IndexingRequest):
    url = f"{API_BASE_URL}/v1/index"
    
    try:
        response = requests.post(url, json=request.dict())
        response.raise_for_status()
        result = response.json()
        return result['message'], gr.update(interactive=False), gr.update(interactive=True)
    except requests.RequestException as e:
        logger.error(f"Error starting indexing: {str(e)}")
        return f"Error: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)
    
def check_indexing_status():
    url = f"{API_BASE_URL}/v1/index_status"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        return result['status'], "\n".join(result['logs'])
    except requests.RequestException as e:
        logger.error(f"Error checking indexing status: {str(e)}")
        return "Error", f"Failed to check indexing status: {str(e)}"

def start_prompt_tuning(request: PromptTuneRequest):
    url = f"{API_BASE_URL}/v1/prompt_tune"
    
    try:
        response = requests.post(url, json=request.dict())
        response.raise_for_status()
        result = response.json()
        return result['message'], gr.update(interactive=False)
    except requests.RequestException as e:
        logger.error(f"Error starting prompt tuning: {str(e)}")
        return f"Error: {str(e)}", gr.update(interactive=True)

def check_prompt_tuning_status():
    url = f"{API_BASE_URL}/v1/prompt_tune_status"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        return result['status'], "\n".join(result['logs'])
    except requests.RequestException as e:
        logger.error(f"Error checking prompt tuning status: {str(e)}")
        return "Error", f"Failed to check prompt tuning status: {str(e)}"

def update_model_params(model_name):
    params = get_model_params(model_name)
    return gr.update(value=json.dumps(params, indent=2))









###########################
css = """
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


def create_interface():
    settings = load_settings()
    llm_api_base = normalize_api_base(settings['api_base'])
    embeddings_api_base = normalize_api_base(settings['embeddings_api_base'])

    with gr.Blocks(theme=gr.themes.Base(), css=css) as demo:
        gr.Markdown("# GraphRAG Indexer")
        
        with gr.Tabs():
            with gr.TabItem("Indexing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Indexing Configuration")
                        
                        with gr.Row():
                            llm_name = gr.Dropdown(label="LLM Model", choices=[], value=settings['llm_model'], allow_custom_value=True)
                            refresh_llm_btn = gr.Button("🔄", size='sm', scale=0)
                        
                        with gr.Row():
                            embed_name = gr.Dropdown(label="Embedding Model", choices=[], value=settings['embedding_model'], allow_custom_value=True)
                            refresh_embed_btn = gr.Button("🔄", size='sm', scale=0)
                        
                        save_config_button = gr.Button("Save Configuration", variant="primary")
                        config_status = gr.Textbox(label="Configuration Status", lines=2)
                        
                        with gr.Row():
                                with gr.Column(scale=1):
                                    root_dir = gr.Textbox(label="Root Directory (Edit in .env file)", value=f"{ROOT_DIR}")      
                        with gr.Group():                                                         
                            verbose = gr.Checkbox(label="Verbose", interactive=True, value=True)
                            nocache = gr.Checkbox(label="No Cache", interactive=True, value=True)
                        
                        with gr.Accordion("Advanced Options", open=True):
                            resume = gr.Textbox(label="Resume Timestamp (optional)")
                            reporter = gr.Dropdown(
                                label="Reporter",
                                choices=["rich", "print", "none"],
                                value="rich",
                                interactive=True
                            )
                            emit_formats = gr.CheckboxGroup(
                                label="Emit Formats",
                                choices=["json", "csv", "parquet"],
                                value=["parquet"],
                                interactive=True
                            )
                            custom_args = gr.Textbox(label="Custom CLI Arguments", placeholder="--arg1 value1 --arg2 value2")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("## Indexing Output")
                        index_output = gr.Textbox(label="Output", lines=10)
                        index_status = gr.Textbox(label="Status", lines=2)
                        
                        run_index_button = gr.Button("Run Indexing", variant="primary")
                        check_status_button = gr.Button("Check Indexing Status")


            with gr.TabItem("Prompt Tuning"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Prompt Tuning Configuration")
                        
                        pt_root = gr.Textbox(label="Root Directory", value=f"{ROOT_DIR}", interactive=True)
                        pt_domain = gr.Textbox(label="Domain (optional)")
                        pt_method = gr.Dropdown(
                            label="Method",
                            choices=["random", "top", "all"],
                            value="random",
                            interactive=True
                        )
                        pt_limit = gr.Number(label="Limit", value=15, precision=0, interactive=True)
                        pt_language = gr.Textbox(label="Language (optional)")
                        pt_max_tokens = gr.Number(label="Max Tokens", value=2000, precision=0, interactive=True)
                        pt_chunk_size = gr.Number(label="Chunk Size", value=200, precision=0, interactive=True)
                        pt_no_entity_types = gr.Checkbox(label="No Entity Types", value=False)
                        pt_output_dir = gr.Textbox(label="Output Directory", value=f"{ROOT_DIR}/prompts", interactive=True)
                        save_pt_config_button = gr.Button("Save Prompt Tuning Configuration", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("## Prompt Tuning Output")
                        pt_output = gr.Textbox(label="Output", lines=10)
                        pt_status = gr.Textbox(label="Status", lines=10)
                        
                        run_pt_button = gr.Button("Run Prompt Tuning", variant="primary")
                        check_pt_status_button = gr.Button("Check Prompt Tuning Status")

            with gr.TabItem("Data Management"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("File Upload", open=True):
                            file_upload = gr.File(label="Upload File", file_types=[".txt", ".csv", ".parquet"])
                            upload_btn = gr.Button("Upload File", variant="primary")
                            upload_output = gr.Textbox(label="Upload Status", visible=True)
                        
                        with gr.Accordion("File Management", open=True):
                            file_list = gr.Dropdown(label="Select File", choices=[], interactive=True)
                            refresh_btn = gr.Button("Refresh File List", variant="secondary")
                            
                            file_content = gr.TextArea(label="File Content", lines=10)
                            
                            with gr.Row():
                                delete_btn = gr.Button("Delete Selected File", variant="stop")
                                save_btn = gr.Button("Save Changes", variant="primary")
                            
                            operation_status = gr.Textbox(label="Operation Status", visible=True)
                    
                    with gr.Column(scale=1):
                        with gr.Accordion("Output Folders", open=True):
                            output_folder_list = gr.Dropdown(label="Select Output Folder", choices=[], interactive=True)
                            refresh_output_btn = gr.Button("Refresh Output Folders", variant="secondary")
                            folder_content_list = gr.Dropdown(label="Folder Contents", choices=[], interactive=True, multiselect=False)
                            
                            file_info = gr.Textbox(label="File Info", lines=3)
                            output_content = gr.TextArea(label="File Content", lines=10)

                        

        # Event handlers
        def refresh_llm_models():
            models = get_local_models(llm_api_base)
            return gr.update(choices=models)

        def refresh_embed_models():
            models = get_local_models(embeddings_api_base)
            return gr.update(choices=models)

        refresh_llm_btn.click(
            refresh_llm_models,
            outputs=[llm_name]
        )

        refresh_embed_btn.click(
            refresh_embed_models,
            outputs=[embed_name]
        )

        # Initialize model lists on page load
        demo.load(refresh_llm_models, outputs=[llm_name])
        demo.load(refresh_embed_models, outputs=[embed_name])

        def create_indexing_request():
            return IndexingRequest(
                llm_model=llm_name.value,
                embed_model=embed_name.value,
                llm_api_base=llm_api_base,
                embed_api_base=embeddings_api_base,
                root=root_dir.value,
                verbose=verbose.value,
                nocache=nocache.value,
                resume=resume.value if resume.value else None,
                reporter=reporter.value,
                emit=[fmt for fmt in emit_formats.value],
                custom_args=custom_args.value if custom_args.value else None
            )

        run_index_button.click(
            lambda: start_indexing(create_indexing_request()),
            outputs=[index_output, run_index_button, check_status_button]
        )

        check_status_button.click(
            check_indexing_status,
            outputs=[index_status, index_output]
        )

        def create_prompt_tune_request():
            return PromptTuneRequest(
                root=pt_root.value,
                domain=pt_domain.value if pt_domain.value else None,
                method=pt_method.value,
                limit=int(pt_limit.value),
                language=pt_language.value if pt_language.value else None,
                max_tokens=int(pt_max_tokens.value),
                chunk_size=int(pt_chunk_size.value),
                no_entity_types=pt_no_entity_types.value,
                output=pt_output_dir.value
            )

        def update_pt_output(request):
            result, button_update = start_prompt_tuning(request)
            return result, button_update, gr.update(value=f"Request: {request.dict()}")

        run_pt_button.click(
            lambda: update_pt_output(create_prompt_tune_request()),
            outputs=[pt_output, run_pt_button, pt_status]
        )

        check_pt_status_button.click(
            check_prompt_tuning_status,
            outputs=[pt_status, pt_output]
        )

        # Add event handlers for real-time updates
        pt_root.change(lambda x: gr.update(value=f"Root Directory changed to: {x}"), inputs=[pt_root], outputs=[pt_status])
        pt_limit.change(lambda x: gr.update(value=f"Limit changed to: {x}"), inputs=[pt_limit], outputs=[pt_status])
        pt_max_tokens.change(lambda x: gr.update(value=f"Max Tokens changed to: {x}"), inputs=[pt_max_tokens], outputs=[pt_status])
        pt_chunk_size.change(lambda x: gr.update(value=f"Chunk Size changed to: {x}"), inputs=[pt_chunk_size], outputs=[pt_status])
        pt_output_dir.change(lambda x: gr.update(value=f"Output Directory changed to: {x}"), inputs=[pt_output_dir], outputs=[pt_status])

        # Event handlers for Data Management
        upload_btn.click(
            upload_file,
            inputs=[file_upload],
            outputs=[upload_output, file_list, operation_status]
        )

        refresh_btn.click(
            update_file_list,
            outputs=[file_list]
        )

        refresh_output_btn.click(
            update_output_folder_list,
            outputs=[output_folder_list]
        )

        file_list.change(
            update_file_content,
            inputs=[file_list],
            outputs=[file_content]
        )

        delete_btn.click(
            delete_file,
            inputs=[file_list],
            outputs=[operation_status, file_list, operation_status]
        )

        save_btn.click(
            save_file_content,
            inputs=[file_list, file_content],
            outputs=[operation_status, operation_status]
        )

        output_folder_list.change(
            update_folder_content_list,
            inputs=[output_folder_list],
            outputs=[folder_content_list]
        )

        folder_content_list.change(
            handle_content_selection,
            inputs=[output_folder_list, folder_content_list],
            outputs=[folder_content_list, file_info, output_content]
        )

        # Event handler for saving configuration
        save_config_button.click(
            update_env_file,
            inputs=[llm_name, embed_name],
            outputs=[config_status]
        )

        # Event handler for saving prompt tuning configuration
        save_pt_config_button.click(
            save_prompt_tuning_config,
            inputs=[pt_root, pt_domain, pt_method, pt_limit, pt_language, pt_max_tokens, pt_chunk_size, pt_no_entity_types, pt_output_dir],
            outputs=[pt_status]
        )

        # Initialize file list and output folder list
        demo.load(update_file_list, outputs=[file_list])
        demo.load(update_output_folder_list, outputs=[output_folder_list])

    return demo

def update_env_file(llm_model, embed_model):
    env_path = os.path.join(ROOT_DIR, '.env')
    
    set_key(env_path, 'LLM_MODEL', llm_model)
    set_key(env_path, 'EMBEDDINGS_MODEL', embed_model)
    
    # Reload the environment variables
    load_dotenv(env_path, override=True)
    
    return f"Environment updated: LLM_MODEL={llm_model}, EMBEDDINGS_MODEL={embed_model}"

def save_prompt_tuning_config(root, domain, method, limit, language, max_tokens, chunk_size, no_entity_types, output_dir):
    config = {
        'prompt_tuning': {
            'root': root,
            'domain': domain,
            'method': method,
            'limit': limit,
            'language': language,
            'max_tokens': max_tokens,
            'chunk_size': chunk_size,
            'no_entity_types': no_entity_types,
            'output': output_dir
        }
    }
    
    config_path = os.path.join(ROOT_DIR, 'prompt_tuning_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return f"Prompt Tuning configuration saved to {config_path}"

demo = create_interface()

if __name__ == "__main__":
    demo.launch(server_port=7861)
