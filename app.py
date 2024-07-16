import gradio as gr
import subprocess
import yaml
import os
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import lancedb
import io
import shutil
import logging
import queue
import threading
import time
import glob
from datetime import datetime
import json
import requests
from ollama import chat
import pyarrow.parquet as pq
import pandas as pd


# Set up logging
log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

queue_handler = QueueHandler(log_queue)
logging.getLogger().addHandler(queue_handler)

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

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def index_graph(root_dir, progress=gr.Progress()):
    command = f"python -m graphrag.index --root {root_dir}"
    logging.info(f"Running indexing command: {command}")
    
    # Create a queue to store the output
    output_queue = queue.Queue()
    
    def run_command_with_output():
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ''):
            output_queue.put(line)
        process.stdout.close()
        process.wait()
    
    # Start the command in a separate thread
    thread = threading.Thread(target=run_command_with_output)
    thread.start()
    
    # Initialize progress
    progress(0, desc="Starting indexing...")
    
    # Process the output and update progress
    full_output = []
    while thread.is_alive() or not output_queue.empty():
        try:
            line = output_queue.get_nowait()
            full_output.append(line)
            
            # Update progress based on the output
            if "Processing file" in line:
                progress((0.5, None), desc="Processing files...")
            elif "Indexing completed" in line:
                progress(1, desc="Indexing completed")
            
            yield "\n".join(full_output), update_logs()
        except queue.Empty:
            time.sleep(0.1)
    
    thread.join()
    logging.info("Indexing completed")
    return "\n".join(full_output), update_logs()

def run_query(root_dir, method, query, history):
    command = f"python -m graphrag.query --root {root_dir} --method {method} \"{query}\""
    result = run_command(command)
    return result

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
        return None
    
    # Sort files by modification time, most recent first
    latest_file = max(graph_files, key=os.path.getmtime)
    return latest_file

def update_visualization(root_dir, folder_name, file_name):
    if not folder_name or not file_name:
        return None, "Please select a folder and a GraphML file."
    file_name = file_name.split("] ")[1] if "]" in file_name else file_name  # Remove file type prefix
    graph_path = os.path.join(root_dir, "output", folder_name, "artifacts", file_name)
    if not graph_path.endswith('.graphml'):
        return None, "Please select a GraphML file for visualization."
    try:
        # Load the GraphML file
        graph = nx.read_graphml(graph_path)

        # Create a 3D spring layout with more separation
        pos = nx.spring_layout(graph, dim=3, seed=42, k=0.5)

        # Extract node positions
        x_nodes = [pos[node][0] for node in graph.nodes()]
        y_nodes = [pos[node][1] for node in graph.nodes()]
        z_nodes = [pos[node][2] for node in graph.nodes()]

        # Extract edge positions
        x_edges, y_edges, z_edges = [], [], []
        for edge in graph.edges():
            x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
            z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])

        # Generate node colors based on node degree
        node_colors = [graph.degree(node) for node in graph.nodes()]
        node_colors = np.array(node_colors)
        node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())

        # Create the trace for edges
        edge_trace = go.Scatter3d(
            x=x_edges, y=y_edges, z=z_edges,
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            hoverinfo='none'
        )

        # Create the trace for nodes
        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers+text',
            marker=dict(
                size=7,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(
                    title='Node Degree',
                    thickness=10,
                    x=1.1,
                    tickvals=[0, 1],
                    ticktext=['Low', 'High']
                ),
                line=dict(width=1)
            ),
            text=[node for node in graph.nodes()],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            hoverinfo='text'
        )

        # Create the 3D plot
        fig = go.Figure(data=[edge_trace, node_trace])

        # Update layout for better visualization
        fig.update_layout(
            title=f'3D Graph Visualization: {os.path.basename(graph_path)}',
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
                    text="Interactive 3D visualization of GraphML data",
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



def chat_with_llm(message, history, system_message, temperature, max_tokens, model):
    messages = [{"role": "system", "content": system_message}]
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": message})

    try:
        response = chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def send_message(root_dir, query_type, query, history, system_message, temperature, max_tokens, model):
    if query_type == "global":
        result = run_query(root_dir, "global", query, history)
        history.append((query, result))
    elif query_type == "local":
        result = run_query(root_dir, "local", query, history)
        history.append((query, result))
    else:  # Direct chat
        result = chat_with_llm(query, history, system_message, temperature, max_tokens, model)
        history.append((query, result))
    return history, gr.update(value=""), update_logs()

def fetch_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']]
        else:
            return ["Error fetching models"]
    except Exception as e:
        return [f"Error: {str(e)}"]

def update_model_choices():
    models = fetch_ollama_models()
    return gr.update(choices=models, value=models[0] if models else None)

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
    overflow-y: auto;
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
    folders = list_output_folders(root_dir.value)
    return gr.update(choices=folders, value=folders[0] if folders else None)

def update_folder_content_list(root_dir, folder_name):
    if not folder_name:
        return gr.update(choices=[])
    contents = list_folder_contents(os.path.join(root_dir, "output", folder_name))
    return gr.update(choices=contents)

def handle_content_selection(root_dir, folder_name, selected_item):
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

def initialize_selected_folder(root_dir, folder_name):
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
                    
                    
                    with gr.Accordion("Indexing", open=True):
                        root_dir = gr.Textbox(label="Root Directory", value=os.path.abspath("./ragtest"))
                        index_btn = gr.Button("Run Indexing", variant="primary")
                        index_output = gr.Textbox(label="Indexing Output", lines=10, visible=True)
                        index_progress = gr.Textbox(label="Indexing Progress", visible=True)
                
                with gr.TabItem("Indexing Outputs"):
                    output_folder_list = gr.Dropdown(label="Select Output Folder", choices=[], interactive=True)
                    refresh_folder_btn = gr.Button("Refresh Folder List", variant="secondary")
                    initialize_folder_btn = gr.Button("Initialize Selected Folder", variant="primary")
                    folder_content_list = gr.Dropdown(label="Select File or Directory", choices=[], interactive=True)
                    file_info = gr.Textbox(label="File Information", interactive=False)
                    output_content = gr.TextArea(label="File Content", lines=20, interactive=False)
                    initialization_status = gr.Textbox(label="Initialization Status")
                
                with gr.TabItem("Settings"):
                    settings = load_settings()
                    with gr.Group():
                        for key, value in settings.items():
                            create_setting_component(key, value)

            with gr.Group(elem_id="log-container"):
                log_output = gr.TextArea(label="Logs", elem_id="log-output")

        with gr.Column(scale=2, elem_id="right-column"):
            with gr.Group(elem_id="chat-container"):
                chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot")
                with gr.Row(elem_id="chat-input-row"):
                    query_type = gr.Radio(["global", "local", "direct"], label="Query Type", value="global")
                    with gr.Column(scale=1):
                        query_input = gr.Textbox(
                            label="Query",
                            placeholder="Enter your query here...",
                            elem_id="query-input"
                        )
                        query_btn = gr.Button("Send Query", variant="primary")
                
                with gr.Accordion("Model Parameters", open=False):
                    system_message = gr.Textbox(label="System Message", value="You are a helpful assistant.", lines=2)
                    temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.7, step=0.1)
                    max_tokens = gr.Slider(label="Max Tokens", minimum=1, maximum=4096, value=150, step=1)
                    model = gr.Dropdown(label="Model", choices=[default_model] + fetch_ollama_models(), value=default_model)
                    refresh_models_btn = gr.Button("Refresh Models", variant="secondary")
                

                with gr.Group(elem_id="visualization-container"):
                    vis_output = gr.Plot(label="Graph Visualization", elem_id="visualization-plot")
                    with gr.Row(elem_id="vis-controls-row"):
                        vis_btn = gr.Button("Visualize Graph", variant="secondary")
                    vis_status = gr.Textbox(label="Visualization Status", elem_id="vis-status", show_label=False)

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
    index_btn.click(
        fn=index_graph,
        inputs=[root_dir],
        outputs=[index_output, log_output],
        show_progress=True
    )
    refresh_folder_btn.click(fn=update_output_folder_list, outputs=[output_folder_list]).then(
        fn=update_logs,
        outputs=[log_output]
    )
    output_folder_list.change(fn=update_folder_content_list, inputs=[root_dir, output_folder_list], outputs=[folder_content_list]).then(
        fn=update_logs,
        outputs=[log_output]
    )
    folder_content_list.change(fn=handle_content_selection, inputs=[root_dir, output_folder_list, folder_content_list], outputs=[folder_content_list, file_info, output_content]).then(
        fn=update_logs,
        outputs=[log_output]
    )
    initialize_folder_btn.click(fn=initialize_selected_folder, inputs=[root_dir, output_folder_list], outputs=[initialization_status, folder_content_list]).then(
        fn=update_logs,
        outputs=[log_output]
    )
    vis_btn.click(fn=update_visualization, inputs=[root_dir, output_folder_list, folder_content_list], outputs=[vis_output, vis_status]).then(
        fn=update_logs,
        outputs=[log_output]
    )
    query_btn.click(
        fn=send_message,
        inputs=[root_dir, query_type, query_input, chatbot, system_message, temperature, max_tokens, model],
        outputs=[chatbot, query_input, log_output]
    )
    query_input.submit(
        fn=send_message,
        inputs=[root_dir, query_type, query_input, chatbot, system_message, temperature, max_tokens, model],
        outputs=[chatbot, query_input, log_output]
    )
    refresh_models_btn.click(
        fn=update_model_choices,
        outputs=[model]
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

if __name__ == "__main__":
    demo.launch()
