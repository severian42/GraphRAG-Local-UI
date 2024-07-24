# GraphRAG Indexer Application

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Application Structure](#application-structure)
4. [Indexing](#indexing)
5. [Prompt Tuning](#prompt-tuning)
6. [Data Management](#data-management)
7. [Configuration](#configuration)
8. [API Integration](#api-integration)
9. [Troubleshooting](#troubleshooting)

## Introduction

The GraphRAG Indexer Application is a Gradio-based user interface for managing the indexing and prompt tuning processes of the GraphRAG (Graph Retrieval-Augmented Generation) system. This application provides an intuitive way to configure, run, and monitor indexing and prompt tuning tasks, as well as manage related data files.

## Setup

1. Ensure you have Python 3.7+ installed.
2. Install required dependencies:
   ```
   pip install gradio requests pydantic python-dotenv pyyaml pandas lancedb
   ```
3. Set up environment variables in `indexing/.env`:
   ```
   API_BASE_URL=http://localhost:8012
   LLM_API_BASE=http://localhost:11434
   EMBEDDINGS_API_BASE=http://localhost:11434
   ROOT_DIR=indexing
   ```
4. Run the application:
   ```
   python index_app.py
   ```

## Application Structure

The application is divided into three main tabs:
1. Indexing
2. Prompt Tuning
3. Data Management

Each tab provides specific functionality related to its purpose.

## Indexing

The Indexing tab allows users to configure and run the GraphRAG indexing process.

### Features:
- Select LLM and Embedding models
- Set root directory for indexing
- Configure verbose and cache options
- Advanced options for resuming, reporting, and output formats
- Run indexing and check status

### Usage:
1. Select the desired LLM and Embedding models from the dropdowns.
2. Set the root directory for indexing.
3. Configure additional options as needed.
4. Click "Run Indexing" to start the process.
5. Use "Check Indexing Status" to monitor progress.

## Prompt Tuning

The Prompt Tuning tab enables users to configure and run prompt tuning for GraphRAG.

### Features:
- Set root directory and domain
- Choose tuning method (random, top, all)
- Configure limit, language, max tokens, and chunk size
- Option to exclude entity types
- Run prompt tuning and check status

### Usage:
1. Set the root directory and optional domain.
2. Choose the tuning method and configure parameters.
3. Click "Run Prompt Tuning" to start the process.
4. Use "Check Prompt Tuning Status" to monitor progress.

## Data Management

The Data Management tab provides tools for managing input files and viewing output folders.

### Features:
- File upload functionality
- File list management (view, refresh, delete)
- Output folder exploration
- File content viewing and editing

### Usage:
1. Use the File Upload section to add new input files.
2. Manage existing files in the File Management section.
3. Explore output folders and their contents in the Output Folders section.

## Configuration

The application uses a combination of environment variables and a `config.yaml` file for configuration. Key settings include:

- LLM and Embedding models
- API endpoints
- Community level for GraphRAG
- Token limits
- API keys and types

To modify these settings, edit the `.env` file or create a `config.yaml` file in the root directory.

## API Integration

The application integrates with a backend API for executing indexing and prompt tuning tasks. Key API endpoints used:

- `/v1/index`: Start indexing process
- `/v1/index_status`: Check indexing status
- `/v1/prompt_tune`: Start prompt tuning process
- `/v1/prompt_tune_status`: Check prompt tuning status

These endpoints are called using the `requests` library, with appropriate error handling and logging.

## Troubleshooting

Common issues and solutions:

1. **Model loading fails**: Ensure the LLM_API_BASE is correctly set and the API is accessible.
2. **Indexing or Prompt Tuning doesn't start**: Check API connectivity and verify that all required fields are filled.
3. **File management issues**: Ensure proper read/write permissions in the ROOT_DIR.

For any persistent issues, check the application logs (visible in the console) for detailed error messages.