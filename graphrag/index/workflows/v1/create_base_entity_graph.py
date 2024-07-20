# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing build_steps method definition."""

from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep
import logging
import json

logger = logging.getLogger(__name__)

workflow_name = "create_base_entity_graph"

def convert_dict_to_json(value):
    if isinstance(value, dict):
        return json.dumps(value)
    return value

def build_steps(
    config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    """
    Create the base table for the entity graph.

    ## Dependencies
    * `workflow:create_summarized_entities`
    """
    clustering_config = config.get(
        "cluster_graph",
        {"strategy": {"type": "leiden"}},
    )
    embed_graph_config = config.get(
        "embed_graph",
        {
            "strategy": {
                "type": "node2vec",
                "num_walks": config.get("embed_num_walks", 10),
                "walk_length": config.get("embed_walk_length", 40),
                "window_size": config.get("embed_window_size", 2),
                "iterations": config.get("embed_iterations", 3),
                "random_seed": config.get("embed_random_seed", 86),
            }
        },
    )

    graphml_snapshot_enabled = config.get("graphml_snapshot", False)
    embed_graph_enabled = config.get("embed_graph_enabled", False)

    steps = [
        {
            "verb": "cluster_graph",
            "args": {
                **clustering_config,
                "column": "entity_graph",
                "to": "clustered_graph",
                "level_to": "level",
            },
            "input": {"source": "workflow:create_summarized_entities"},
        },
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "clustered_graph",
                "column": "clustered_graph",
                "formats": [{"format": "text", "extension": "graphml"}],
            },
        },
        {
            "verb": "embed_graph",
            "enabled": embed_graph_enabled,
            "args": {
                "column": "clustered_graph",
                "to": "embeddings",
                **embed_graph_config,
            },
        },
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "embedded_graph",
                "column": "entity_graph",
                "formats": [{"format": "text", "extension": "graphml"}],
            },
        },
    ]

    # Add steps for renaming columns and converting dictionaries to JSON
    steps.extend([
        {
            "verb": "rename",
            "args": {
                "columns": {
                    "function": "rename_and_convert_columns"
                }
            }
        },
    ])

    # Add steps to handle duplicate columns
    steps.extend([
        {
            "verb": "select",
            "args": {
                "columns": ["level", "entity_graph", "embeddings"] if embed_graph_enabled else ["level", "entity_graph"],
            },
        },
        {
            "verb": "rename",
            "args": {
                "columns": {
                    "level": "level_final",
                    "entity_graph": "entity_graph_final",
                    "embeddings": "embeddings_final" if embed_graph_enabled else None,
                }
            }
        },
        {
            "verb": "select",
            "args": {
                "columns": ["level_final", "entity_graph_final", "embeddings_final"] if embed_graph_enabled else ["level_final", "entity_graph_final"],
            },
        }
    ])

    logger.info(f"Created {len(steps)} steps for {workflow_name}")
    return steps
