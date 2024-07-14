# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing build_steps method definition."""

from graphrag.index.config import PipelineWorkflowConfig, PipelineWorkflowStep
import logging

logger = logging.getLogger(__name__)

workflow_name = "create_base_entity_graph"


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
        }
    ]

    if embed_graph_enabled:
        steps.append({
            "verb": "embed_graph",
            "args": {
                **embed_graph_config,
                "column": "clustered_graph",
                "to": "embedded_graph",
            },
        })

    if graphml_snapshot_enabled:
        steps.append({
            "verb": "snapshot_rows",
            "args": {
                "base_name": "entity_graph",
                "column": "embedded_graph" if embed_graph_enabled else "clustered_graph",
                "formats": [{"format": "text", "extension": "graphml"}],
            },
        })

    logger.info(f"Created {len(steps)} steps for {workflow_name}")
    return steps