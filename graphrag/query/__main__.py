# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Query Engine package root."""

import argparse
from enum import Enum
import logging
import os

from .cli import run_global_search, run_local_search

INVALID_METHOD_ERROR = "Invalid method"


class SearchType(Enum):
    """The type of search to run."""

    LOCAL = "local"
    GLOBAL = "global"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        help="The path with the output data from the pipeline",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--root",
        help="The data project root. Default value: the current directory",
        required=False,
        default=".",
        type=str,
    )

    parser.add_argument(
        "--method",
        help="The method to run, one of: local or global",
        required=True,
        type=SearchType,
        choices=list(SearchType),
    )

    parser.add_argument(
        "--community_level",
        help="Community level in the Leiden community hierarchy from which we will load the community reports higher value means we use reports on smaller communities",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--response_type",
        help="Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report",
        type=str,
        default="Multiple Paragraphs",
    )

    parser.add_argument(
        "query",
        nargs=1,
        help="The query to run",
        type=str,
    )

    args = parser.parse_args()

    # Create the config
    from graphrag.config import create_graphrag_config
    config = create_graphrag_config(root_dir=args.root)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Created config with root_dir: {config.root_dir}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Root directory: {args.root}")

    match args.method:
        case SearchType.LOCAL:
            run_local_search(
                args.data,
                args.root,
                args.community_level,
                args.response_type,
                args.query[0],
                config,
            )
        case SearchType.GLOBAL:
            run_global_search(
                args.data,
                args.root,
                args.community_level,
                args.response_type,
                args.query[0],
                config,
            )
        case _:
            raise ValueError(INVALID_METHOD_ERROR)