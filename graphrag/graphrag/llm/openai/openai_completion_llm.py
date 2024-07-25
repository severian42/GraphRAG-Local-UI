# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A text-completion based LLM."""

import logging
from typing import Any, Dict
from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
from .utils import get_completion_llm_args

log = logging.getLogger(__name__)


class OpenAICompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        super().__init__()
        self._client = client
        self._configuration = configuration

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Dict[str, Any]
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self._configuration
        )
        try:
            # Remove 'model' from args if it's there, as we'll use self._configuration.model
            args.pop('model', None)
            completion = await self._client.completions.create(
                model=self._configuration.model,
                prompt=input,
                **args
            )
            return completion.choices[0].text
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}") from e