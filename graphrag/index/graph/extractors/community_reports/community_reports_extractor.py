# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

import logging
import traceback
from dataclasses import dataclass
from typing import Any
import json

from graphrag.index.typing import ErrorHandlerFn
from graphrag.index.utils import dict_has_keys_with_types
from graphrag.llm import CompletionLLM

from .prompts import COMMUNITY_REPORT_PROMPT

log = logging.getLogger(__name__)


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: dict


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _llm: CompletionLLM
    _input_text_key: str
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        input_text_key: str | None = None,
        extraction_prompt: str | None = None,
        on_error: ErrorHandlerFn | None = None,
        max_report_length: int | None = None,
    ):
        """Init method definition."""
        self._llm = llm_invoker
        self._input_text_key = input_text_key or "input_text"
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500

    async def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Call method definition."""
        try:
            response = await self._llm(
                self._extraction_prompt,
                name="create_community_report",
                variables={self._input_text_key: inputs[self._input_text_key]},
                model_parameters={"max_tokens": self._max_report_length},
            )
            
            try:
                output = json.loads(response.text)
            except json.JSONDecodeError:
                # If JSON parsing fails, use the text as is
                output = {"full_content": response.text}
            
            # Ensure all expected fields exist
            default_output = {
                "title": "",
                "summary": "",
                "rating": 0.0,
                "rating_explanation": "",
                "findings": [],
                "community": inputs.get('community', 'default'),
                "full_content": response.text
            }
            default_output.update(output)
            
            return default_output
        except Exception as e:
            logging.error(f"Error generating community report: {str(e)}")
            return {
                "title": "",
                "summary": "",
                "rating": 0.0,
                "rating_explanation": "",
                "findings": [],
                "community": inputs.get('community', 'default'),
                "full_content": f"Error generating report: {str(e)}"
            }

    def _get_text_output(self, parsed_output: dict) -> str:
        title = parsed_output.get("title", "Report")
        summary = parsed_output.get("summary", "")
        findings = parsed_output.get("findings", [])

        def finding_summary(finding: dict):
            if isinstance(finding, str):
                return finding
            return finding.get("summary")

        def finding_explanation(finding: dict):
            if isinstance(finding, str):
                return ""
            return finding.get("explanation")

        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        return f"# {title}\n\n{summary}\n\n{report_sections}"

    def _parse_non_json_output(self, text: str) -> dict:
        lines = text.split('\n')
        output = {
            "title": "",
            "summary": "",
            "rating": 0.0,
            "rating_explanation": "",
            "findings": []
        }
        
        current_section = None
        current_finding = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("TITLE:"):
                output["title"] = line[6:].strip()
            elif line.startswith("SUMMARY:"):
                current_section = "summary"
            elif line.startswith("IMPACT SEVERITY RATING:"):
                try:
                    output["rating"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.startswith("RATING EXPLANATION:"):
                output["rating_explanation"] = line.split(":")[-1].strip()
            elif line.startswith("DETAILED FINDINGS:"):
                current_section = "findings"
            elif current_section == "summary":
                output["summary"] += line + " "
            elif current_section == "findings":
                if line.startswith("- "):
                    if current_finding:
                        output["findings"].append(current_finding)
                    current_finding = {"summary": line[2:], "explanation": ""}
                elif current_finding:
                    current_finding["explanation"] += line + " "
        
        if current_finding:
            output["findings"].append(current_finding)
        
        output["summary"] = output["summary"].strip()
        for finding in output["findings"]:
            finding["explanation"] = finding["explanation"].strip()
        
        return output