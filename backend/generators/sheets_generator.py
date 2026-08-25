"""Sheets generator - turns a prompt into a real, downloadable .xlsx file.

Same two-step pattern: LLM drafts structured tabular content as JSON
(headers + rows), this module renders it into an actual Excel file.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict

from openpyxl import Workbook
from openpyxl.styles import Font


DRAFT_SYSTEM_PROMPT = (
    "You are a spreadsheet drafting assistant. Given a topic, produce "
    "tabular data as JSON with this exact shape:\n"
    '{"sheet_name": "...", "headers": ["Col1", "Col2"], "rows": [["a", "b"], ["c", "d"]]}\n'
    "Every row must have the same number of cells as headers. "
    "Return ONLY the JSON, no other text, no markdown code fences."
)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    return json.loads(text)


def draft_sheet_content(llm, topic: str) -> Dict[str, Any]:
    response = llm.chat(
        [{"role": "user", "content": f"Create a spreadsheet about: {topic}"}],
        system_prompt=DRAFT_SYSTEM_PROMPT,
        temperature=0.3,
    )
    try:
        return _extract_json(response.text)
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(
            f"Model didn't return valid JSON for sheet content: {exc}. "
            f"Raw response started with: {response.text[:200]!r}"
        ) from exc


def build_xlsx(content: Dict[str, Any], output_dir: str) -> str:
    """Render drafted content into a real .xlsx file. Returns the file path."""
    wb = Workbook()
    ws = wb.active
    ws.title = (content.get("sheet_name") or "Sheet1")[:31]  # Excel's sheet-name length limit

    headers = content.get("headers", [])
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for row in content.get("rows", []):
        # Defensive: pad/truncate rows that don't match header length rather
        # than letting openpyxl raise or silently misalign columns.
        fixed_row = (list(row) + [""] * len(headers))[:len(headers)] if headers else row
        ws.append(fixed_row)

    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"sheet_{uuid.uuid4().hex[:10]}.xlsx"
    path = os.path.join(output_dir, filename)
    wb.save(path)
    return path


def generate_xlsx(llm, topic: str, output_dir: str) -> str:
    content = draft_sheet_content(llm, topic)
    return build_xlsx(content, output_dir)
