"""Website generator - turns a prompt into a real, downloadable static
index.html file.

Honest scope: this produces a single static HTML file the user can
download, preview, or host anywhere (Vercel/Netlify/S3/GitHub Pages).
It does NOT deploy anything live on its own - "Websites" in the big AI
chat products usually means a full build+hosting pipeline, which is out
of scope here. If live one-click hosting is wanted later, that's a
separate, larger feature (e.g. calling Vercel's deploy API).
"""
from __future__ import annotations

import os
import re
import uuid


DRAFT_SYSTEM_PROMPT = (
    "You are a web developer. Given a description, write a single, complete, "
    "self-contained HTML file (inline <style> and <script>, no external "
    "dependencies) that implements it. Return ONLY the raw HTML, starting "
    "with <!DOCTYPE html>, no markdown code fences, no explanation."
)


def _extract_html(text: str) -> str:
    text = text.strip()
    fence_match = re.search(r"```(?:html)?\s*(<!DOCTYPE.*?</html>)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1)
    doctype_idx = text.lower().find("<!doctype")
    if doctype_idx != -1:
        return text[doctype_idx:]
    return text  # best effort - still save something rather than nothing


def draft_website_html(llm, description: str) -> str:
    response = llm.chat(
        [{"role": "user", "content": f"Build a website: {description}"}],
        system_prompt=DRAFT_SYSTEM_PROMPT,
        temperature=0.5,
        max_tokens=3000,
    )
    return _extract_html(response.text)


def build_website(html: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"site_{uuid.uuid4().hex[:10]}.html"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(html)
    return path


def generate_website(llm, description: str, output_dir: str) -> str:
    html = draft_website_html(llm, description)
    return build_website(html, output_dir)
