"""MedCollab — Utilities Package"""

from src.utils.llm import get_llm
from src.utils.visualization import render_causal_chain, build_nx_graph


def extract_content(response) -> str:
    """
    Extract text content from an LLM response, handling provider differences.

    - OpenAI/Anthropic: response.content is a plain string
    - Gemini: response.content is a list of dicts like
      [{'type': 'text', 'text': '...', 'extras': {...}}]
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        return "\n".join(parts).strip()
    return str(content).strip()


__all__ = ["get_llm", "render_causal_chain", "build_nx_graph", "extract_content"]
