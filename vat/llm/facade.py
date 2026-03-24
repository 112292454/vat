"""高层 LLM 调用 facade。"""

import json
from typing import Any, Dict, List

from vat.llm.client import call_llm


def call_text_llm(
    *,
    messages: List[dict],
    model: str,
    temperature: float = 1.0,
    api_key: str = "",
    base_url: str = "",
    proxy: str = "",
    **kwargs: Any,
) -> str:
    """统一的文本调用入口，返回 message.content。"""
    response = call_llm(
        messages=messages,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        proxy=proxy,
        **kwargs,
    )
    return response.choices[0].message.content.strip()


def extract_json_block(content: str) -> str:
    """从普通文本或 markdown code block 中提取 JSON 文本。"""
    text = content.strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def call_json_llm(
    *,
    messages: List[dict],
    model: str,
    temperature: float = 1.0,
    api_key: str = "",
    base_url: str = "",
    proxy: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """统一的 JSON 调用入口。"""
    content = call_text_llm(
        messages=messages,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        proxy=proxy,
        **kwargs,
    )
    return json.loads(extract_json_block(content))
