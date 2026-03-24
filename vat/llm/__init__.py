"""
VAT LLM模块 - 统一的LLM客户端接口
"""
from .client import call_llm, get_llm_client, prewarm_vertex_access_token
from .facade import call_text_llm, call_json_llm, extract_json_block

__all__ = [
    "call_llm",
    "get_llm_client",
    "prewarm_vertex_access_token",
    "call_text_llm",
    "call_json_llm",
    "extract_json_block",
]
