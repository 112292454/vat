"""
VAT LLM模块 - 统一的LLM客户端接口
"""
from .client import call_llm, get_llm_client

__all__ = ["call_llm", "get_llm_client"]
