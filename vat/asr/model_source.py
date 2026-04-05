"""Whisper 模型来源可用性检查。"""

from pathlib import Path
from typing import Optional


def resolve_faster_whisper_repo_id(model_name: str) -> str:
    import faster_whisper.utils as faster_whisper_utils

    return faster_whisper_utils._MODELS.get(model_name, model_name)


def has_local_whisper_cache(model_name: str, download_root: Optional[str]) -> bool:
    """检查本地是否已有可用的 Whisper 模型缓存。"""
    required_files = ("model.bin", "config.json")
    repo_id = resolve_faster_whisper_repo_id(model_name)

    if download_root:
        root = Path(download_root)
        if all((root / name).exists() for name in required_files):
            return True

        repo_cache_dir = root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        if repo_cache_dir.exists():
            for snapshot_dir in repo_cache_dir.iterdir():
                if snapshot_dir.is_dir() and all((snapshot_dir / name).exists() for name in required_files):
                    return True

        try:
            from huggingface_hub.file_download import _CACHED_NO_EXIST, try_to_load_from_cache

            for name in required_files:
                cached = try_to_load_from_cache(repo_id, name, cache_dir=root)
                if cached is None or cached is _CACHED_NO_EXIST:
                    return False
            return True
        except Exception:
            return False

    try:
        from huggingface_hub.file_download import _CACHED_NO_EXIST, try_to_load_from_cache

        for name in required_files:
            cached = try_to_load_from_cache(repo_id, name)
            if cached is None or cached is _CACHED_NO_EXIST:
                return False
        return True
    except Exception:
        return False


def can_access_huggingface(timeout_seconds: float = 5.0) -> bool:
    """快速检查 HuggingFace 连通性。"""
    try:
        import httpx

        response = httpx.get("https://huggingface.co", timeout=timeout_seconds, follow_redirects=True)
        return response.status_code < 500
    except Exception:
        return False


def format_whisper_model_load_error(model_name: str, download_root: Optional[str], error: Exception) -> str:
    """格式化 Whisper 模型加载失败信息。"""
    model_dir = download_root or "默认 HuggingFace 缓存目录"
    return (
        f"Whisper 模型加载失败（model={model_name}）。"
        f"首次 ASR 需要可用网络下载 Whisper 模型，或预先将模型缓存放到 {model_dir}。"
        f"如果当前环境不能访问 HuggingFace，请先在有网环境下载模型后再重试。"
        f"原始错误: {type(error).__name__}: {error}"
    )


def ensure_whisper_model_source_available(
    model_name: str,
    download_root: Optional[str],
    timeout_seconds: float = 5.0,
) -> None:
    """在真正加载模型前，快速检查模型来源是否可用。"""
    if has_local_whisper_cache(model_name, download_root):
        return

    if can_access_huggingface(timeout_seconds=timeout_seconds):
        return

    raise RuntimeError(
        format_whisper_model_load_error(
            model_name,
            download_root,
            RuntimeError("无法连接 HuggingFace，且本地模型缓存不存在"),
        )
    )
