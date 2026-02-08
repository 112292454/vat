"""
文件管理 API

支持查看和编辑视频输出目录中的文件
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os

from vat.database import Database
from vat.web.deps import get_db

router = APIRouter(prefix="/api/files", tags=["files"])


def ranged_file_response(file_path: Path, request: Request, media_type: str):
    """
    支持 HTTP Range 请求的文件响应（用于视频/音频 seek）
    """
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    
    if range_header:
        # 解析 Range header: bytes=start-end
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        
        # 确保范围有效
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        content_length = end - start + 1
        
        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 64 * 1024  # 64KB chunks
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        return StreamingResponse(
            iter_file(),
            status_code=206,  # Partial Content
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            }
        )
    else:
        # 无 Range header，返回完整文件
        return FileResponse(
            file_path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes"}
        )


class SaveFileRequest(BaseModel):
    """保存文件请求"""
    content: str


@router.get("/view/{video_id}/{filename:path}")
async def view_file(video_id: str, filename: str, request: Request, db: Database = Depends(get_db)):
    """
    查看文件内容
    
    - 文本文件返回 JSON 格式内容
    - 视频/音频返回支持 Range 的流式响应（支持 seek）
    """
    
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    
    if not video.output_dir:
        raise HTTPException(404, "Video output directory not set")
    
    output_dir = Path(video.output_dir)
    target_file = output_dir / filename
    
    # 安全检查：确保在 output_dir 内
    try:
        target_file.resolve().relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    
    if not target_file.exists():
        raise HTTPException(404, "File not found")
    
    if not target_file.is_file():
        raise HTTPException(400, "Not a file")
    
    # 根据文件类型返回
    suffix = target_file.suffix.lower()
    
    if suffix in [".srt", ".vtt", ".txt", ".json", ".yaml", ".md", ".ass", ".log"]:
        # 文本文件：返回 JSON 格式内容
        try:
            content = target_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = target_file.read_text(encoding="utf-8", errors="replace")
        return {
            "type": "text",
            "filename": filename,
            "content": content,
            "size": target_file.stat().st_size,
            "editable": suffix in [".srt", ".txt", ".json", ".ass"]
        }
    
    elif suffix in [".mp4", ".webm", ".mkv"]:
        # 视频文件：支持 Range 请求（可 seek）
        media_types = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska"}
        return ranged_file_response(target_file, request, media_types.get(suffix, "video/mp4"))
    
    elif suffix in [".mp3", ".wav", ".m4a"]:
        # 音频文件：支持 Range 请求（可 seek）
        media_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
        return ranged_file_response(target_file, request, media_types.get(suffix, "audio/mpeg"))
    
    elif suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        # 图片文件
        media_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"
        }
        return FileResponse(target_file, media_type=media_types.get(suffix, "image/jpeg"))
    
    else:
        # 其他文件：尝试作为文本读取
        try:
            content = target_file.read_text(encoding="utf-8")
            return {"type": "text", "filename": filename, "content": content, "editable": False}
        except Exception:
            raise HTTPException(400, f"Unsupported file type: {suffix}")


@router.put("/save/{video_id}/{filename:path}")
async def save_file(video_id: str, filename: str, request: SaveFileRequest, db: Database = Depends(get_db)):
    """
    保存文件（仅支持文本文件）
    """
    import shutil
    
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    
    if not video.output_dir:
        raise HTTPException(404, "Video output directory not set")
    
    output_dir = Path(video.output_dir)
    target_file = output_dir / filename
    
    # 安全检查
    try:
        target_file.resolve().relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    
    if not target_file.exists():
        raise HTTPException(404, "File not found")
    
    suffix = target_file.suffix.lower()
    if suffix not in [".srt", ".txt", ".json", ".ass"]:
        raise HTTPException(400, "Only text files can be edited")
    
    # 备份原文件
    backup_path = target_file.with_suffix(target_file.suffix + ".bak")
    shutil.copy(target_file, backup_path)
    
    # 写入新内容
    target_file.write_text(request.content, encoding="utf-8")
    
    return {
        "status": "saved",
        "filename": filename,
        "backup": backup_path.name
    }


@router.get("/download/{video_id}/{filename:path}")
async def download_file(video_id: str, filename: str, db: Database = Depends(get_db)):
    """下载文件"""
    
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    
    if not video.output_dir:
        raise HTTPException(404, "Video output directory not set")
    
    output_dir = Path(video.output_dir)
    target_file = output_dir / filename
    
    # 安全检查
    try:
        target_file.resolve().relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    
    if not target_file.exists():
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        target_file,
        filename=filename,
        media_type="application/octet-stream"
    )
