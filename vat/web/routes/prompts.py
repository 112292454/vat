"""Custom Prompt 管理 API"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/prompts", tags=["prompts"])

# Custom prompts 目录
CUSTOM_PROMPTS_DIR = Path(__file__).parent.parent.parent / "llm" / "prompts" / "custom"


class PromptInfo(BaseModel):
    """Prompt 信息"""
    name: str  # 文件名（不含扩展名）
    type: str  # translate 或 optimize
    content: Optional[str] = None


class PromptContent(BaseModel):
    """Prompt 内容"""
    content: str


def _get_prompt_path(prompt_type: str, name: str) -> Path:
    """获取 prompt 文件路径"""
    if prompt_type not in ("translate", "optimize"):
        raise HTTPException(400, "type 必须是 translate 或 optimize")
    
    # 安全检查：防止路径遍历
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "无效的 prompt 名称")
    
    return CUSTOM_PROMPTS_DIR / prompt_type / f"{name}.md"


@router.get("")
async def list_prompts():
    """列出所有 custom prompts"""
    prompts = []
    
    for prompt_type in ("translate", "optimize"):
        type_dir = CUSTOM_PROMPTS_DIR / prompt_type
        if type_dir.exists():
            for f in type_dir.glob("*.md"):
                prompts.append({
                    "name": f.stem,
                    "type": prompt_type,
                    "size": f.stat().st_size
                })
    
    return {"prompts": prompts}


@router.get("/{prompt_type}/{name}")
async def get_prompt(prompt_type: str, name: str):
    """获取 prompt 内容"""
    path = _get_prompt_path(prompt_type, name)
    
    if not path.exists():
        raise HTTPException(404, "Prompt 不存在")
    
    content = path.read_text(encoding="utf-8")
    return {
        "name": name,
        "type": prompt_type,
        "content": content
    }


@router.post("/{prompt_type}/{name}")
async def create_prompt(prompt_type: str, name: str, data: PromptContent):
    """创建新 prompt"""
    path = _get_prompt_path(prompt_type, name)
    
    if path.exists():
        raise HTTPException(400, "Prompt 已存在，请使用 PUT 更新")
    
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    path.write_text(data.content, encoding="utf-8")
    return {"message": "创建成功", "name": name, "type": prompt_type}


@router.put("/{prompt_type}/{name}")
async def update_prompt(prompt_type: str, name: str, data: PromptContent):
    """更新 prompt"""
    path = _get_prompt_path(prompt_type, name)
    
    if not path.exists():
        raise HTTPException(404, "Prompt 不存在")
    
    path.write_text(data.content, encoding="utf-8")
    return {"message": "更新成功", "name": name, "type": prompt_type}


@router.delete("/{prompt_type}/{name}")
async def delete_prompt(prompt_type: str, name: str):
    """删除 prompt"""
    path = _get_prompt_path(prompt_type, name)
    
    if not path.exists():
        raise HTTPException(404, "Prompt 不存在")
    
    path.unlink()
    return {"message": "删除成功", "name": name, "type": prompt_type}
