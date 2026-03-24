"""场景自动识别模块"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from vat.llm.facade import call_text_llm
from vat.utils.logger import setup_logger

logger = setup_logger("scene_identifier")

SCENES_CONFIG_PATH = Path(__file__).parent / "scenes.yaml"


class SceneIdentifier:
    """场景识别器，基于 LLM 判断视频场景"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = "", base_url: str = "", proxy: str = ""):
        """
        初始化场景识别器
        
        Args:
            model: LLM 模型名称
            api_key: API Key 覆写（空=使用全局配置）
            base_url: Base URL 覆写（空=使用全局配置）
            proxy: 代理地址覆写（空=使用环境变量）
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.proxy = proxy
        self.scenes_config = self._load_scenes_config()
    
    def _load_scenes_config(self) -> Dict[str, Any]:
        """加载场景配置文件"""
        if not SCENES_CONFIG_PATH.exists():
            logger.warning(f"场景配置文件不存在: {SCENES_CONFIG_PATH}")
            return {"scenes": [], "default_scene": "chatting"}
        
        with open(SCENES_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"已加载 {len(config.get('scenes', []))} 个场景配置")
        return config
    
    def detect_scene(
        self, 
        title: str, 
        description: str = ""
    ) -> Dict[str, Any]:
        """
        检测视频场景
        
        Args:
            title: 视频标题
            description: 视频简介
            
        Returns:
            {
                "scene_id": str,
                "scene_name": str,
                "auto_detected": bool
            }
        """
        if not title:
            logger.warning("视频标题为空，使用默认场景")
            return self._get_default_scene()
        
        try:
            # 构建 system prompt
            system_prompt = self._build_system_prompt()
            
            # 构建 user prompt
            user_prompt = f"""Title: {title}

Description: {description if description else "(no description)"}

Output: """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            # 调用 LLM
            scene_id = call_text_llm(
                messages=messages, model=self.model, temperature=0.1,
                api_key=self.api_key, base_url=self.base_url,
                proxy=self.proxy,
            )
            scene_id = scene_id.lower()
            
            # 验证场景 ID 是否有效
            if not self._is_valid_scene(scene_id):
                logger.warning(f"LLM 返回无效场景: {scene_id}，使用默认场景")
                return self._get_default_scene()
            
            scene_name = self._get_scene_name(scene_id)
            logger.info(f"场景识别成功: {scene_id} ({scene_name})")
            
            return {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "auto_detected": True,
            }
        
        except Exception as e:
            logger.error(f"场景识别失败: {e}，使用默认场景")
            return self._get_default_scene()
    
    def _build_system_prompt(self) -> str:
        """构建场景识别的 system prompt"""
        scenes = self.scenes_config.get("scenes", [])
        
        scene_list = []
        for scene in scenes:
            scene_id = scene["id"]
            name = scene["name"]
            desc = scene["description"]
            keywords = ", ".join(scene.get("keywords", []))
            scene_list.append(f"{scene_id} - {name}: {desc} (keywords: {keywords})")
        
        prompt = f"""You are a video content classifier for vtuber livestream videos.

Based on the video title and description, determine which scene type best matches:

{chr(10).join(scene_list)}

Rules:
1. Output ONLY the scene ID (lowercase), no explanation
2. If uncertain, output "{self.scenes_config.get('default_scene', 'chatting')}"
3. Consider keywords, but also understand the context

Examples:
- "【Minecraft】建造天空城堡！" → gaming
- "雑談配信～最近あったこと話す～" → chatting
- "ASMR 耳かき💕" → asmr
- "歌ってみた！カラオケ配信" → singing
"""
        return prompt
    
    def _is_valid_scene(self, scene_id: str) -> bool:
        """验证场景 ID 是否有效"""
        scenes = self.scenes_config.get("scenes", [])
        return any(scene["id"] == scene_id for scene in scenes)
    
    def _get_scene_name(self, scene_id: str) -> str:
        """获取场景名称"""
        scenes = self.scenes_config.get("scenes", [])
        for scene in scenes:
            if scene["id"] == scene_id:
                return scene["name"]
        return "Unknown"
    
    def _get_default_scene(self) -> Dict[str, Any]:
        """获取默认场景"""
        default_id = self.scenes_config.get("default_scene", "chatting")
        return {
            "scene_id": default_id,
            "scene_name": self._get_scene_name(default_id),
            "auto_detected": False,
        }
    
    def get_scene_prompts(self, scene_id: str) -> Dict[str, str]:
        """
        获取场景的提示词
        
        Args:
            scene_id: 场景 ID
            
        Returns:
            {"split": str, "translate": str, "optimize": str}
        """
        scenes = self.scenes_config.get("scenes", [])
        for scene in scenes:
            if scene["id"] == scene_id:
                return scene.get("prompts", {})
        return {}
