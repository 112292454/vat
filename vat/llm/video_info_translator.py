"""
视频信息翻译器

负责翻译YouTube视频的标题、描述、标签，并智能判断B站分区
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .client import get_or_create_client
from vat.utils.logger import setup_logger


logger = setup_logger("video_info_translator")


@dataclass
class TranslatedVideoInfo:
    """翻译后的视频信息"""
    # 原始信息
    original_title: str
    original_description: str
    original_tags: List[str]
    
    # 翻译后的信息
    title_translated: str           # 翻译后的标题，格式：[主播名] 标题内容
    description_summary: str        # 简介摘要（1-2句话，用于简介开头）
    description_translated: str     # 完整忠实翻译的简介
    tags_translated: List[str]      # 翻译后的标签
    tags_generated: List[str]       # 生成的额外相关标签
    
    # B站分区推荐
    recommended_tid: int            # 推荐的B站分区ID
    recommended_tid_name: str       # 分区名称
    tid_reason: str                 # 推荐理由
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'original_title': self.original_title,
            'original_description': self.original_description,
            'original_tags': self.original_tags,
            'title_translated': self.title_translated,
            'description_summary': self.description_summary,
            'description_translated': self.description_translated,
            'tags_translated': self.tags_translated,
            'tags_generated': self.tags_generated,
            'recommended_tid': self.recommended_tid,
            'recommended_tid_name': self.recommended_tid_name,
            'tid_reason': self.tid_reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranslatedVideoInfo':
        """从字典创建，兼容旧数据"""
        # 兼容旧字段名
        if 'description_optimized' in data and 'description_summary' not in data:
            data['description_summary'] = data.pop('description_optimized')
        if 'title_optimized' in data:
            data.pop('title_optimized', None)  # 移除旧字段
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# B站分区映射（tid -> 名称）
# 数据来源: https://github.com/biliup/biliup/wiki
# 仅保留与 VTuber 内容相关的分区

# 大区 ID
BILIBILI_MAIN_ZONES = {
    160: "生活",
    4: "游戏",
    5: "娱乐",
    3: "音乐",
    1: "动画",
    129: "舞蹈",
    119: "鬼畜",
}

# 小区 ID -> (名称, 大区ID)
BILIBILI_ZONES = {
    # 生活区 (160) - VTuber 日常/杂谈首选
    21: ("日常", 160),
    138: ("搞笑", 160),
    162: ("绘画", 160),
    
    # 游戏区 (4) - 游戏直播/实况
    17: ("单机游戏", 4),
    65: ("网络游戏", 4),
    172: ("手机游戏", 4),
    171: ("电子竞技", 4),
    136: ("音游", 4),
    121: ("GMV", 4),
    
    # 娱乐区 (5) - 综艺/杂谈
    71: ("综艺", 5),
    137: ("明星", 5),
    
    # 音乐区 (3) - 唱歌/演奏
    31: ("翻唱", 3),
    59: ("演奏", 3),
    28: ("原创音乐", 3),
    130: ("音乐综合", 3),
    29: ("音乐现场", 3),
    193: ("MV", 3),
    
    # 动画区 (1) - 二次创作
    24: ("MAD·AMV", 1),
    25: ("MMD·3D", 1),
    27: ("综合", 1),
    47: ("短片·手书·配音", 1),
    
    # 舞蹈区 (129) - 宅舞/舞蹈
    20: ("宅舞", 129),
    154: ("舞蹈综合", 129),
    
    # 鬼畜区 (119) - 鬼畜/音MAD
    22: ("鬼畜调教", 119),
    26: ("音MAD", 119),
}

# 获取分区名称（兼容旧代码）
def get_zone_name(tid: int) -> str:
    """获取分区名称"""
    zone = BILIBILI_ZONES.get(tid)
    if zone:
        return zone[0] if isinstance(zone, tuple) else zone
    return BILIBILI_MAIN_ZONES.get(tid, "日常")

# 常见内容类型到分区的映射建议
CONTENT_TYPE_TO_TID = {
    "vtuber": 21,           # VTuber 日常/杂谈 -> 日常
    "gaming": 17,           # 游戏实况 -> 单机游戏
    "music": 31,            # 唱歌 -> 翻唱
    "singing": 31,          # 唱歌 -> 翻唱
    "chatting": 21,         # 聊天/杂谈 -> 日常
    "drawing": 162,         # 绘画 -> 绘画
    "dance": 20,            # 跳舞 -> 宅舞
    "comedy": 138,          # 搞笑 -> 搞笑
    "anime": 27,            # 动画相关 -> 综合
    "mmd": 25,              # MMD -> MMD·3D
}


TRANSLATE_VIDEO_INFO_PROMPT = '''将以下YouTube视频信息翻译为中文，用于搬运到B站（BiliBili）。

## 原始信息
频道/主播: {uploader}
标题: {title}
描述: {description}
标签: {tags}

## 翻译要求

翻译应灵活结合上下文，以忠实原文为原则，同时参考主播信息和视频内容，恰当润色以吸引观众。

### 主播名翻译参考
- 频道名 "{uploader}" 的中文常用译名请参考已有翻译惯例
- 常见VTuber译名：白上フブキ→白上吹雪/白上/吹雪/FBK/fubuki/好狐（对于此类可以有多种翻译或者简称/昵称的，应当结合语境灵活决定）、兎田ぺこら→兔田佩克拉、宝鐘マリン→宝钟玛琳、星街すいせい→星街彗星、rurudo→rurudo（保留原名）……（更多主播译名应该采用通用写法）
- 如不确定，保留原名或使用音译

### 标题翻译规则（极其重要！标题直接决定用户是否点击！）

**核心原则**：忠实原文含义 + 适度润色吸引点击。不能瞎编，但也不能干巴巴地直译。风格应该类似下述description_summary要求，保持简洁活泼可爱吸引人点击，但是更贴近原文，稍微正式一点。

1. **忠实原文**：标题翻译必须基于原文内容，不可添加原文没有的含义。但可以：
   - 将含蓄/省略的表达补全得更生动（日文标题常省略主语、用语气词暗示）
   - 用更符合中文习惯的表达方式重组语序
   - 适度使用感叹号、问号、省略号等其他符号增加情感张力

2. **B站标题风格参考**（什么样的标题在B站有点击率）：
   - 带情感共鸣：用感叹、疑问、悬念引发好奇（"居然...！" "到底能不能...？"）
   - 带具体信息：游戏名、事件描述要具体，不要笼统
   - 带节目感：系列内容保留集数/话数标记（如 #1、第2话）
   - 避免纯直译的干巴巴感："杂谈配信" → 不如 "杂谈！和大家聊聊天~"
   - 避免过度夸张：不要加原文没有的"震惊""爆笑"等词

3. **不要添加主播名前缀**：主播名会在后处理自动添加为"【白上吹雪】$你的翻译"格式，因此翻译结果中不需要保留频道名前缀，或者通用标识（如【hololive】等可能在该主播的所有视频中都出现的）

4. **保留有意义的标记**：除频道名以外的方括号标记（如【3DLIVE】【APEX】【#话题标签】【少儿不宜】等）应当保留并翻译

5. **完整翻译**：日文/英文内容优先使用约定俗成译名（Minecraft→我的世界），无法确定的用音译或原文

6. **参考示例**（注意体会翻译风格）：
   - 「【白上フブキ】雑談配信！みんなとおしゃべり」
     → `杂谈直播！想和大家聊聊天~`（去掉频道名前缀，语气自然亲切）
   - 「【初放送】フブキCh。(^・ω・^§)ﾉ　白上フブキのみんなのお耳にちょコンっと放送！」
     → `【初次直播】在大家的耳边悄悄放送！(^・ω・^§)ﾉ`（保留颜文字，去掉频道名）
   - 「【子どもは見ちゃダメ】あのR指定映画を見てみよう【#デッドプールウルヴァリン】」
     → `【少儿不宜】来看看那部R级电影吧【#死侍与金刚狼】`（保留内容标记）
   - 「【3DLIVE】みかんのうた/白上フブキ&周防帕特拉　(cover)」
     → `【3DLIVE】橘子之歌/白上吹雪&周防パトラ (Cover)`（保留3DLIVE标记）
   - 「【APEX LEGENDS】３人でちゃんぽん食べたいねーっていう願望がですね？」
     → `【APEX】三个人一起吃鸡的愿望能实现吗！？`（APEX简写，意译更生动）
   - 「Getting Over It やっていくぅ！！」
     → `挑战掘地求升！！冲冲冲！`（补全语气，保留热情感）
   - 「＃５ マホロアエピローグ 異空をかける旅人|星のカービィ Wii デラックス」
     → `#5 玛霍洛亚篇章 穿越异空的旅人｜星之卡比 Wii 豪华版`（保留集数标记）
    
### 简介翻译规则（重要！）

1. **description_summary**（1-2句话的摘要）：
   - 简洁概括视频内容，语气可爱俏皮

2. **description_translated**（完整翻译）：
   - 翻译原描述的**核心内容**
   - **以下内容直接省略，不要翻译也不要说明**：
     - BGM/音乐信息、版权声明
     - 外部链接、社交媒体引流
     - 商业推广/广告/赞助信息
     - 会员/周边商品信息
     - 系列视频列表
   - **不要写类似这种说明**："原文含有更多链接与商品信息，这里省略外链"、"以下为BGM信息..."
   - 如果原简介几乎全是上述无用内容，description_translated 可以留空或只写一句话

### 标签和分区
- tags_translated: 翻译有意义的原标签
- tags_generated: 补充3-5个B站常用标签
- 根据内容推荐合适的B站分区

### B站分区参考
{zones_info}

## JSON输出（严格按此格式）
```json
{{
  "title_translated": "翻译后的标题（不含主播名前缀）",
  "description_summary": "1-2句话的简短摘要",
  "description_translated": "完整翻译的简介内容（省略无用信息，不要写省略说明）",
  "tags_translated": [],
  "tags_generated": [],
  "recommended_tid": 0,
  "recommended_tid_name": "",
  "tid_reason": ""
}}
```
'''


class VideoInfoTranslator:
    """
    视频信息翻译器
    
    负责将YouTube视频的标题、描述、标签翻译为中文，
    并优化为适合B站的内容格式，同时推荐合适的分区。
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = "", base_url: str = "", proxy: str = ""):
        """
        初始化翻译器
        
        Args:
            model: 使用的LLM模型
            api_key: API Key 覆写（空=使用全局配置）
            base_url: Base URL 覆写（空=使用全局配置）
            proxy: 代理地址覆写（空=使用环境变量）
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.proxy = proxy
        self.client = None
    
    def _get_client(self):
        """获取LLM客户端（延迟初始化，支持 per-stage credentials）"""
        if self.client is None:
            self.client = get_or_create_client(self.api_key, self.base_url, self.proxy)
        return self.client
    
    def _build_zones_info(self) -> str:
        """构建分区信息字符串（按大区分组）"""
        lines = []
        # 按大区分组显示
        for main_tid, main_name in sorted(BILIBILI_MAIN_ZONES.items()):
            sub_zones = [(tid, info[0]) for tid, info in BILIBILI_ZONES.items() if info[1] == main_tid]
            if sub_zones:
                lines.append(f"\n【{main_name}区】")
                for tid, name in sorted(sub_zones):
                    lines.append(f"  - {tid}: {name}")
        return "\n".join(lines)
    
    def translate(
        self,
        title: str,
        description: str,
        tags: List[str],
        uploader: str = "",
        default_tid: int = 21
    ) -> TranslatedVideoInfo:
        """
        翻译视频信息
        
        Args:
            title: 原始标题
            description: 原始描述
            tags: 原始标签列表
            uploader: 主播/频道名
            default_tid: 默认分区ID（LLM失败时使用）
            
        Returns:
            TranslatedVideoInfo: 翻译后的视频信息
        """
        logger.info(f"开始翻译视频信息: {title[:50]}...")
        
        # 关键字段校验：信息缺失会导致 LLM 翻译不准确
        if not title or not title.strip():
            raise ValueError("翻译失败：title 为空，无法进行有意义的翻译")
        if not uploader or not uploader.strip():
            logger.warning("uploader 为空，LLM 将缺少频道/主播上下文，翻译质量可能下降")
        
        # 构建prompt
        tags_str = ", ".join(tags) if tags else "无"
        prompt = TRANSLATE_VIDEO_INFO_PROMPT.format(
            title=title,
            description=description[:2000] if description else "无",  # 限制描述长度
            tags=tags_str,
            uploader=uploader or "未知",
            zones_info=self._build_zones_info()
        )
        
        # 重试机制：JSON解析错误或网络问题时重试
        max_retries = 3
        last_error = None
        content = None
        
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的视频内容翻译专家。请严格按JSON格式输出。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                
                content = response.choices[0].message.content.strip()
                
                # 提取JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                
                # 构建返回对象（兼容新旧字段名）
                description_summary = result.get('description_summary') or result.get('description_optimized', '')
                
                # 校验 LLM 返回的关键字段——这些缺失意味着翻译无效
                if 'title_translated' not in result or not result['title_translated'].strip():
                    raise ValueError(f"LLM 返回的 JSON 中 title_translated 缺失或为空: {list(result.keys())}")
                
                # 后处理：去除 LLM 可能添加的主播名前缀（我们会在模板中添加）
                title_translated = result['title_translated']
                title_translated = self._strip_uploader_prefix(title_translated, uploader)
                
                return TranslatedVideoInfo(
                    original_title=title,
                    original_description=description,
                    original_tags=tags,
                    title_translated=title_translated,
                    description_summary=description_summary,
                    description_translated=result.get('description_translated', ''),
                    tags_translated=result.get('tags_translated', []),
                    tags_generated=result.get('tags_generated', []),
                    recommended_tid=result.get('recommended_tid', default_tid),
                    recommended_tid_name=result.get('recommended_tid_name', get_zone_name(default_tid)),
                    tid_reason=result.get('tid_reason', "默认分区")
                )
                
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"翻译JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 * (attempt + 1))  # 简单退避
                continue
            except Exception as e:
                last_error = e
                # 网络错误等可重试
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    logger.warning(f"翻译网络错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 * (attempt + 1))
                    continue
                else:
                    # 其他错误不重试
                    logger.error(f"视频信息翻译失败: {e}")
                    break
        
        # 所有重试失败——抛出异常，不静默降级
        # 返回未翻译的日文标题到B站是不可接受的
        error_msg = f"视频信息翻译最终失败 (已重试 {max_retries} 次): {last_error}"
        logger.error(error_msg)
        if content:
            logger.debug(f"最后响应: {content[:500]}")
        raise RuntimeError(error_msg)
    
    def _strip_uploader_prefix(self, title: str, uploader: str) -> str:
        """
        去除标题中的主播名前缀
        
        LLM 有时会在标题前添加 [主播名] 前缀，但我们会在模板中添加，
        所以需要后处理去除重复的前缀。
        
        常见格式：
        - [主播名] 标题
        - 【主播名】标题
        - [主播名]标题
        """
        if not uploader or not title:
            return title
        
        # 去除常见的前缀格式
        # 匹配 [xxx] 或 【xxx】 开头的模式
        prefix_patterns = [
            rf'^\s*\[{re.escape(uploader)}\]\s*',      # [主播名] 
            rf'^\s*【{re.escape(uploader)}】\s*',      # 【主播名】
            rf'^\s*\[[^\]]*{re.escape(uploader)}[^\]]*\]\s*',  # [xxx主播名xxx]
            rf'^\s*【[^】]*{re.escape(uploader)}[^】]*】\s*',  # 【xxx主播名xxx】
        ]
        
        for pattern in prefix_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # 注意：不做通用的 [xxx]/【xxx】 剥离！
        # 标题中的 【3DLIVE】【少儿不宜】【APEX】 等是有意义的内容标记，必须保留。
        # 只剥离明确匹配到 uploader 名字的前缀。
        
        return title.strip()
    
    # _fallback_translate 已移除：翻译失败时应抛出异常，
    # 而非静默返回未翻译的原始日文内容（发到B站后用户看到日文标题是不可接受的）。
    # 调用方（executor/playlist_service）负责 catch 异常并决定是否跳过该视频。
