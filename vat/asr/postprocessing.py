"""
ASR 后处理模块：幻觉检测、重复清理、日语特殊处理

借鉴自 WhisperJAV 项目，针对 VTB 直播场景优化
"""
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

from vat.utils.logger import setup_logger

logger = setup_logger("asr_postprocessing")


# ============================================================================
# 幻觉检测常量
# ============================================================================

# 日语常见幻觉文本（完整行匹配）
JAPANESE_HALLUCINATION_EXACT: Set[str] = {
    # 常见误识别
    'www', 'wwww', 'wwwww', 'ｗｗｗ', 'ｗｗｗｗ',
    'ok', 'OK', 'Ok',
    '笑', '（笑）', '(笑)',
    'w', 'ｗ',
    # 结束语幻觉
    'ご視聴ありがとうございました',
    'ご視聴ありがとうございます', 
    'ありがとうございました',
    'チャンネル登録お願いします',
    'チャンネル登録よろしくお願いします',
    'グッドボタンお願いします',
    '高評価お願いします',
    # 字幕相关幻觉
    '字幕',
    '字幕：',
    '翻訳：',
    '編集：',
    # 音效描述（通常是误识别）
    '♪',
    '♪♪',
    '♪♪♪',
    '🎵',
    # 空白/无意义
    '...',
    '…',
    '。。。',
    '、、、',
}

# 幻觉正则模式
HALLUCINATION_REGEX_PATTERNS: List[Tuple[str, str, float]] = [
    # (pattern, category, confidence)
    (r'^(OK|ok|Ok)+$', 'common_hallucination', 1.0),
    (r'^[wWｗＷ]+$', 'common_hallucination', 1.0),
    (r'^笑+$', 'common_hallucination', 1.0),
    (r'^(ご|お)?視聴.*ありがとう.*$', 'closing_phrase', 0.95),
    (r'^チャンネル登録.*$', 'closing_phrase', 0.95),
    (r'^[♪🎵🎶]+$', 'music_symbol', 0.9),
    (r'^[\.\…。、]+$', 'punctuation_only', 1.0),
    # 括号包裹的描述性文本
    (r'^[\(（\[【「『《].*[\)）\]】」』》]$', 'bracketed_context', 0.8),
]

# 日语重复模式清理
REPETITION_PATTERNS: List[Tuple[str, str, str]] = [
    # (name, pattern, replacement)
    # 极端短语重复（带分隔符）：あ!!あ!!あ!! -> あ!!
    ('phrase_with_separator', r'((?:[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]{1,8}[、,!\s！！？？。。・]+))\1{3,}', r'\1'),
    # 多字符词重复：ハッハッハッハッ -> ハッハッ
    ('multi_char_word', r'((?:[\u3040-\u309f\u30a0-\u30ff]{2,4}))\1{3,}', r'\1\1'),
    # 逗号短语重复：ゆーちゃん、ゆーちゃん、ゆーちゃん -> ゆーちゃん、
    ('phrase_with_comma', r'((?:[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]{1,10}[、,]\s*))\1{2,}', r'\1'),
    # 单字符洪水：ううううう -> うう
    ('single_char_flood', r'([\u3040-\u309f\u30a0-\u30ff])\1{3,}', r'\1\1'),
    # 前缀+字符重复：あらららら -> あらら
    ('prefix_plus_char', r'([\u3040-\u309f\u30a0-\u30ff]{1,2})([\u3040-\u309f\u30a0-\u30ff])\2{3,}', r'\1\2\2'),
    # 元音延长：あ〜〜〜〜 -> あ〜〜
    ('vowel_extension', r'([\u3040-\u309f\u30a0-\u30ff])([〜ー])\2{3,}', r'\1\2\2'),
]

# 日语句尾助词（用于断句优化）
JAPANESE_SENTENCE_ENDINGS = {
    'ね', 'よ', 'わ', 'の', 'さ', 'な', 'ぞ', 'ぜ', 'かな', 'かね',
    'よね', 'わね', 'のね', 'だね', 'ですね', 'ますね',
    'よな', 'だな', 'かな', 'のかな',
}

# 日语相槌（不应被删除的短回应）
JAPANESE_AIZUCHI = {
    'うん', 'ううん', 'はい', 'ええ', 'あー', 'えー', 'おー',
    'そう', 'そうそう', 'なるほど', 'へー', 'ほー', 'ふーん',
    'まあ', 'まあまあ', 'やっぱり', 'やっぱ',
    'ちょっと', 'えっと', 'あのー', 'えーっと',
}


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class PostProcessingResult:
    """后处理结果"""
    original_text: str
    processed_text: str
    is_hallucination: bool = False
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def was_modified(self) -> bool:
        return self.original_text != self.processed_text


@dataclass
class PostProcessingStats:
    """后处理统计"""
    total_segments: int = 0
    hallucinations_removed: int = 0
    repetitions_cleaned: int = 0
    empty_removed: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'total_segments': self.total_segments,
            'hallucinations_removed': self.hallucinations_removed,
            'repetitions_cleaned': self.repetitions_cleaned,
            'empty_removed': self.empty_removed,
        }


# ============================================================================
# 幻觉检测器
# ============================================================================

class HallucinationDetector:
    """
    幻觉检测器：识别并移除 Whisper 常见的幻觉输出
    
    幻觉类型：
    1. 完整行匹配（如 "www", "ご視聴ありがとうございました"）
    2. 正则模式匹配（如重复的标点、括号包裹的描述）
    3. 高置信度过滤（基于文本特征）
    """
    
    def __init__(
        self,
        exact_matches: Optional[Set[str]] = None,
        regex_patterns: Optional[List[Tuple[str, str, float]]] = None,
        min_confidence: float = 0.8,
        custom_blacklist: Optional[List[str]] = None,
    ):
        """
        初始化幻觉检测器
        
        Args:
            exact_matches: 精确匹配的幻觉文本集合
            regex_patterns: 正则模式列表 [(pattern, category, confidence), ...]
            min_confidence: 最小置信度阈值
            custom_blacklist: 用户自定义黑名单
        """
        self.exact_matches = exact_matches or JAPANESE_HALLUCINATION_EXACT
        self.regex_patterns = regex_patterns or HALLUCINATION_REGEX_PATTERNS
        self.min_confidence = min_confidence
        self.custom_blacklist = set(custom_blacklist) if custom_blacklist else set()
        
        # 合并自定义黑名单
        self.exact_matches = self.exact_matches | self.custom_blacklist
        
        # 预编译正则表达式
        self._compiled_patterns = [
            (re.compile(pattern), category, confidence)
            for pattern, category, confidence in self.regex_patterns
        ]
        
        logger.debug(f"HallucinationDetector 初始化: {len(self.exact_matches)} 精确匹配, "
                     f"{len(self._compiled_patterns)} 正则模式")
    
    def detect(self, text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        检测文本是否为幻觉
        
        Args:
            text: 待检测文本
            
        Returns:
            (is_hallucination, match_info)
        """
        if not text or not text.strip():
            return True, {'type': 'empty', 'confidence': 1.0}
        
        normalized = text.strip().lower()
        
        # 1. 精确匹配
        if normalized in self.exact_matches or text.strip() in self.exact_matches:
            return True, {
                'type': 'exact_match',
                'pattern': normalized,
                'confidence': 1.0,
                'category': 'hallucination',
            }
        
        # 2. 正则匹配
        for compiled_pattern, category, confidence in self._compiled_patterns:
            if confidence < self.min_confidence:
                continue
            if compiled_pattern.match(text.strip()):
                return True, {
                    'type': 'regex_match',
                    'pattern': compiled_pattern.pattern,
                    'confidence': confidence,
                    'category': category,
                }
        
        # 3. 括号包裹检测（描述性文本）
        bracket_info = self._check_bracketed(text.strip())
        if bracket_info:
            return True, bracket_info
        
        return False, None
    
    def _check_bracketed(self, text: str) -> Optional[Dict[str, Any]]:
        """检查是否为括号包裹的描述性文本"""
        bracket_pairs = [
            ('(', ')'), ('（', '）'),
            ('[', ']'), ('［', '］'),
            ('{', '}'), ('｛', '｝'),
            ('【', '】'), ('『', '』'),
            ('「', '」'), ('《', '》'),
        ]
        
        for left, right in bracket_pairs:
            if text.startswith(left) and text.endswith(right):
                inner = text[len(left):-len(right)].strip()
                # 如果内部是描述性文本（如 "拍手"、"笑声" 等）
                if inner and len(inner) <= 10:
                    return {
                        'type': 'bracketed_context',
                        'pattern': f'{left}...{right}',
                        'confidence': 0.85,
                        'category': 'context_caption',
                        'inner_text': inner,
                    }
        return None
    
    def is_valid_japanese_content(self, text: str) -> bool:
        """
        检查文本是否为有效的日语内容（避免误删）
        
        保护规则：
        - 包含汉字+假名混合
        - 包含常见句式结构
        - 是相槌（短回应）
        """
        text = text.strip()
        
        # 相槌保护
        if text in JAPANESE_AIZUCHI:
            return True
        
        # 检查是否包含日语特征
        has_hiragana = bool(re.search(r'[\u3040-\u309f]', text))
        has_katakana = bool(re.search(r'[\u30a0-\u30ff]', text))
        has_kanji = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        # 混合脚本通常是有效内容
        script_count = sum([has_hiragana, has_katakana, has_kanji])
        if script_count >= 2:
            return True
        
        # 包含常见日语语法结构
        grammar_markers = ['です', 'ます', 'だ', 'である', 'でした', 'ました', 
                          'いる', 'ある', 'する', 'した', 'って', 'という']
        if any(marker in text for marker in grammar_markers):
            return True
        
        # 包含数字或货币
        if re.search(r'[\d¥$€£円]', text):
            return True
        
        return False


# ============================================================================
# 重复清理器
# ============================================================================

class RepetitionCleaner:
    """
    重复清理器：清理 Whisper 输出中的异常重复
    
    处理模式：
    - 字符洪水（如 "うううう"）
    - 短语重复（如 "ハッハッハッ"）
    - 标点重复
    """
    
    def __init__(
        self,
        patterns: Optional[List[Tuple[str, str, str]]] = None,
        threshold: int = 2,  # 保留的最大重复次数
    ):
        """
        初始化重复清理器
        
        Args:
            patterns: 清理模式列表 [(name, pattern, replacement), ...]
            threshold: 重复阈值
        """
        self.patterns = patterns or REPETITION_PATTERNS
        self.threshold = threshold
        
        # 预编译正则表达式
        self._compiled_patterns = [
            (name, re.compile(pattern), replacement)
            for name, pattern, replacement in self.patterns
        ]
        
        logger.debug(f"RepetitionCleaner 初始化: {len(self._compiled_patterns)} 清理模式")
    
    def clean(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        清理文本中的重复
        
        Args:
            text: 待清理文本
            
        Returns:
            (cleaned_text, modifications)
        """
        if not text or not text.strip():
            return text, []
        
        modifications = []
        current_text = text
        
        for name, compiled_pattern, replacement in self._compiled_patterns:
            try:
                original = current_text
                new_text = compiled_pattern.sub(replacement, current_text)
                
                if new_text != original:
                    modifications.append({
                        'type': name,
                        'pattern': compiled_pattern.pattern,
                        'original': original,
                        'modified': new_text,
                        'category': 'repetition_cleaning',
                    })
                    current_text = new_text
                    
            except Exception as e:
                logger.warning(f"重复清理模式 '{name}' 处理失败: {e}")
                continue
        
        return current_text.strip(), modifications
    
    def is_all_repetition(self, text: str) -> bool:
        """检查文本是否几乎全是重复"""
        stripped = re.sub(r'[\s\u3000]', '', text)  # 移除空白
        if len(stripped) < 10:
            return False
        
        # 检查单字符占比
        char_counts = Counter(stripped)
        if char_counts:
            most_common_char, count = char_counts.most_common(1)[0]
            if count / len(stripped) > 0.8:
                return True
        
        return False


# ============================================================================
# 日语后处理器
# ============================================================================

class JapanesePostProcessor:
    """
    日语后处理器：针对日语特性的优化处理
    
    功能：
    - 句尾助词处理
    - 相槌识别
    - 方言适配
    - VTB 用语处理
    """
    
    # VTB 常见用语（不应被删除或修改）
    VTB_TERMS = {
        'スパチャ', 'スーパーチャット', 'メンバーシップ', 'メン限',
        'コラボ', 'ゲリラ', '枠', '配信', '生放送',
        'コメント', 'チャット', 'リスナー', '視聴者',
        'グッズ', 'オリ曲', 'カバー', '歌枠', '雑談',
        'あくたん', 'そらちゃん', 'ころね', 'ぺこら',  # 常见昵称后缀
    }
    
    def __init__(self):
        self.sentence_endings = JAPANESE_SENTENCE_ENDINGS
        self.aizuchi = JAPANESE_AIZUCHI
        
        logger.debug("JapanesePostProcessor 初始化")
    
    def process(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        处理日语文本
        
        Args:
            text: 待处理文本
            
        Returns:
            (processed_text, modifications)
        """
        if not text or not text.strip():
            return text, []
        
        modifications = []
        current_text = text
        
        # 1. 全角/半角标准化
        normalized = self._normalize_punctuation(current_text)
        if normalized != current_text:
            modifications.append({
                'type': 'punctuation_normalization',
                'original': current_text,
                'modified': normalized,
            })
            current_text = normalized
        
        # 2. 清理多余空格
        cleaned = self._clean_whitespace(current_text)
        if cleaned != current_text:
            modifications.append({
                'type': 'whitespace_cleanup',
                'original': current_text,
                'modified': cleaned,
            })
            current_text = cleaned
        
        return current_text, modifications
    
    def _normalize_punctuation(self, text: str) -> str:
        """标准化标点符号"""
        # 日语常用全角标点
        replacements = [
            ('!', '！'),
            ('?', '？'),
            (',', '、'),
            # 保留句号的原样（可能是全角或半角）
        ]
        
        result = text
        for old, new in replacements:
            result = result.replace(old, new)
        
        return result
    
    def _clean_whitespace(self, text: str) -> str:
        """清理多余空白
        
        日语文本不使用空格分词，ASR 有时会在每个词/字之间插入空格，
        例如 "微熱 に なっ て き た ん だ けど"，这些空格是无意义的。
        
        策略：
        - CJK 字符之间的空格：直接移除
        - CJK 与 ASCII 之间的空格：移除（日语字幕中不需要）
        - ASCII 单词之间的空格：保留（如 "CLIP STUDIO PAINT"）
        """
        if not text:
            return text
        
        # 判断是否为 CJK 为主的文本
        cjk_count = len(re.findall(r'[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]', text))
        total_alpha = len(re.findall(r'\S', text))
        
        if total_alpha > 0 and cjk_count / total_alpha >= 0.3:
            # CJK 为主的文本：移除 CJK 字符周围的空格
            # 1. 移除两个 CJK/kana 字符之间的空格
            result = re.sub(
                r'(?<=[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef])'
                r'[\s]+'
                r'(?=[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef])',
                '', text
            )
            # 2. 移除 CJK 与 ASCII 之间的空格
            result = re.sub(
                r'(?<=[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef])[\s]+(?=[A-Za-z0-9])',
                '', result
            )
            result = re.sub(
                r'(?<=[A-Za-z0-9])[\s]+(?=[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef])',
                '', result
            )
        else:
            # 非 CJK 文本：只合并连续空格
            result = re.sub(r'[ \t]+', ' ', text)
        
        return result.strip()
    
    def is_aizuchi(self, text: str) -> bool:
        """检查是否为相槌"""
        return text.strip() in self.aizuchi
    
    def has_sentence_ending(self, text: str) -> bool:
        """检查是否有句尾助词"""
        text = text.strip()
        for ending in self.sentence_endings:
            if text.endswith(ending):
                return True
        return False


# ============================================================================
# 综合后处理器
# ============================================================================

class ASRPostProcessor:
    """
    ASR 综合后处理器
    
    整合幻觉检测、重复清理、日语处理的完整流程
    """
    
    def __init__(
        self,
        enable_hallucination_detection: bool = True,
        enable_repetition_cleaning: bool = True,
        enable_japanese_processing: bool = True,
        custom_blacklist: Optional[List[str]] = None,
        min_confidence: float = 0.8,
    ):
        """
        初始化综合后处理器
        
        Args:
            enable_hallucination_detection: 启用幻觉检测
            enable_repetition_cleaning: 启用重复清理
            enable_japanese_processing: 启用日语处理
            custom_blacklist: 自定义幻觉黑名单
            min_confidence: 幻觉检测最小置信度
        """
        self.enable_hallucination = enable_hallucination_detection
        self.enable_repetition = enable_repetition_cleaning
        self.enable_japanese = enable_japanese_processing
        
        # 初始化子处理器
        if self.enable_hallucination:
            self.hallucination_detector = HallucinationDetector(
                custom_blacklist=custom_blacklist,
                min_confidence=min_confidence,
            )
        else:
            self.hallucination_detector = None
        
        if self.enable_repetition:
            self.repetition_cleaner = RepetitionCleaner()
        else:
            self.repetition_cleaner = None
        
        if self.enable_japanese:
            self.japanese_processor = JapanesePostProcessor()
        else:
            self.japanese_processor = None
        
        self.stats = PostProcessingStats()
        
        logger.info(f"ASRPostProcessor 初始化: hallucination={self.enable_hallucination}, "
                    f"repetition={self.enable_repetition}, japanese={self.enable_japanese}")
    
    def process_text(self, text: str) -> PostProcessingResult:
        """
        处理单个文本
        
        Args:
            text: 待处理文本
            
        Returns:
            PostProcessingResult
        """
        self.stats.total_segments += 1
        
        if not text or not text.strip():
            self.stats.empty_removed += 1
            return PostProcessingResult(
                original_text=text,
                processed_text='',
                is_hallucination=True,
                modifications=[{'type': 'empty_text'}],
            )
        
        modifications = []
        current_text = text
        is_hallucination = False
        
        # 1. 幻觉检测
        if self.hallucination_detector:
            is_hall, hall_info = self.hallucination_detector.detect(current_text)
            
            if is_hall:
                # 二次验证：确保不是有效日语内容
                if self.japanese_processor and self.hallucination_detector.is_valid_japanese_content(current_text):
                    logger.debug(f"幻觉检测跳过（有效日语内容）: {current_text[:30]}...")
                else:
                    is_hallucination = True
                    self.stats.hallucinations_removed += 1
                    modifications.append(hall_info)
                    return PostProcessingResult(
                        original_text=text,
                        processed_text='',
                        is_hallucination=True,
                        modifications=modifications,
                    )
        
        # 2. 重复清理
        if self.repetition_cleaner:
            # 先检测原始文本是否几乎全是重复（如大量 "うううう"）
            if self.repetition_cleaner.is_all_repetition(current_text):
                self.stats.hallucinations_removed += 1
                modifications.append({
                    'type': 'all_repetition',
                    'original': current_text[:50] + '...' if len(current_text) > 50 else current_text,
                    'category': 'repetition_hallucination',
                })
                return PostProcessingResult(
                    original_text=text,
                    processed_text='',
                    is_hallucination=True,
                    modifications=modifications,
                )
            
            cleaned, rep_mods = self.repetition_cleaner.clean(current_text)
            if rep_mods:
                modifications.extend(rep_mods)
                current_text = cleaned
                self.stats.repetitions_cleaned += 1
            
            # 清理后如果文本太短（原本很长但清理后几乎没了），也视为幻觉
            if len(text) > 20 and len(current_text.strip()) < 5:
                self.stats.hallucinations_removed += 1
                modifications.append({
                    'type': 'cleaned_to_empty',
                    'original_len': len(text),
                    'cleaned_len': len(current_text),
                })
                return PostProcessingResult(
                    original_text=text,
                    processed_text='',
                    is_hallucination=True,
                    modifications=modifications,
                )
        
        # 3. 日语处理
        if self.japanese_processor:
            processed, jp_mods = self.japanese_processor.process(current_text)
            if jp_mods:
                modifications.extend(jp_mods)
                current_text = processed
        
        return PostProcessingResult(
            original_text=text,
            processed_text=current_text,
            is_hallucination=False,
            modifications=modifications,
        )
    
    def process_segments(self, segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], PostProcessingStats]:
        """
        批量处理字幕段
        
        Args:
            segments: 字幕段列表，每项包含 'text' 字段
            
        Returns:
            (processed_segments, stats)
        """
        processed = []
        
        for seg in segments:
            text = seg.get('text', '')
            result = self.process_text(text)
            
            if not result.is_hallucination and result.processed_text:
                new_seg = seg.copy()
                new_seg['text'] = result.processed_text
                if result.was_modified:
                    new_seg['_original_text'] = result.original_text
                    new_seg['_modifications'] = result.modifications
                processed.append(new_seg)
        
        logger.info(f"后处理完成: 输入 {len(segments)} 段, 输出 {len(processed)} 段, "
                    f"移除幻觉 {self.stats.hallucinations_removed}, "
                    f"清理重复 {self.stats.repetitions_cleaned}")
        
        return processed, self.stats
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """重置统计"""
        self.stats = PostProcessingStats()


# ============================================================================
# 便捷函数
# ============================================================================

def postprocess_asr_text(
    text: str,
    enable_hallucination: bool = True,
    enable_repetition: bool = True,
    enable_japanese: bool = True,
) -> str:
    """
    便捷函数：后处理 ASR 文本
    
    Args:
        text: 待处理文本
        enable_hallucination: 启用幻觉检测
        enable_repetition: 启用重复清理
        enable_japanese: 启用日语处理
        
    Returns:
        处理后的文本（如果是幻觉则返回空字符串）
    """
    processor = ASRPostProcessor(
        enable_hallucination_detection=enable_hallucination,
        enable_repetition_cleaning=enable_repetition,
        enable_japanese_processing=enable_japanese,
    )
    result = processor.process_text(text)
    return result.processed_text


def is_hallucination(text: str) -> bool:
    """
    便捷函数：检查文本是否为幻觉
    
    Args:
        text: 待检测文本
        
    Returns:
        是否为幻觉
    """
    detector = HallucinationDetector()
    is_hall, _ = detector.detect(text)
    return is_hall
