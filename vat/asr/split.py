import difflib
import re
from typing import List, Tuple

from vat.llm import call_llm
from vat.llm.prompts import get_prompt
from vat.utils.logger import setup_logger
from vat.utils.text_utils import count_words, is_mainly_cjk

logger = setup_logger("split_by_llm")

MAX_WORD_COUNT = 20  # 英文单词或中文字符的最大数量
MAX_STEPS = 3  # Agent loop最大尝试次数

# 模型升级列表：当前模型失败后尝试下一个更强的模型
MODEL_UPGRADE_CHAIN = [
    "gpt-4o-mini",
    "gpt-5-nano", 
    "gpt-5-mini",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gpt-5.2-chat",
]


def _get_next_model(current_model: str) -> str | None:
    """获取升级后的模型，如果已是最强模型则返回 None"""
    try:
        idx = MODEL_UPGRADE_CHAIN.index(current_model)
        if idx + 1 < len(MODEL_UPGRADE_CHAIN):
            return MODEL_UPGRADE_CHAIN[idx + 1]
    except ValueError:
        # 当前模型不在升级链中，不升级
        pass
    return None


def split_by_llm(
    text: str,
    model: str = "gpt-4o-mini",
    max_word_count_cjk: int = 18,
    max_word_count_english: int = 12,
    min_word_count_cjk: int = 4,
    min_word_count_english: int = 2,
    recommend_word_count_cjk: int = 12,
    recommend_word_count_english: int = 8,
    scene_prompt: str = "",
    mode: str = "sentence",
    allow_model_upgrade: bool = False,
) -> List[str]:
    """使用LLM进行文本断句

    Args:
        text: 待断句的文本
        model: LLM模型名称
        max_word_count_cjk: 中文最大字符数（硬性限制）
        max_word_count_english: 英文最大单词数（硬性限制）
        min_word_count_cjk: 中文最小字符数（软性建议）
        min_word_count_english: 英文最小单词数（软性建议）
        recommend_word_count_cjk: 中文推荐字符数（软性建议，理想长度）
        recommend_word_count_english: 英文推荐单词数（软性建议，理想长度）
        scene_prompt: 场景特定提示词，会插入到 system prompt 中（可选）
        mode: 断句模式，"sentence"（句子级）或 "semantic"（语义级）
        allow_model_upgrade: 是否允许在失败时升级到更强模型

    Returns:
        断句后的文本列表
    """
    assert text and isinstance(text, str), "调用契约错误: text 必须是非空字符串"
    assert mode in ("sentence", "semantic"), f"调用契约错误: mode 必须是 'sentence' 或 'semantic'，得到 '{mode}'"
    
    current_model = model
    
    while True:
        try:
            result, success = _split_with_agent_loop(
                text, current_model, max_word_count_cjk, max_word_count_english,
                min_word_count_cjk, min_word_count_english, 
                recommend_word_count_cjk, recommend_word_count_english,
                scene_prompt, mode
            )
            
            if success:
                assert result, "逻辑错误: _split_with_agent_loop 返回了空列表"
                return result
            
            # 失败，尝试升级模型
            if allow_model_upgrade:
                next_model = _get_next_model(current_model)
                if next_model:
                    logger.warning(f"模型 {current_model} 断句失败，升级到 {next_model}")
                    current_model = next_model
                    continue
            
            # 无法升级或不允许升级，返回最后结果
            logger.warning(f"断句未通过验证，但无法继续升级模型，返回最后结果")
            return result if result else [text]
            
        except Exception as e:
            logger.error(f"断句失败: {e}")
            return [text]


def _split_with_agent_loop(
    text: str,
    model: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    min_word_count_cjk: int,
    min_word_count_english: int,
    recommend_word_count_cjk: int = 12,
    recommend_word_count_english: int = 8,
    scene_prompt: str = "",
    mode: str = "sentence",
) -> Tuple[List[str], bool]:
    """使用agent loop 建立反馈循环进行文本断句，自动验证和修正
    
    Returns:
        (result, success): 断句结果和是否成功
    """
    assert text, "调用契约错误: text 不能为空"
    prompt_path = f"split/{mode}"
    system_prompt = get_prompt(
        prompt_path,
        max_word_count_cjk=max_word_count_cjk,
        max_word_count_english=max_word_count_english,
        min_word_count_cjk=min_word_count_cjk,
        min_word_count_english=min_word_count_english,
        recommend_word_count_cjk=recommend_word_count_cjk,
        recommend_word_count_english=recommend_word_count_english,
    )
    
    # 插入场景特定提示词（如果有）
    if scene_prompt:
        # 在 </instructions> 之后插入场景提示词
        insert_marker = "</instructions>"
        if insert_marker in system_prompt:
            scene_block = f"\n\n<scene_specific>\n{scene_prompt.strip()}\n</scene_specific>"
            system_prompt = system_prompt.replace(
                insert_marker, 
                insert_marker + scene_block
            )
        else:
            # 如果没有找到标记，追加到末尾
            system_prompt = f"{system_prompt}\n\n<scene_specific>\n{scene_prompt.strip()}\n</scene_specific>"

    user_prompt = (
        f"Please use multiple <br> tags to separate the following sentence:\n{text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_result = None

    for step in range(MAX_STEPS):
        response = call_llm(
            messages=messages,
            model=model,
            temperature=0.1,
        )

        if not response or not response.choices:
            logger.error("LLM 返回响应为空")
            continue

        result_text = response.choices[0].message.content
        if not result_text:
            logger.error("LLM 返回内容为空")
            continue

        # 解析结果
        result_text_cleaned = re.sub(r"\n+", "", result_text)
        split_result = [
            segment.strip()
            for segment in result_text_cleaned.split("<br>")
            if segment.strip()
        ]
        
        if not split_result:
            logger.warning(f"解析结果为空: {result_text}")
            continue
            
        last_result = split_result

        # 验证结果
        is_valid, error_message = _validate_split_result(
            original_text=text,
            split_result=split_result,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            min_word_count_cjk=min_word_count_cjk,
            min_word_count_english=min_word_count_english,
        )

        if is_valid:
            # 最终一致性检查：合并后应该与原文本高度相似
            text_is_cjk = is_mainly_cjk(text)
            merged = ("" if text_is_cjk else " ").join(split_result)
            # 允许标点符号和空白字符的微小差异
            assert len(merged) >= len(text) * 0.8, f"逻辑错误: 断句后文本严重丢失 (原长 {len(text)}, 现长 {len(merged)})"
            return split_result, True

        # 添加反馈到对话
        logger.warning(
            f"模型输出错误，断句验证失败，频繁出现建议更换更智能的模型。开始反馈循环 (第{step + 1}次尝试):\n {error_message}\n\n"
        )
        messages.append({"role": "assistant", "content": result_text})
        messages.append(
            {
                "role": "user",
                "content": f"Error: {error_message}\nFix the errors above and output the COMPLETE corrected text with <br> tags (include ALL segments, not just the fixed ones), no explanation.",
            }
        )

    # MAX_STEPS 次尝试后仍未通过验证
    return last_result if last_result else [text], False


def _validate_split_result(
    original_text: str,
    split_result: List[str],
    max_word_count_cjk: int,
    max_word_count_english: int,
    min_word_count_cjk: int = 4,
    min_word_count_english: int = 2,
) -> Tuple[bool, str]:
    """验证断句结果：内容一致性、分段数量、长度限制

    返回: (是否有效, 错误反馈)
    
    注意: 最小长度违规只记录debug日志，不触发反馈循环（因为修正困难）
    """
    # 检查是否为空
    if not split_result:
        return False, "No segments found. Split the text with <br> tags."

    # 检查内容是否被修改
    # 规范化：移除所有空白字符，只比较内容字符
    def normalize(s: str) -> str:
        return re.sub(r"\s+", "", s)
    
    original_normalized = normalize(original_text)
    merged_normalized = normalize("".join(split_result))
    text_is_cjk = is_mainly_cjk(original_text)
    
    # 内容一致性检查：允许少量差异（LLM 可能修改标点、添删少量字符）
    # _realign_timestamps 的 diff 容错算法会处理这些差异
    # 只在差异过大（>5%）时触发反馈循环
    CONTENT_DIFF_THRESHOLD = 0.05  # 允许 5% 字符差异
    
    if original_normalized != merged_normalized:
        matcher = difflib.SequenceMatcher(None, original_normalized, merged_normalized, autojunk=False)
        similarity = matcher.ratio()
        diff_ratio = 1 - similarity
        
        # 使用 difflib 定位差异（用于日志和反馈）
        context_size = 10 if text_is_cjk else 20
        differences = []
        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode == "equal":
                continue
            before = original_normalized[max(0, a0 - context_size):a0]
            orig_part = original_normalized[a0:a1] or "(empty)"
            after = original_normalized[a1:a1 + context_size]
            new_part = merged_normalized[b0:b1] or "(empty)"
            
            if opcode == "replace":
                differences.append(f"...{before}[{orig_part}]{after}... → [{new_part}]")
            elif opcode == "delete":
                differences.append(f"...{before}[{orig_part}]{after}... → deleted")
            elif opcode == "insert":
                differences.append(f"...{before}↑{after}... → inserted [{new_part}]")
        
        if diff_ratio > CONTENT_DIFF_THRESHOLD:
            # 差异过大，触发反馈循环要求 LLM 修正
            error_msg = (
                f"Content mismatch too large ({diff_ratio:.1%}, threshold {CONTENT_DIFF_THRESHOLD:.0%}). "
                f"Original: {len(original_normalized)} chars, result: {len(merged_normalized)} chars:\n"
            )
            error_msg += "\n".join(f"- {diff}" for diff in differences[:5])
            error_msg += "\nKeep original text EXACTLY unchanged, only insert <br> between words."
            return False, error_msg
        else:
            # 少量差异：记录警告但接受结果，交给 _realign_timestamps 容错处理
            logger.info(
                f"内容差异 {diff_ratio:.1%} 在容许范围内 (≤{CONTENT_DIFF_THRESHOLD:.0%})，"
                f"交由对齐算法容错处理"
            )
            if differences:
                logger.debug(f"差异详情: {'; '.join(differences[:3])}")

    # 检查每段长度是否超限
    violations = []
    for i, segment in enumerate(split_result, 1):
        word_count = count_words(segment)

        max_allowed = max_word_count_cjk if text_is_cjk else max_word_count_english
        tolerance = max_allowed * 1  # 0容差

        if word_count > tolerance:
            segment_preview = segment[:40] + "..." if len(segment) > 40 else segment
            violations.append(
                f"Segment {i} '{segment_preview}': {word_count} {'chars' if text_is_cjk else 'words'} > {max_allowed} limit"
            )

    if violations:
        error_msg = "Length violations:\n" + "\n".join(f"- {v}" for v in violations)
        error_msg += "\n\nSplit these long segments further with <br>, then output the COMPLETE text with ALL segments (not just the fixed ones)."
        return False, error_msg

    # 检查最小长度（只记录debug日志，不触发反馈循环）
    min_allowed = min_word_count_cjk if text_is_cjk else min_word_count_english
    short_segments = []
    for i, segment in enumerate(split_result, 1):
        word_count = count_words(segment)
        if word_count < min_allowed:
            segment_preview = segment[:40] + "..." if len(segment) > 40 else segment
            short_segments.append(
                f"Segment {i} '{segment_preview}': {word_count} {'chars' if text_is_cjk else 'words'} < {min_allowed} min"
            )
    
    if short_segments:
        logger.debug(
            f"最小长度提示（不影响结果）:\n" + "\n".join(f"- {s}" for s in short_segments)
        )

    return True, ""


if __name__ == "__main__":
    sample_text = "大家好我叫杨玉溪来自有着良好音乐氛围的福建厦门自记事起我眼中的世界就是朦胧的童话书是各色杂乱的线条电视机是颜色各异的雪花小伙伴是只听其声不便骑行的马赛克后来我才知道这是一种眼底黄斑疾病虽不至于失明但终身无法治愈"
    sentences = split_by_llm(sample_text)
    print(f"断句结果 ({len(sentences)} 段):")
    for i, seg in enumerate(sentences, 1):
        print(f"  {i}. {seg}")
