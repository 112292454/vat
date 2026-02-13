"""LLM 翻译器（使用 OpenAI），集成字幕优化功能"""

import json
import difflib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import as_completed

import json_repair
import openai

from vat.llm import call_llm
from vat.llm.prompts import get_prompt
from vat.translator.base import BaseTranslator, SubtitleProcessData, logger
from vat.translator.types import TargetLanguage
from vat.utils.cache import generate_cache_key
from vat.asr.asr_data import ASRData, ASRDataSeg
from vat.utils.text_utils import count_words


class LLMTranslator(BaseTranslator):
    """LLM 翻译器（OpenAI兼容API），集成字幕优化功能"""

    MAX_STEPS = 3

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        output_dir: str,
        model: str,
        custom_translate_prompt: str,
        is_reflect: bool,
        enable_optimize: bool = False,
        custom_optimize_prompt: str = "",
        enable_context: bool = True,
        api_key: str = "",
        base_url: str = "",
        optimize_model: str = "",
        optimize_api_key: str = "",
        optimize_base_url: str = "",
        update_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """
        初始化 LLM 翻译器
        
        Args:
            thread_num: 并发线程数
            batch_num: 每批处理的字幕数量
            target_language: 目标语言
            output_dir: 输出目录，用于保存翻译结果
            model: LLM 模型名称
            custom_translate_prompt: 翻译自定义提示词
            is_reflect: 是否启用反思翻译
            enable_optimize: 是否启用字幕优化（前置步骤）
            custom_optimize_prompt: 优化自定义提示词
            enable_context: 是否启用前文上下文
            update_callback: 数据回调（每批翻译结果）
            progress_callback: 进度消息回调
        """
        super().__init__(
            thread_num=thread_num,
            batch_num=batch_num,
            target_language=target_language,
            output_dir=output_dir,
            update_callback=update_callback,
            progress_callback=progress_callback,
        )

        self.model = model
        self.custom_prompt = custom_translate_prompt
        self.is_reflect = is_reflect
        self.enable_optimize = enable_optimize
        self.optimize_prompt = custom_optimize_prompt
        self.enable_context = enable_context
        self.api_key = api_key
        self.base_url = base_url
        # optimize 可独立覆写，留空则使用 translate 的凭据
        self.optimize_model = optimize_model or model
        self.optimize_api_key = optimize_api_key if optimize_api_key else api_key
        self.optimize_base_url = optimize_base_url if optimize_base_url else base_url
        
        # 存储前一个 batch 的翻译结果（用于上下文）
        self._previous_batch_result: Optional[Dict[str, str]] = None

    def translate_subtitle(self, subtitle_data: ASRData) -> ASRData:
        """翻译字幕文件（集成可选的优化前置步骤）"""
        try:
            # 1. 可选：字幕优化（内部方法）
            if self.enable_optimize:
                logger.info("开始字幕优化（LLM Translator 内部）...")
                subtitle_data = self._optimize_subtitle(subtitle_data)
                logger.info("字幕优化完成")

            # 2. 执行翻译（调用基类逻辑）
            return super().translate_subtitle(subtitle_data)
        except Exception as e:
            logger.error(f"翻译失败：{str(e)}")
            raise RuntimeError(f"翻译失败：{str(e)}")

    def _optimize_subtitle(self, asr_data: ASRData) -> ASRData:
        """
        内部方法：优化字幕内容
        
        优化完成后自动保存到 output_dir/original_optimized.srt
        """
        assert asr_data is not None, "调用契约错误: asr_data 不能为空"
        
        if not asr_data.segments:
            logger.warning("字幕内容为空，跳过优化")
            return asr_data

        # 转换为字典格式
        subtitle_dict = {str(i): seg.text for i, seg in enumerate(asr_data.segments, 1)}
        
        # 分批处理（使用基类的批量大小）
        items = list(subtitle_dict.items())
        chunks = [
            dict(items[i : i + self.batch_num])
            for i in range(0, len(items), self.batch_num)
        ]

        # 并行优化（复用线程池）
        optimized_dict: Dict[str, str] = {}
        futures = []
        total_chunks = len(chunks)
        
        if not self.executor:
            raise ValueError("线程池未初始化")
        
        for chunk in chunks:
            future = self.executor.submit(self._optimize_chunk, chunk)
            futures.append((future, chunk))

        # 收集结果
        for idx, (future, chunk) in enumerate(futures, 1):
            if not self.is_running:
                break
            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                logger.error(f"优化批次失败: {e}")
                optimized_dict.update(chunk)  # 失败时保留原文
            
            msg = f"优化进度: {idx}/{total_chunks} 批次完成"
            if idx % max(1, total_chunks // 10) == 0:
                logger.info(msg)
            if self.progress_callback:
                self.progress_callback(msg)

        # 验证数量一致性
        assert len(optimized_dict) == len(subtitle_dict), \
            f"逻辑错误: 优化后字幕数量 ({len(optimized_dict)}) 与原文数量 ({len(subtitle_dict)}) 不一致"

        # 创建新 segments
        new_segments = [
            ASRDataSeg(
                text=optimized_dict.get(str(i), seg.text),
                start_time=seg.start_time,
                end_time=seg.end_time,
                translated_text=seg.translated_text
            )
            for i, seg in enumerate(asr_data.segments, 1)
        ]
        
        assert len(new_segments) == len(asr_data.segments), \
            f"逻辑错误: 生成的 segments 数量 ({len(new_segments)}) 与原文数量 ({len(asr_data.segments)}) 不一致"
        
        result = ASRData(new_segments)
        
        # 保存优化后的原始字幕
        optimized_srt = self.output_dir / "original_optimized.srt"
        result.save(str(optimized_srt))
        logger.info(f"优化后原文已保存: {optimized_srt}")
        
        return result

    def _optimize_chunk(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        """
        优化单个字幕批次
        使用 Agent Loop 自动验证和修正
        """
        start_idx = next(iter(subtitle_chunk))
        end_idx = next(reversed(subtitle_chunk))
        logger.debug(f"正在优化字幕：{start_idx} - {end_idx}")
        
        prompt = get_prompt("optimize/subtitle")
        
        user_prompt = (
            f"Correct the following subtitles. Keep the original language, do not translate:\n"
            f"<input_subtitle>{json.dumps(subtitle_chunk, ensure_ascii=False)}</input_subtitle>"
        )

        if self.optimize_prompt:
            user_prompt += f"\nReference content:\n<reference>{self.optimize_prompt}</reference>"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_result = subtitle_chunk
        
        # Agent Loop
        for step in range(self.MAX_STEPS):
            try:
                response = call_llm(
                    messages=messages, model=self.optimize_model, temperature=0.2,
                    api_key=self.optimize_api_key, base_url=self.optimize_base_url,
                )
                
                result_text = response.choices[0].message.content
                if not result_text:
                    raise ValueError("LLM返回空结果")
                
                result_dict = json_repair.loads(result_text)
                if not isinstance(result_dict, dict):
                    raise ValueError(f"LLM返回结果类型错误，期望dict，实际{type(result_dict)}")
                
                last_result = result_dict
                
                # 验证结果
                is_valid, error_message = self._validate_optimization_result(
                    original_chunk=subtitle_chunk,
                    optimized_chunk=result_dict
                )
                
                if is_valid:
                    return result_dict
                
                # 验证失败，添加反馈
                logger.warning(f"优化验证失败，开始反馈循环 (第{step + 1}次尝试): {error_message}")
                messages.append({"role": "assistant", "content": result_text})
                messages.append({
                    "role": "user",
                    "content": f"Validation failed: {error_message}\n"
                              f"Please fix the errors and output ONLY a valid JSON dictionary.DO NOT REPLY ANY ADDITIONEL EXPLANATION OR OTHER PREVIOUS TEXT."
                })
                
            except Exception as e:
                logger.warning(f"优化批次尝试 {step+1} 失败: {e}")
                if step == self.MAX_STEPS - 1:
                    return last_result
        
        return last_result

    def _validate_optimization_result(
        self, original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
    ) -> Tuple[bool, str]:
        """验证优化结果"""
        expected_keys = set(original_chunk.keys())
        actual_keys = set(optimized_chunk.keys())

        # 检查键匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_parts = []
            
            if missing:
                error_parts.append(f"Missing keys: {sorted(missing)}")
            if extra:
                error_parts.append(f"Extra keys: {sorted(extra)}")

            error_msg = (
                "\n".join(error_parts) + f"\nRequired keys: {sorted(expected_keys)}\n"
                f"Please return the COMPLETE optimized dictionary with ALL {len(expected_keys)} keys."
            )
            return False, error_msg

        # 检查改动是否过大
        excessive_changes = []
        for key in expected_keys:
            original_text = original_chunk[key]
            optimized_text = optimized_chunk[key]

            original_cleaned = re.sub(r"\s+", " ", original_text).strip()
            optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()

            matcher = difflib.SequenceMatcher(None, original_cleaned, optimized_cleaned)
            similarity = matcher.ratio()
            similarity_threshold = 0.3 if count_words(original_text) <= 10 else 0.5

            if similarity < similarity_threshold:
                excessive_changes.append(
                    f"Key '{key}': similarity {similarity:.1%} < {similarity_threshold:.0%}. "
                    f"Original: '{original_text}' → Optimized: '{optimized_text}'"
                )

        if excessive_changes:
            error_msg = ";\n".join(excessive_changes)
            error_msg += (
                "\n\nYour optimizations changed the text too much. "
                "Keep high similarity (≥70% for normal text) by making MINIMAL changes."
            )
            return False, error_msg

        return True, ""

    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        logger.debug(
            f"正在翻译字幕：{subtitle_chunk[0].index} - {subtitle_chunk[-1].index}"
        )

        subtitle_dict = {str(data.index): data.original_text for data in subtitle_chunk}

        # 获取提示词
        if self.is_reflect:
            prompt = get_prompt(
                "translate/reflect",
                target_language=self.target_language,
                custom_prompt=self.custom_prompt,
            )
        else:
            prompt = get_prompt(
                "translate/standard",
                target_language=self.target_language,
                custom_prompt=self.custom_prompt,
            )

        try:
            # 构建带上下文的输入（新增）
            user_input = self._build_input_with_context(subtitle_dict)
            
            result_dict = self._agent_loop(prompt, user_input, expected_keys=set(subtitle_dict.keys()))

            # 处理反思翻译模式的结果
            if self.is_reflect and isinstance(result_dict, dict):
                processed_result = {
                    k: f"{v.get('native_translation', v) if isinstance(v, dict) else v}"
                    for k, v in result_dict.items()
                }
            else:
                processed_result = {k: f"{v}" for k, v in result_dict.items()}

            # 保存当前 batch 结果供下次使用（新增）
            self._previous_batch_result = processed_result.copy()

            for data in subtitle_chunk:
                data.translated_text = processed_result.get(
                    str(data.index), data.original_text
                )
            return subtitle_chunk
        except openai.RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error: {str(e)}")
            # Rate limit 错误可以重试，但这里应该抛出异常让上层处理
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {str(e)}")
            # 认证错误应该立即失败，不应该降级处理
            raise RuntimeError(f"API 认证失败: {str(e)}") from e
        except openai.NotFoundError as e:
            logger.error(f"OpenAI NotFound Error: {str(e)}")
            # 模型不存在错误应该立即失败
            raise RuntimeError(f"模型不存在: {str(e)}") from e
        except Exception as e:
            import traceback
            logger.error(f"翻译块失败: {str(e)}, 尝试降级处理,traceback: {traceback.format_exc()}")
            # 其他错误尝试降级处理
            try:
                return self._translate_chunk_single(subtitle_chunk)
            except Exception as fallback_error:
                logger.error(f"降级翻译也失败: {str(fallback_error)}")
                # 如果降级也失败，抛出异常
                raise RuntimeError(f"翻译失败且降级处理也失败: {str(e)}") from e

    def _agent_loop(
        self, 
        system_prompt: str, 
        user_input: str,
        expected_keys: Optional[set] = None
    ) -> Dict[str, str]:
        """Agent loop翻译/优化字幕块"""
        assert system_prompt, "调用契约错误: system_prompt 不能为空"
        assert user_input, "调用契约错误: user_input 不能为空"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        last_response_dict = None
        
        for _ in range(self.MAX_STEPS):
            response = call_llm(
                messages=messages, model=self.model,
                api_key=self.api_key, base_url=self.base_url,
            )
            if not response or not response.choices:
                raise RuntimeError("LLM 未返回有效响应")
            
            content = response.choices[0].message.content.strip()
            if not content:
                raise RuntimeError("LLM 返回内容为空")
            
            response_dict = json_repair.loads(content)
            last_response_dict = response_dict
            
            # 使用 expected_keys 验证（如果提供）
            validation_keys = expected_keys if expected_keys else set(response_dict.keys())
            is_valid, error_message = self._validate_llm_response(
                response_dict, validation_keys
            )
            if is_valid:
                return response_dict
            else:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(response_dict, ensure_ascii=False),
                })
                messages.append({
                    "role": "user",
                    "content": f"Error: {error_message}\n\n"
                              f"Fix the errors above and output ONLY a valid JSON dictionary with ALL {len(validation_keys)} keys",
                })

        return last_response_dict

    def _validate_llm_response(
        self, response_dict: Any, expected_keys: set
    ) -> Tuple[bool, str]:
        """验证LLM翻译结果（支持普通和反思模式）"""
        if not isinstance(response_dict, dict):
            return (
                False,
                f"Output must be a dict, got {type(response_dict).__name__}. Use format: {{'0': 'text', '1': 'text'}}",
            )

        actual_keys = set(response_dict.keys())

        def sort_keys(keys):
            return sorted(keys, key=lambda x: int(x) if x.isdigit() else x)

        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_parts = []

            if missing:
                error_parts.append(
                    f"Missing keys {sort_keys(missing)} - you must translate these items"
                )
            if extra:
                error_parts.append(
                    f"Extra keys {sort_keys(extra)} - these keys are not in input, remove them"
                )

            return (False, "; ".join(error_parts))

        # 如果是反思模式，检查嵌套结构
        if self.is_reflect:
            for key, value in response_dict.items():
                if not isinstance(value, dict):
                    return (
                        False,
                        f"Key '{key}': value must be a dict with 'native_translation' field. Got {type(value).__name__}.",
                    )

                if "native_translation" not in value:
                    available_keys = list(value.keys())
                    return (
                        False,
                        f"Key '{key}': missing 'native_translation' field. Found keys: {available_keys}. Must include 'native_translation'.",
                    )

        return True, ""

    def _translate_chunk_single(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """单条翻译模式（降级方案）"""
        single_prompt = get_prompt(
            "translate/single", target_language=self.target_language
        )

        for data in subtitle_chunk:
            try:
                response = call_llm(
                    messages=[
                        {"role": "system", "content": single_prompt},
                        {"role": "user", "content": data.original_text},
                    ],
                    model=self.model,
                    temperature=0.7,
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                translated_text = response.choices[0].message.content.strip()
                data.translated_text = translated_text
            except Exception as e:
                logger.error(f"单条翻译失败 {data.index}: {str(e)}")

        return subtitle_chunk

    def _build_input_with_context(self, subtitle_dict: Dict[str, str]) -> str:
        """
        构建带上下文的输入
        
        Args:
            subtitle_dict: 当前 batch 的字幕字典
            
        Returns:
            格式化的输入字符串
        """
        if not self.enable_context or self._previous_batch_result is None:
            # 第一个 batch 或未启用上下文
            return json.dumps(subtitle_dict, ensure_ascii=False)
        
        # 构建上下文部分
        context_lines = []
        for key, text in self._previous_batch_result.items():
            context_lines.append(f"[{key}]: {text}")
        
        context_text = "\n".join(context_lines)
        
        # 组合格式
        input_text = f"""Previous context (for reference only, maintain consistency with these translations, but DO NOT TRANSLATE THE PREVIOUS CONTEXT ITSELF):
{context_text}

Translate the following (output ONLY these keys):
{json.dumps(subtitle_dict, ensure_ascii=False)}"""
        
        return input_text

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(chunk)
        lang = self.target_language.value
        model = self.model
        return f"{class_name}:{chunk_key}:{lang}:{model}"
