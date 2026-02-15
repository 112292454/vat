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
        proxy: str = "",
        optimize_proxy: str = "",
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
            proxy: 翻译 LLM 代理地址（空字符串=使用环境变量）
            optimize_proxy: 优化 LLM 代理地址（空字符串=使用 proxy）
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
        self.proxy = proxy
        self.optimize_proxy = optimize_proxy  # 由 config.get_stage_proxy 统一 resolve，无类内 fallback
        # optimize 可独立覆写，留空则使用 translate 的凭据（api_key/base_url 的 fallback 由调用方处理）
        self.optimize_model = optimize_model or model
        self.optimize_api_key = optimize_api_key if optimize_api_key else api_key
        self.optimize_base_url = optimize_base_url if optimize_base_url else base_url
        
        # 存储前一个 batch 的翻译结果（用于上下文）
        self._previous_batch_result: Optional[Dict[str, str]] = None
        # 处理过程中的非致命警告（由 pipeline 读取并写入 processing_notes）
        self._processing_warnings: List[str] = []

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
        failed_optimize_chunks = 0
        last_optimize_error = None
        for idx, (future, chunk) in enumerate(futures, 1):
            if not self.is_running:
                break
            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                logger.error(f"优化批次失败: {e}")
                optimized_dict.update(chunk)  # 失败时保留原文
                failed_optimize_chunks += 1
                last_optimize_error = e
            
            msg = f"优化进度: {idx}/{total_chunks} 批次完成"
            # if idx % max(1, total_chunks // 10) == 0:
            #     logger.info(msg)
            if self.progress_callback:
                self.progress_callback(msg)

        # 失败容忍：仅容忍偶发网络抖动级别的失败
        # 阈值：最多容忍 max(2, total*5%) 个批次失败，超过则判定为系统性问题
        if failed_optimize_chunks > 0:
            tolerance = max(2, int(total_chunks * 0.05))
            if failed_optimize_chunks > tolerance:
                raise RuntimeError(
                    f"优化失败过多: {failed_optimize_chunks}/{total_chunks} 个批次失败"
                    f"（容忍上限 {tolerance}），可能存在 API/网络问题: {last_optimize_error}"
                )
            # 少量失败（网络抖动级别）：记录为 processing_warning（pipeline 读取后写入 DB）
            warn_msg = f"{failed_optimize_chunks}/{total_chunks} 个优化批次失败，已保留原文"
            logger.warning(warn_msg)
            self._processing_warnings.append(warn_msg)

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
                    proxy=self.optimize_proxy,
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
                
            except (openai.BadRequestError, openai.AuthenticationError, openai.NotFoundError) as e:
                # 不可重试的配置错误（地区限制、认证失败、模型不存在），立即失败
                raise RuntimeError(f"优化 API 不可用: {e}") from e
            except Exception as e:
                logger.warning(f"优化批次尝试 {step+1} 失败: {e}")
                if step == self.MAX_STEPS - 1:
                    return last_result
        
        # 反馈循环用尽仍未通过验证，尝试自动修复错位
        if isinstance(last_result, dict) and set(last_result.keys()) == set(subtitle_chunk.keys()):
            realigned = self._try_realign_shifted_keys(subtitle_chunk, last_result)
            if realigned is not None:
                return realigned
        
        return last_result

    @staticmethod
    def _normalize_kana(text: str) -> str:
        """片假名→平假名归一化（用于相似度比较）
        
        避免 ルルドライオン→るるどらいおん 这种合法的片假名→平假名规范化
        被误判为"改动过大"。
        """
        # 片假名 U+30A1-U+30F6 → 平假名 U+3041-U+3096（偏移 0x60）
        result = []
        for ch in text:
            cp = ord(ch)
            if 0x30A1 <= cp <= 0x30F6:
                result.append(chr(cp - 0x60))
            else:
                result.append(ch)
        return ''.join(result)

    @staticmethod
    def _try_realign_shifted_keys(
        original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """检测并修复 LLM 返回值系统性错位的问题
        
        当 LLM 在优化时将字幕内容整体移位（如 Opt[k]=Orig[k+2]），
        尝试通过贪心匹配将每个 optimized value 对应回最相似的 original key。
        
        Returns:
            修复后的 dict，如果无法修复则返回 None
        """
        keys = sorted(original_chunk.keys(), key=lambda k: int(k))
        opt_values = [optimized_chunk[k] for k in keys]
        orig_values = [original_chunk[k] for k in keys]
        
        # 检测有多少个 key 的值与原文不匹配但与其他 key 的原文高度匹配
        shifted_count = 0
        for i, opt_val in enumerate(opt_values):
            orig_val = orig_values[i]
            # 如果与自身原文相似度 >= 0.5，不算错位
            self_sim = difflib.SequenceMatcher(None, orig_val, opt_val).ratio()
            if self_sim >= 0.5:
                continue
            # 检查是否与其他 key 的原文高度匹配
            for j, other_orig in enumerate(orig_values):
                if i == j:
                    continue
                cross_sim = difflib.SequenceMatcher(None, other_orig, opt_val).ratio()
                if cross_sim >= 0.8:
                    shifted_count += 1
                    break
        
        # 如果错位数量 >= 3，认为是系统性错位，尝试修复
        if shifted_count < 3:
            return None
        
        logger.warning(f"检测到优化结果系统性错位 ({shifted_count} 个 key)，尝试自动修复")
        
        # 贪心匹配：为每个 key 找最佳匹配的 optimized value
        # 对于每个 original，找到与其最相似的 optimized value
        used = set()
        realigned = {}
        
        # 按原文与优化值的最佳匹配排序
        matches = []
        for i, key in enumerate(keys):
            for j, opt_val in enumerate(opt_values):
                sim = difflib.SequenceMatcher(None, orig_values[i], opt_val).ratio()
                matches.append((sim, i, j))
        
        matches.sort(reverse=True)
        used_orig = set()
        used_opt = set()
        
        for sim, i, j in matches:
            if i in used_orig or j in used_opt:
                continue
            used_orig.add(i)
            used_opt.add(j)
            realigned[keys[i]] = opt_values[j]
        
        # 未匹配的用原文填充
        for i, key in enumerate(keys):
            if key not in realigned:
                realigned[key] = orig_values[i]
        
        logger.info(f"自动修复完成，重新对齐了 {shifted_count} 个错位的 key")
        return realigned

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

        # 检查改动是否过大（使用片假名→平假名归一化避免误判）
        excessive_changes = []
        for key in expected_keys:
            original_text = original_chunk[key]
            optimized_text = optimized_chunk[key]

            original_cleaned = re.sub(r"\s+", " ", original_text).strip()
            optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()

            # 先用归一化后的文本比较（片假名→平假名不算差异）
            orig_normalized = self._normalize_kana(original_cleaned)
            opt_normalized = self._normalize_kana(optimized_cleaned)
            
            matcher = difflib.SequenceMatcher(None, orig_normalized, opt_normalized)
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
        except openai.BadRequestError as e:
            # 400 错误（地区不支持、参数无效等）是配置问题，不可重试
            logger.error(f"OpenAI BadRequest Error: {str(e)}")
            raise RuntimeError(f"API 请求被拒绝（可能是地区限制或参数错误）: {str(e)}") from e
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
                proxy=self.proxy,
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
        """单条翻译模式（降级方案）
        
        逐条翻译字幕。翻译不允许部分失败：任何一条失败即立即抛出异常。
        """
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
                    proxy=self.proxy,
                )
                translated_text = response.choices[0].message.content.strip()
                data.translated_text = translated_text
            except Exception as e:
                # 翻译零容忍：任何一条失败即立报错，不允许部分翻译缺失
                raise RuntimeError(
                    f"字幕 {data.index} 翻译失败: {e}"
                ) from e

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
