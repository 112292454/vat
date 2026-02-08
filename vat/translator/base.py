"""翻译器基类"""

import atexit
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Union

from vat.asr.asr_data import ASRData, ASRDataSeg
from vat.subtitle_utils.entities import SubtitleProcessData
from vat.translator.types import TargetLanguage
from vat.utils.cache import generate_cache_key, get_translate_cache, is_cache_enabled
from vat.utils.logger import setup_logger

logger = setup_logger("subtitle_translator")


class BaseTranslator(ABC):
    """翻译器基类"""

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        output_dir: Union[str, Path],
        update_callback: Optional[Callable] = None,
    ):
        """
        初始化翻译器基类
        
        Args:
            thread_num: 并发线程数
            batch_num: 每批处理的字幕数量
            target_language: 目标语言
            output_dir: 输出目录，用于保存翻译结果
            update_callback: 进度回调
        """
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.target_language = target_language
        self.output_dir = Path(output_dir)
        self.is_running = True
        self.update_callback = update_callback
        self.executor = None
        self._cache = get_translate_cache()
        self._cache_hit_count = 0  # 缓存命中计数

        self._init_thread_pool()

    def _init_thread_pool(self):
        """初始化线程池"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)

    def translate_subtitle(self, subtitle_data: ASRData) -> ASRData:
        """
        翻译字幕文件
        
        翻译完成后自动保存到 output_dir/translated.srt
        
        Args:
            subtitle_data: 输入字幕数据
            
        Returns:
            翻译后的 ASRData
        """
        try:
            asr_data = subtitle_data

            # 将ASRData转换为SubtitleProcessData列表
            translate_data_list = [
                SubtitleProcessData(index=i, original_text=seg.text)
                for i, seg in enumerate(asr_data.segments, 1)
            ]

            # 分批处理字幕
            chunks = self._split_chunks(translate_data_list)

            # 多线程翻译
            translated_list = self._parallel_translate(chunks)

            # 设置字幕段的翻译文本
            new_segments = self._set_segments_translated_text(
                asr_data.segments, translated_list
            )

            result = ASRData(new_segments)
            
            # 保存翻译结果
            translated_srt = self.output_dir / "translated.srt"
            result.save(str(translated_srt))
            
            # 打印缓存命中汇总（如果有）
            total_chunks = len(chunks)
            if self._cache_hit_count > 0:
                logger.info(f"翻译结果已保存: {translated_srt} (缓存命中: {self._cache_hit_count}/{total_chunks} 批次)")
            else:
                logger.info(f"翻译结果已保存: {translated_srt}")
            
            return result
        except Exception as e:
            logger.error(f"翻译失败：{str(e)}")
            raise RuntimeError(f"翻译失败：{str(e)}")

    def _split_chunks(
        self, translate_data_list: List[SubtitleProcessData]
    ) -> List[List[SubtitleProcessData]]:
        """将字幕分割成块"""
        return [
            translate_data_list[i : i + self.batch_num]
            for i in range(0, len(translate_data_list), self.batch_num)
        ]

    def _parallel_translate(
        self, chunks: List[List[SubtitleProcessData]]
    ) -> List[SubtitleProcessData]:
        """并行翻译所有块"""
        futures = []
        translated_list = []
        failed_chunks = []
        total_chunks = len(chunks)

        logger.info(f"开始翻译，共 {total_chunks} 个批次")

        for chunk in chunks:
            future = self.executor.submit(self._safe_translate_chunk, chunk)
            futures.append((future, chunk))

        for idx, (future, chunk) in enumerate(futures, 1):
            if not self.is_running:
                break
            try:
                result = future.result()
                translated_list.extend(result)
                if idx % max(1, total_chunks // 10) == 0:
                    logger.info(f"翻译进度: {idx}/{total_chunks} 批次完成")
            except Exception as e:
                logger.error(f"翻译块失败：{str(e)}")
                failed_chunks.append((chunk, str(e)))
                # 失败时保留原文，但标记为未翻译
                for data in chunk:
                    data.translated_text = data.original_text  # 标记为未翻译
                translated_list.extend(chunk)

        # 如果所有块都失败了，抛出异常
        if failed_chunks and len(failed_chunks) == len(chunks):
            error_messages = [f"块 {i+1}: {err}" for i, (_, err) in enumerate(failed_chunks)]
            raise RuntimeError(f"所有翻译块都失败了:\n" + "\n".join(error_messages))
        
        # 如果部分块失败，记录警告但继续
        if failed_chunks:
            logger.warning(f"{len(failed_chunks)}/{len(chunks)} 个翻译块失败，已保留原文")

        return translated_list

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(chunk)
        lang = self.target_language.value
        return f"{class_name}:{chunk_key}:{lang}"

    def _safe_translate_chunk(
        self, chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """安全的翻译块"""
        try:
            cache_key = self._get_cache_key(chunk)
            cache_enabled = is_cache_enabled()
            if cache_enabled:
                cached_result = self._cache.get(cache_key, default=None)
                if cached_result is not None:
                    # 记录 cache 命中
                    self._cache_hit_count += 1
                    return cached_result

            result = self._translate_chunk(chunk)

            if self.update_callback:
                self.update_callback(result)

            if cache_enabled:
                self._cache.set(cache_key, result, expire=86400 * 7)
            return result

        except Exception as e:
            logger.exception(f"翻译失败: {str(e)}")
            raise

    @staticmethod
    def _set_segments_translated_text(
        original_segments: List[ASRDataSeg], translated_list: List[SubtitleProcessData]
    ) -> List[ASRDataSeg]:
        """设置字幕段的翻译文本"""
        # 创建索引到翻译文本的映射
        translation_map = {data.index: data.translated_text for data in translated_list}

        for i, seg in enumerate(original_segments, 1):
            if i not in translation_map:
                logger.error(f"字幕段 {i} 没有翻译")
                continue
            seg.translated_text = translation_map[i]

        return original_segments

    @abstractmethod
    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        pass

    def stop(self):
        """停止翻译器"""
        if not self.is_running:
            return

        self.is_running = False
        if hasattr(self, "executor") and self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.error(f"关闭线程池时出错：{str(e)}")
            finally:
                self.executor = None
