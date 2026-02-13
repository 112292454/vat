"""
配置管理系统
"""
import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict


def _resolve_env_var(value: Optional[str]) -> str:
    """Resolve ${VAR_NAME} environment variable placeholders.
    
    Args:
        value: String that may contain ${VAR_NAME} placeholder, or None/empty
        
    Returns:
        Resolved value, or empty string if env var not set or value is None/empty
    """
    if not value:
        return ""
    match = re.match(r'^\$\{(.*)\}$', value)
    if match:
        env_var = match.group(1)
        env_val = os.environ.get(env_var)
        if env_val:
            return env_val
        else:
            logging.getLogger("vat.config").warning(f"环境变量 {env_var} 未设置")
            return ""
    return value


@dataclass
class StorageConfig:
    """存储配置"""
    work_dir: str
    output_dir: str
    database_path: str
    models_dir: str  # 所有模型文件的根目录
    resource_dir: str
    fonts_dir: str
    subtitle_style_dir: str
    cache_dir: str
    
    def __post_init__(self):
        # 展开用户目录并转换为绝对路径
        self.work_dir = str(Path(self.work_dir).expanduser().absolute())
        self.output_dir = str(Path(self.output_dir).expanduser().absolute())
        self.database_path = str(Path(self.database_path).expanduser().absolute())
        self.models_dir = str(Path(self.models_dir).expanduser().absolute())
        self.resource_dir = str(Path(self.resource_dir).expanduser().absolute())
        self.fonts_dir = str(Path(self.fonts_dir).expanduser().absolute())
        self.subtitle_style_dir = str(Path(self.subtitle_style_dir).expanduser().absolute())
        self.cache_dir = str(Path(self.cache_dir).expanduser().absolute())


@dataclass
class YouTubeDownloaderConfig:
    """YouTube下载器配置"""
    format: str
    max_workers: int
    download_subtitles: bool = True
    subtitle_languages: List[str] = None  # 默认在 __post_init__ 中设置
    subtitle_format: str = "vtt"
    
    def __post_init__(self):
        if self.subtitle_languages is None:
            self.subtitle_languages = ["ja", "zh", "en"]


@dataclass
class VideoInfoTranslateConfig:
    """视频信息翻译 LLM 配置（下载/同步时翻译标题、描述等）
    
    model/api_key/base_url 留空则继承全局 llm 配置。
    """
    model: str = ""           # 使用的模型（空=使用全局 llm.model）
    api_key: str = ""         # 覆写 API Key（支持 ${VAR_NAME}）
    base_url: str = ""        # 覆写 Base URL
    
    def __post_init__(self):
        self.api_key = _resolve_env_var(self.api_key) if self.api_key else ""
        self.base_url = _resolve_env_var(self.base_url) if self.base_url else ""


@dataclass
class SceneIdentifyConfig:
    """场景识别 LLM 配置（下载时根据标题/简介判断视频场景）
    
    model/api_key/base_url 留空则继承全局 llm 配置。
    """
    model: str = ""           # 使用的模型（空=使用全局 llm.model）
    api_key: str = ""         # 覆写 API Key（支持 ${VAR_NAME}）
    base_url: str = ""        # 覆写 Base URL
    
    def __post_init__(self):
        self.api_key = _resolve_env_var(self.api_key) if self.api_key else ""
        self.base_url = _resolve_env_var(self.base_url) if self.base_url else ""


@dataclass
class DownloaderConfig:
    """下载器配置"""
    youtube: YouTubeDownloaderConfig
    video_info_translate: VideoInfoTranslateConfig = field(default_factory=VideoInfoTranslateConfig)
    scene_identify: SceneIdentifyConfig = field(default_factory=SceneIdentifyConfig)


@dataclass
class PostProcessingConfig:
    """ASR 后处理配置"""
    enable_hallucination_detection: bool = True   # 启用幻觉检测
    enable_repetition_cleaning: bool = True       # 启用重复清理
    enable_japanese_processing: bool = True       # 启用日语特殊处理
    min_confidence: float = 0.8                   # 幻觉检测最小置信度
    custom_blacklist: List[str] = field(default_factory=list)  # 自定义幻觉黑名单


@dataclass
class VocalSeparationConfig:
    """人声分离配置"""
    enable: bool = False                          # 是否启用人声分离（默认关闭）
    auto_detect_bgm: bool = True                  # 自动检测是否需要人声分离（游戏/歌回场景）
    model_filename: str = "vocal_separator/model.ckpt"  # 模型权重路径（相对于 storage.models_dir）
    save_accompaniment: bool = False              # 是否保存伴奏


@dataclass
class SplitConfig:
    """智能断句配置（归属 ASR）"""
    enable: bool
    mode: str  # "sentence" | "semantic"
    max_words_cjk: int      # 硬性限制：每句最大字符数
    max_words_english: int  # 硬性限制：每句最大单词数
    min_words_cjk: int      # 软性建议：每句最小字符数（避免过短片段）
    min_words_english: int  # 软性建议：每句最小单词数
    model: str  # 可以独立配置 LLM 模型
    
    # 推荐长度（软性建议）
    recommend_words_cjk: int = 18     # 理想的每句字符数
    recommend_words_english: int = 10  # 理想的每句单词数
    
    # 模型升级链：断句失败时自动尝试更强模型
    allow_model_upgrade: bool = False         # 是否允许模型升级（默认关闭）
    model_upgrade_chain: List[str] = field(default_factory=list)  # 模型升级顺序
    
    # 分块配置
    enable_chunking: bool = True              # 是否启用分块（短视频可关闭）
    chunk_size_sentences: int = 50            # 每块句子数（按原始ASR片段计数）
    chunk_overlap_sentences: int = 5          # 块之间重叠句子数
    chunk_min_threshold: int = 30             # 小于此句子数不分块，直接全文处理
    
    # LLM 连接覆写（可选，留空则使用全局 llm 配置）
    api_key: str = ""    # 覆写 API Key（支持 ${VAR_NAME} 环境变量）
    base_url: str = ""   # 覆写 Base URL
    
    def __post_init__(self):
        self.api_key = _resolve_env_var(self.api_key) if self.api_key else ""
        self.base_url = _resolve_env_var(self.base_url) if self.base_url else ""


@dataclass
class ASRConfig:
    """语音识别配置"""
    backend: str
    model: str
    language: str
    device: str
    compute_type: str
    vad_filter: bool
    beam_size: int
    models_subdir: str
    
    # 高级参数
    word_timestamps: bool
    condition_on_previous_text: bool
    temperature: List[float]
    compression_ratio_threshold: float
    log_prob_threshold: float
    no_speech_threshold: float
    initial_prompt: str
    repetition_penalty: float
    hallucination_silence_threshold: Optional[float]
    
    # VAD参数
    vad_threshold: float
    vad_min_speech_duration_ms: int
    vad_max_speech_duration_s: float
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
    
    # ChunkedASR 分块处理配置
    enable_chunked: bool
    chunk_length_sec: int
    chunk_overlap_sec: int
    chunk_concurrency: int
    
    # Split 配置（嵌套）
    split: SplitConfig
    
    # Pipeline模式配置（已废弃，保留默认值以兼容旧配置）
    use_pipeline: bool = False
    enable_diarization: bool = False
    enable_punctuation: bool = False
    pipeline_batch_size: int = 8
    pipeline_chunk_length: int = 30
    num_speakers: Optional[int] = 1
    min_speakers: Optional[int] = 1
    max_speakers: Optional[int] = 2
    
    # 后处理配置（新增）
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    
    # 人声分离配置（新增）
    vocal_separation: VocalSeparationConfig = field(default_factory=VocalSeparationConfig)


@dataclass
class LocalTranslatorConfig:
    """本地翻译模型配置"""
    model_filename: str
    backend: str
    n_gpu_layers: int
    context_size: int




@dataclass
class OptimizeConfig:
    """字幕优化配置（归属 Translator）
    
    model, thread_num, batch_size 默认继承自父级 LLMTranslatorConfig，可在此覆写。
    api_key, base_url 默认继承自父级 → 全局 llm，可在此覆写。
    """
    enable: bool
    custom_prompt: str  # 文件名（相对于 vat/llm/prompts/custom/），空字符串表示不使用
    
    # 可选覆写（留空继承父级 translator.llm → 全局 llm）
    model: str = ""          # 覆写模型
    api_key: str = ""        # 覆写 API Key（支持 ${VAR_NAME}）
    base_url: str = ""       # 覆写 Base URL
    batch_size: int = 0      # 覆写批大小（0=继承父级）
    thread_num: int = 0      # 覆写线程数（0=继承父级）
    
    def __post_init__(self):
        """读取自定义提示词文件"""
        # 解析环境变量
        self.api_key = _resolve_env_var(self.api_key) if self.api_key else ""
        self.base_url = _resolve_env_var(self.base_url) if self.base_url else ""
        
        if not self.custom_prompt:
            # 空字符串表示不使用自定义提示词
            return
        
        # 获取 prompts/custom 目录的绝对路径
        prompts_custom_dir = Path(__file__).parent / "llm" / "prompts" / "custom" / "optimize"
        prompt_file = prompts_custom_dir / (self.custom_prompt + ".md")
        
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"优化自定义提示词文件不存在: {self.custom_prompt}\n"
                f"期望路径: {prompt_file}\n"
                f"请将文件放置在 vat/llm/prompts/custom/optimize/ 目录下"
            )
        
        # 读取文件内容并替换配置值
        try:
            self.custom_prompt = prompt_file.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(
                f"读取优化自定义提示词文件失败: {self.custom_prompt}\n"
                f"文件路径: {prompt_file}\n"
                f"错误: {e}"
            )


@dataclass
class LLMTranslatorConfig:
    """LLM翻译器配置"""
    model: str
    enable_reflect: bool
    batch_size: int
    thread_num: int
    custom_prompt: str  # 文件名（相对于 vat/llm/prompts/custom/），空字符串表示不使用
    
    # Optimize 配置（嵌套）
    optimize: OptimizeConfig
    
    # 上下文配置（新增）
    enable_context: bool = True  # 是否启用前文上下文
    
    # LLM 连接覆写（可选，留空则使用全局 llm 配置）
    api_key: str = ""    # 覆写 API Key（支持 ${VAR_NAME} 环境变量）
    base_url: str = ""   # 覆写 Base URL
    
    def __post_init__(self):
        """解析环境变量 + 读取自定义提示词文件"""
        self.api_key = _resolve_env_var(self.api_key) if self.api_key else ""
        self.base_url = _resolve_env_var(self.base_url) if self.base_url else ""
        
        if not self.custom_prompt:
            # 空字符串表示不使用自定义提示词
            return
        
        # 获取 prompts/custom 目录的绝对路径
        prompts_custom_dir = Path(__file__).parent / "llm" / "prompts" / "custom" / "translate"
        prompt_file = prompts_custom_dir / (self.custom_prompt + ".md")   
        
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"翻译自定义提示词文件不存在: {self.custom_prompt}\n"
                f"期望路径: {prompt_file}\n"
                f"请将文件放置在 vat/llm/prompts/custom/translate/ 目录下"  
            )
        
        # 读取文件内容并替换配置值
        try:
            self.custom_prompt = prompt_file.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(
                f"读取翻译自定义提示词文件失败: {self.custom_prompt}\n"
                f"文件路径: {prompt_file}\n"
                f"错误: {e}"
            )


@dataclass
class TranslatorConfig:
    """翻译器配置"""
    backend_type: str  # 翻译后端类型：llm（在线大模型）/ local（本地模型，暂未实现）
    source_language: str
    target_language: str
    llm: LLMTranslatorConfig
    local: LocalTranslatorConfig
    skip_translate: bool = False  # 跳过翻译，直接使用ASR原文（debug用）


@dataclass
class LLMConfig:
    """
    统一的LLM配置（所有LLM调用共享：断句、翻译、视频信息提取等）
    
    配置加载时自动设置环境变量，避免各模块重复设置
    """
    api_key: str
    base_url: str
    model: str = ""  # 全局默认模型（各阶段未指定 model 时的 fallback）
    
    _initialized: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """处理API Key并设置环境变量"""
        if self._initialized:
            return
            
        logger = logging.getLogger("vat.config")
        
        # 解析环境变量占位符
        self.api_key = _resolve_env_var(self.api_key)
        self.base_url = _resolve_env_var(self.base_url)
        
        # 统一设置环境变量（所有LLM调用都从环境变量读取）
        if self.api_key and not self.api_key.startswith("${"):
            os.environ["OPENAI_API_KEY"] = self.api_key
            logger.debug("已设置 OPENAI_API_KEY 环境变量")
        
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
            logger.debug(f"已设置 OPENAI_BASE_URL 环境变量: {self.base_url}")
        
        # 检查配置完整性
        if not self.api_key or not self.base_url:
            logger.warning(
                "LLM 配置不完整，部分功能（智能断句、翻译、视频信息翻译）可能无法使用。"
                "请在配置文件中设置 llm.api_key 和 llm.base_url"
            )
        
        object.__setattr__(self, '_initialized', True)
    
    def is_available(self) -> bool:
        """检查LLM配置是否可用"""
        return bool(self.api_key and self.base_url)


# ProcessorConfig 已删除
# Split 配置已迁移到 ASRConfig.split
# Optimize 配置已迁移到 LLMTranslatorConfig.optimize


# @dataclass
# class ASSStyleConfig:
#     """ASS字幕样式配置"""
#     font: str
#     font_size: int
#     primary_color: str
#     outline_color: str
#     back_color: str
#     bold: bool
#     italic: bool
#     outline: float
#     shadow: float
#     margin_v: int


@dataclass
class EmbedderConfig:
    """字幕嵌入配置"""
    subtitle_formats: List[str]
    embed_mode: str
    output_container: str
    video_codec: str
    audio_codec: str
    crf: int
    preset: str
    use_gpu: bool
    subtitle_style: str


@dataclass
class BilibiliUploadTemplates:
    """B站上传模板配置"""
    title: str = "${translated_title}"
    description: str = "${translated_desc}"
    custom_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class BilibiliUploaderConfig:
    """B站上传器配置"""
    cookies_file: str
    line: str = "AUTO"
    threads: int = 3
    copyright: int = 2  # 1=自制, 2=转载
    default_tid: int = 21
    default_tags: List[str] = field(default_factory=lambda: ["VTuber", "日本"])
    auto_cover: bool = True
    cover_source: str = "thumbnail"
    season_id: Optional[int] = None
    templates: Optional[BilibiliUploadTemplates] = None


@dataclass
class UploaderConfig:
    """上传器配置"""
    bilibili: BilibiliUploaderConfig


@dataclass
class GPUConfig:
    """GPU 配置"""
    device: str  # "auto", "cuda:N", "cpu"
    allow_cpu_fallback: bool
    min_free_memory_mb: int


@dataclass
class ConcurrencyConfig:
    """并发配置"""
    gpu_devices: List[int]
    max_concurrent_per_gpu: int


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str
    file: str
    format: str


@dataclass
class ProxyConfig:
    """
    全局代理配置（统一管理，用于下载器、HuggingFace 模型加载、LLM API 调用等）
    """
    http_proxy: str  # HTTP/HTTPS 代理地址，空字符串表示不使用代理
    
    _initialized: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """设置代理环境变量"""
        if self._initialized:
            return
        
        logger = logging.getLogger("vat.config")
        
        if self.http_proxy:
            # 设置环境变量（用于 requests、httpx、HuggingFace 等库）
            os.environ["HTTP_PROXY"] = self.http_proxy
            os.environ["HTTPS_PROXY"] = self.http_proxy
            os.environ["http_proxy"] = self.http_proxy
            os.environ["https_proxy"] = self.http_proxy
            logger.debug(f"已设置全局代理环境变量: {self.http_proxy}")
        else:
            # 清除代理环境变量
            for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
                os.environ.pop(var, None)
            logger.debug("代理未配置，已清除代理环境变量")
        
        object.__setattr__(self, '_initialized', True)
    
    def get_proxy(self) -> Optional[str]:
        """获取代理地址，空字符串返回 None"""
        return self.http_proxy if self.http_proxy else None


@dataclass
class WebConfig:
    """Web UI 配置"""
    host: str = "0.0.0.0"    # 监听地址
    port: int = 8080          # 监听端口


@dataclass
class Config:
    """主配置类"""
    storage: StorageConfig
    downloader: DownloaderConfig
    asr: ASRConfig
    translator: TranslatorConfig
    embedder: EmbedderConfig
    uploader: UploaderConfig
    gpu: GPUConfig  # GPU 配置
    concurrency: ConcurrencyConfig
    logging: LoggingConfig
    llm: LLMConfig  # 统一的LLM配置（在配置加载时自动设置环境变量）
    proxy: ProxyConfig  # 全局代理配置
    web: WebConfig  # Web UI 配置
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        path = Path(yaml_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        # 存储配置
        storage = StorageConfig(**data['storage'])
        
        # 下载器配置
        downloader_data = data['downloader']
        youtube = YouTubeDownloaderConfig(**downloader_data['youtube'])
        vit_data = downloader_data.get('video_info_translate', {})
        vit_config = VideoInfoTranslateConfig(
            model=vit_data.get('model', ''),
            api_key=vit_data.get('api_key', ''),
            base_url=vit_data.get('base_url', ''),
        )
        si_data = downloader_data.get('scene_identify', {})
        si_config = SceneIdentifyConfig(
            model=si_data.get('model', ''),
            api_key=si_data.get('api_key', ''),
            base_url=si_data.get('base_url', ''),
        )
        downloader = DownloaderConfig(youtube=youtube, video_info_translate=vit_config, scene_identify=si_config)
        
        # 语音识别配置（含嵌套的 split、postprocessing、vocal_separation 配置）
        asr_data = data['asr']
        split_config = SplitConfig(**asr_data['split'])
        
        # 后处理配置（可选，有默认值）
        postprocessing_data = asr_data.get('postprocessing', {})
        postprocessing_config = PostProcessingConfig(
            enable_hallucination_detection=postprocessing_data.get('enable_hallucination_detection', True),
            enable_repetition_cleaning=postprocessing_data.get('enable_repetition_cleaning', True),
            enable_japanese_processing=postprocessing_data.get('enable_japanese_processing', True),
            min_confidence=postprocessing_data.get('min_confidence', 0.8),
            custom_blacklist=postprocessing_data.get('custom_blacklist', []),
        )
        
        # 人声分离配置（可选，有默认值）
        vocal_sep_data = asr_data.get('vocal_separation', {})
        vocal_separation_config = VocalSeparationConfig(
            enable=vocal_sep_data.get('enable', False),
            auto_detect_bgm=vocal_sep_data.get('auto_detect_bgm', True),
            model_filename=vocal_sep_data.get('model_filename', 'vocal_separator/model.ckpt'),
            save_accompaniment=vocal_sep_data.get('save_accompaniment', False),
        )
        
        # 移除嵌套键以避免传递给 ASRConfig
        asr_data_copy = dict(asr_data)
        asr_data_copy.pop('split', None)
        asr_data_copy.pop('postprocessing', None)
        asr_data_copy.pop('vocal_separation', None)
        asr = ASRConfig(
            **asr_data_copy, 
            split=split_config,
            postprocessing=postprocessing_config,
            vocal_separation=vocal_separation_config
        )
        
        # 翻译器配置（含嵌套的 optimize 配置）
        translator_data = data['translator']
        llm_data = translator_data['llm']
        optimize_config = OptimizeConfig(**llm_data['optimize'])
        # 移除 optimize 键
        llm_data_copy = dict(llm_data)
        llm_data_copy.pop('optimize', None)
        llm_config = LLMTranslatorConfig(**llm_data_copy, optimize=optimize_config)
        local_config = LocalTranslatorConfig(**translator_data['local'])
        translator = TranslatorConfig(
            backend_type=translator_data['backend_type'],
            source_language=translator_data['source_language'],
            target_language=translator_data['target_language'],
            llm=llm_config,
            local=local_config,
            skip_translate=translator_data.get('skip_translate', False),
        )
        
        # 字幕嵌入配置
        embedder_data = data['embedder']
        embedder = EmbedderConfig(
            subtitle_formats=embedder_data['subtitle_formats'],
            embed_mode=embedder_data['embed_mode'],
            output_container=embedder_data['output_container'],
            video_codec=embedder_data['video_codec'],
            audio_codec=embedder_data['audio_codec'],
            crf=embedder_data['crf'],
            preset=embedder_data['preset'],
            use_gpu=embedder_data['use_gpu'],
            subtitle_style=embedder_data['subtitle_style'],
        )
        
        # 上传器配置
        # 连接设置来自 default.yaml，内容设置来自 config/upload.yaml（支持 Web UI 在线编辑）
        uploader_data = data['uploader']
        bilibili_conn = uploader_data['bilibili']
        
        # 从 config/upload.yaml 加载内容设置（投稿参数、模板等）
        from vat.uploaders.upload_config import UploadConfigManager
        upload_mgr = UploadConfigManager()
        upload_content = upload_mgr.load().bilibili
        
        # 合并模板
        templates = BilibiliUploadTemplates(
            title=upload_content.templates.title,
            description=upload_content.templates.description,
            custom_vars=upload_content.templates.custom_vars,
        )
        
        bilibili = BilibiliUploaderConfig(
            # 连接设置（default.yaml）
            cookies_file=bilibili_conn.get('cookies_file', 'cookies/bilibili/account.json'),
            line=bilibili_conn.get('line', 'AUTO'),
            threads=bilibili_conn.get('threads', 3),
            # 内容设置（config/upload.yaml）
            copyright=upload_content.copyright,
            default_tid=upload_content.default_tid,
            default_tags=upload_content.default_tags,
            auto_cover=upload_content.auto_cover,
            cover_source=upload_content.cover_source,
            season_id=upload_content.season_id,
            templates=templates,
        )
        uploader = UploaderConfig(bilibili=bilibili)
        
        # GPU 配置
        gpu_data = data.get('gpu', {})
        gpu = GPUConfig(
            device=gpu_data.get('device', 'auto'),
            allow_cpu_fallback=gpu_data.get('allow_cpu_fallback', False),
            min_free_memory_mb=gpu_data.get('min_free_memory_mb', 2000)
        )
        
        # 并发配置
        concurrency = ConcurrencyConfig(**data['concurrency'])
        
        # 日志配置
        logging = LoggingConfig(**data['logging'])
        
        # LLM配置（统一管理，自动设置环境变量）
        llm_data = data.get('llm', {})
        llm = LLMConfig(
            api_key=llm_data.get('api_key', ''),
            base_url=llm_data.get('base_url', ''),
            model=llm_data.get('model', ''),
        )
        
        # 代理配置（全局统一管理，自动设置环境变量）
        proxy_data = data.get('proxy', {})
        proxy = ProxyConfig(
            http_proxy=proxy_data.get('http_proxy', '')
        )
        
        # Web UI 配置（可选，有默认值）
        web_data = data.get('web', {})
        web = WebConfig(
            host=web_data.get('host', '0.0.0.0'),
            port=web_data.get('port', 8080),
        )
        
        return cls(
            storage=storage,
            downloader=downloader,
            asr=asr,
            translator=translator,
            embedder=embedder,
            uploader=uploader,
            gpu=gpu,
            concurrency=concurrency,
            logging=logging,
            llm=llm,
            proxy=proxy,
            web=web,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        def convert(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj
        return convert(self)
    
    def to_yaml(self, yaml_path: str) -> None:
        """保存到YAML文件"""
        path = Path(yaml_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
    
    def ensure_directories(self):
        """确保所有必要的目录存在"""
        dirs = [
            self.storage.work_dir,
            self.storage.output_dir,
            Path(self.storage.database_path).parent,
            self.storage.models_dir,
            self.storage.cache_dir,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_whisper_models_dir(self) -> str:
        """获取 Whisper 模型存储目录的完整路径"""
        return str(Path(self.storage.models_dir) / self.asr.models_subdir)
    
    def get_translator_model_path(self) -> str:
        """获取翻译模型文件的完整路径"""
        return str(Path(self.storage.models_dir) / self.translator.local.model_filename)
    
    def get_stage_llm_credentials(self, stage: str) -> Dict[str, str]:
        """获取指定阶段的有效 LLM 凭据（api_key, base_url, model）
        
        Resolve 优先级：
        - split:     asr.split        → global llm
        - translate:  translator.llm    → global llm
        - optimize:   translator.llm.optimize → translator.llm → global llm
        
        Args:
            stage: 阶段名称 ("split" | "translate" | "optimize")
            
        Returns:
            {"api_key": str, "base_url": str, "model": str}
            api_key/base_url 为空字符串表示使用全局配置（由 call_llm 的 get_or_create_client 处理）
        """
        global_key = self.llm.api_key
        global_url = self.llm.base_url
        
        if stage == "split":
            return {
                "api_key": self.asr.split.api_key or global_key,
                "base_url": self.asr.split.base_url or global_url,
                "model": self.asr.split.model,
            }
        elif stage == "translate":
            return {
                "api_key": self.translator.llm.api_key or global_key,
                "base_url": self.translator.llm.base_url or global_url,
                "model": self.translator.llm.model,
            }
        elif stage == "optimize":
            # optimize → translator.llm → global
            trans_key = self.translator.llm.api_key or global_key
            trans_url = self.translator.llm.base_url or global_url
            opt = self.translator.llm.optimize
            return {
                "api_key": opt.api_key or trans_key,
                "base_url": opt.base_url or trans_url,
                "model": opt.model or self.translator.llm.model,
            }
        else:
            raise ValueError(f"未知的阶段: {stage}，支持: split, translate, optimize")
    
    def get_optimize_effective_config(self) -> Dict[str, Any]:
        """获取 optimize 阶段的完整有效配置（含继承的 batch_size, thread_num 等）
        
        Returns:
            {"model": str, "api_key": str, "base_url": str,
             "batch_size": int, "thread_num": int}
        """
        creds = self.get_stage_llm_credentials("optimize")
        opt = self.translator.llm.optimize
        parent = self.translator.llm
        return {
            **creds,
            "batch_size": opt.batch_size if opt.batch_size > 0 else parent.batch_size,
            "thread_num": opt.thread_num if opt.thread_num > 0 else parent.thread_num,
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置
    优先级：指定路径 > ./config/config.yaml > ./config/default.yaml
    """
    if config_path:
        return Config.from_yaml(config_path)
    
    # 尝试用户自定义配置
    local_config = Path("config/config.yaml")
    if local_config.exists():
        return Config.from_yaml(str(local_config))
    
    # 使用默认配置文件
    default_config = Path("config/default.yaml")
    if default_config.exists():
        return Config.from_yaml(str(default_config))
    
    # 尝试项目目录的默认配置
    project_default = Path(__file__).parent.parent / "config" / "default.yaml"
    if project_default.exists():
        return Config.from_yaml(str(project_default))
    
    raise RuntimeError("未找到配置文件")
