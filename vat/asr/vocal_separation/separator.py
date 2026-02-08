"""
人声分离器：使用 Mel-Band-Roformer 模型

模型路径约定：
- 模型权重：storage.models_dir/vocal_separator/model.ckpt
- 配置文件：vat/asr/vocal_separation/config.yaml（内置）
"""
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from vat.utils.logger import setup_logger

logger = setup_logger("vocal_separation")

# 模型下载 URL（HuggingFace）
MODEL_DOWNLOAD_URL = "https://huggingface.co/KimberleyJensen/Mel-Band-Roformer-Vocal/resolve/main/MelBandRoformer.ckpt"
MODEL_FILENAME = "model.ckpt"

# 内置配置（与原项目 config_vocals_mel_band_roformer.yaml 一致）
DEFAULT_MODEL_CONFIG = {
    'model': {
        'dim': 384,
        'depth': 6,
        'stereo': True,
        'num_stems': 1,
        'time_transformer_depth': 1,
        'freq_transformer_depth': 1,
        'num_bands': 60,
        'dim_head': 64,
        'heads': 8,
        'attn_dropout': 0,
        'ff_dropout': 0,
        'flash_attn': True,
        'dim_freqs_in': 1025,
        'sample_rate': 44100,
        'stft_n_fft': 2048,
        'stft_hop_length': 441,
        'stft_win_length': 2048,
        'stft_normalized': False,
        'mask_estimator_depth': 2,
        'multi_stft_resolution_loss_weight': 1.0,
        'multi_stft_resolutions_window_sizes': (4096, 2048, 1024, 512, 256),
        'multi_stft_hop_size': 147,
        'multi_stft_normalized': False,
    },
    'training': {
        'instruments': ['vocals', 'other'],
        'target_instrument': 'vocals',
    },
    'inference': {
        'num_overlap': 2,
        'chunk_size': 352800,
    },
}


@dataclass
class VocalSeparationResult:
    """人声分离结果"""
    vocals_path: Optional[Path] = None
    accompaniment_path: Optional[Path] = None
    success: bool = False
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class VocalSeparator:
    """
    人声分离器
    
    使用 Mel-Band-Roformer 模型分离人声和背景音乐
    
    模型路径：
    - 默认：{storage.models_dir}/vocal_separator/model.ckpt
    - 可通过配置 asr.vocal_separation.model_path 自定义
    """
    
    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        model_filename: str = "vocal_separator/model.ckpt",
        device: str = "auto",
        chunk_size: Optional[int] = None,
    ):
        """
        初始化人声分离器
        
        Args:
            models_dir: 模型根目录（storage.models_dir）
            model_filename: 模型文件相对路径
            device: 设备选择 ("auto", "cuda", "cuda:N", "cpu")
            chunk_size: 分块大小（样本数），None 使用默认值
        """
        if models_dir is None:
            # 从配置获取默认模型目录
            try:
                from vat.config import load_config
                config = load_config()
                models_dir = config.storage.models_dir
            except Exception:
                models_dir = Path.home() / ".vat" / "models"
        
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / model_filename
        self.chunk_size = chunk_size or DEFAULT_MODEL_CONFIG['inference']['chunk_size']
        
        # 解析设备
        self._device = self._resolve_device(device)
        
        # 延迟加载
        self._model = None
        self._config = DEFAULT_MODEL_CONFIG
        
        logger.info(f"VocalSeparator 初始化: device={self._device}, model={self.model_path}")
    
    def _resolve_device(self, device: str) -> torch.device:
        """
        解析设备字符串
        
        行为：
        1. 如果CUDA_VISIBLE_DEVICES已被外部设置，使用cuda:0（实际由环境变量控制）
        2. device="auto"时自动选择最优GPU
        3. 支持显式指定cuda:N或cpu
        """
        # 检查外部CUDA_VISIBLE_DEVICES设置
        external_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if external_cuda_devices is not None and external_cuda_devices != "":
            if torch.cuda.is_available():
                logger.info(f"VocalSeparator 使用外部指定的 GPU (CUDA_VISIBLE_DEVICES={external_cuda_devices})")
                return torch.device("cuda:0")  # 由CUDA_VISIBLE_DEVICES控制实际GPU
            else:
                logger.warning("CUDA_VISIBLE_DEVICES 已设置但 CUDA 不可用，回退到 CPU")
                return torch.device("cpu")
        
        if device == "auto":
            if torch.cuda.is_available():
                # 使用统一的GPU选择逻辑
                try:
                    from vat.utils.gpu import select_best_gpu
                    gpu_id = select_best_gpu(min_free_memory_mb=4000)  # 人声分离约需4GB
                    if gpu_id is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        logger.info(f"VocalSeparator 自动选择 GPU {gpu_id}")
                        return torch.device("cuda:0")
                except Exception as e:
                    logger.warning(f"GPU 自动选择失败: {e}，使用 cuda:0")
                return torch.device("cuda:0")
            else:
                logger.warning("CUDA 不可用，使用 CPU 模式（性能会显著下降）")
                return torch.device("cpu")
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("请求 CUDA 设备但 CUDA 不可用")
            return torch.device(device)
        else:
            return torch.device("cpu")
    
    def _download_model(self) -> bool:
        """
        自动下载模型权重文件
        
        使用系统代理设置（如果配置了 proxy）
        
        Returns:
            是否下载成功
        """
        import urllib.request
        import ssl
        
        # 确保目录存在
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始下载人声分离模型...")
        logger.info(f"  URL: {MODEL_DOWNLOAD_URL}")
        logger.info(f"  目标: {self.model_path}")
        
        try:
            # 获取代理设置
            proxy_url = None
            try:
                from vat.config import load_config
                config = load_config()
                if hasattr(config, 'proxy') and config.proxy:
                    proxy_url = config.proxy.http_proxy or config.proxy.https_proxy
            except Exception:
                pass
            
            # 也检查环境变量
            if not proxy_url:
                proxy_url = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY') or \
                           os.environ.get('http_proxy') or os.environ.get('https_proxy')
            
            # 配置代理
            if proxy_url:
                logger.info(f"  使用代理: {proxy_url}")
                proxy_handler = urllib.request.ProxyHandler({
                    'http': proxy_url,
                    'https': proxy_url,
                })
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)
            
            # 创建请求
            request = urllib.request.Request(
                MODEL_DOWNLOAD_URL,
                headers={'User-Agent': 'VAT/1.0'}
            )
            
            # 下载文件（带进度显示）
            temp_path = self.model_path.with_suffix('.tmp')
            
            with urllib.request.urlopen(request, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB
                
                with open(temp_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"  下载进度: {progress:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
            
            # 下载完成，重命名
            temp_path.rename(self.model_path)
            logger.info(f"模型下载完成: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def _load_model(self):
        """延迟加载模型（如果不存在则自动下载）"""
        if self._model is not None:
            return
        
        logger.info(f"加载 Mel-Band-Roformer 模型: {self.model_path}")
        
        # 如果模型不存在，尝试自动下载
        if not self.model_path.exists():
            logger.info("模型文件不存在，尝试自动下载...")
            if not self._download_model():
                raise FileNotFoundError(
                    f"模型文件不存在且下载失败: {self.model_path}\n"
                    f"请手动下载模型权重文件到该路径。\n"
                    f"下载地址: {MODEL_DOWNLOAD_URL}"
                )
        
        try:
            from .mel_band_roformer import MelBandRoformer
            
            # 创建模型
            model_config = self._config['model']
            self._model = MelBandRoformer(**model_config)
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 处理 DataParallel 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self._model.load_state_dict(new_state_dict)
            self._model = self._model.to(self._device)
            self._model.eval()
            
            logger.info(f"模型加载成功: {self._count_parameters()} 参数")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _count_parameters(self) -> str:
        """统计模型参数数量"""
        if self._model is None:
            return "N/A"
        total = sum(p.numel() for p in self._model.parameters())
        if total > 1e6:
            return f"{total / 1e6:.1f}M"
        return f"{total / 1e3:.1f}K"
    
    def separate(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_accompaniment: bool = False,
    ) -> VocalSeparationResult:
        """
        分离音频中的人声
        
        Args:
            audio_path: 输入音频文件路径
            output_dir: 输出目录（默认为输入文件同目录）
            save_accompaniment: 是否保存伴奏
            
        Returns:
            VocalSeparationResult
        """
        import soundfile as sf
        
        start_time = time.time()
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return VocalSeparationResult(
                success=False,
                error_message=f"输入文件不存在: {audio_path}"
            )
        
        if output_dir is None:
            output_dir = audio_path.parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._load_model()
            
            # 读取音频
            logger.info(f"读取音频: {audio_path}")
            mix, sr = sf.read(str(audio_path))
            
            # 检查采样率
            expected_sr = self._config['model']['sample_rate']
            if sr != expected_sr:
                logger.warning(f"采样率不匹配: {sr} != {expected_sr}，将进行重采样")
                import librosa
                mix = librosa.resample(mix.T, orig_sr=sr, target_sr=expected_sr).T
                sr = expected_sr
            
            # 处理单声道
            original_mono = False
            if len(mix.shape) == 1:
                original_mono = True
                mix = np.stack([mix, mix], axis=-1)
            
            # 转换为张量
            mixture = torch.tensor(mix.T, dtype=torch.float32)
            
            # 分离
            logger.info(f"开始分离人声 (长度: {mix.shape[0] / sr:.1f}s)")
            separated = self._demix(mixture)
            
            # 保存结果
            vocals = separated['vocals']
            if original_mono:
                vocals = vocals[0]
            else:
                vocals = vocals.T
            
            vocals_filename = f"{audio_path.stem}_vocals.wav"
            vocals_path = output_dir / vocals_filename
            sf.write(str(vocals_path), vocals, sr, subtype='PCM_16')
            logger.info(f"人声已保存: {vocals_path}")
            
            # 保存伴奏（可选）
            accompaniment_path = None
            if save_accompaniment:
                # 计算伴奏 = 原始 - 人声
                if original_mono:
                    other = mix[:, 0] - vocals
                else:
                    other = mix - vocals
                
                other_filename = f"{audio_path.stem}_accompaniment.wav"
                accompaniment_path = output_dir / other_filename
                sf.write(str(accompaniment_path), other, sr, subtype='PCM_16')
                logger.info(f"伴奏已保存: {accompaniment_path}")
            
            processing_time = time.time() - start_time
            
            return VocalSeparationResult(
                vocals_path=vocals_path,
                accompaniment_path=accompaniment_path,
                success=True,
                processing_time_seconds=processing_time,
            )
            
        except Exception as e:
            logger.error(f"人声分离失败: {e}")
            return VocalSeparationResult(
                success=False,
                error_message=str(e),
                processing_time_seconds=time.time() - start_time,
            )
    
    def _demix(self, mix: torch.Tensor) -> dict:
        """
        执行分离推理
        
        Args:
            mix: 输入混合音频张量 (channels, samples)
            
        Returns:
            分离结果字典 {'vocals': array}
        """
        C = self.chunk_size
        N = self._config['inference']['num_overlap']
        step = C // N
        fade_size = C // 10
        border = C - step
        
        # 边界填充
        if mix.shape[1] > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode='reflect')
        
        # 创建窗口函数
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window = torch.ones(C)
        window[-fade_size:] *= fadeout
        window[:fade_size] *= fadein
        window = window.to(self._device)
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 确定输出形状
                target_instrument = self._config['training'].get('target_instrument')
                if target_instrument is not None:
                    req_shape = (1,) + tuple(mix.shape)
                else:
                    req_shape = (len(self._config['training']['instruments']),) + tuple(mix.shape)
                
                mix = mix.to(self._device)
                result = torch.zeros(req_shape, dtype=torch.float32).to(self._device)
                counter = torch.zeros(req_shape, dtype=torch.float32).to(self._device)
                
                total_length = mix.shape[1]
                i = 0
                
                while i < total_length:
                    part = mix[:, i:i + C]
                    length = part.shape[-1]
                    
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(part, (0, C - length), mode='reflect')
                        else:
                            part = nn.functional.pad(part, (0, C - length), mode='constant', value=0)
                    
                    x = self._model(part.unsqueeze(0))[0]
                    
                    win = window.clone()
                    if i == 0:
                        win[:fade_size] = 1
                    elif i + C >= total_length:
                        win[-fade_size:] = 1
                    
                    result[..., i:i+length] += x[..., :length] * win[..., :length]
                    counter[..., i:i+length] += win[..., :length]
                    i += step
                
                # 平均重叠区域
                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)
                
                # 移除边界填充
                if mix.shape[1] > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]
        
        # 构建结果
        target_instrument = self._config['training'].get('target_instrument')
        if target_instrument is None:
            return {k: v for k, v in zip(self._config['training']['instruments'], estimated_sources)}
        else:
            return {target_instrument: estimated_sources[0]}
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model_path.exists()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'device': str(self._device),
            'loaded': self._model is not None,
            'chunk_size': self.chunk_size,
        }


# ============================================================================
# 便捷函数
# ============================================================================

_separator_instance: Optional[VocalSeparator] = None


def get_vocal_separator(
    device: str = "auto",
    models_dir: Optional[str] = None,
) -> VocalSeparator:
    """
    获取人声分离器实例（单例）
    """
    global _separator_instance
    
    if _separator_instance is None:
        _separator_instance = VocalSeparator(
            device=device,
            models_dir=models_dir,
        )
    
    return _separator_instance


def separate_vocals(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    device: str = "auto",
) -> VocalSeparationResult:
    """
    便捷函数：分离音频中的人声
    """
    separator = get_vocal_separator(device=device)
    return separator.separate(audio_path, output_dir)


def is_vocal_separation_available(models_dir: Optional[str] = None) -> bool:
    """检查人声分离功能是否可用"""
    if models_dir is None:
        try:
            from vat.config import load_config
            config = load_config()
            models_dir = config.storage.models_dir
        except Exception:
            models_dir = str(Path.home() / ".vat" / "models")
    
    model_path = Path(models_dir) / "vocal_separator" / "model.ckpt"
    return model_path.exists()


def check_vocal_separation_requirements(models_dir: Optional[str] = None) -> Tuple[bool, str]:
    """
    检查人声分离的依赖要求
    
    Returns:
        (is_ready, message)
    """
    issues = []
    
    if models_dir is None:
        try:
            from vat.config import load_config
            config = load_config()
            models_dir = config.storage.models_dir
        except Exception:
            models_dir = str(Path.home() / ".vat" / "models")
    
    model_path = Path(models_dir) / "vocal_separator" / "model.ckpt"
    
    if not model_path.exists():
        issues.append(
            f"模型文件不存在: {model_path}\n"
            f"（首次使用时会自动下载，约 3.5GB）"
        )
    
    # 检查依赖库
    try:
        import soundfile
    except ImportError:
        issues.append("缺少依赖: soundfile (pip install soundfile)")
    
    try:
        import librosa
    except ImportError:
        issues.append("缺少依赖: librosa (pip install librosa)")
    
    try:
        from beartype import beartype
    except ImportError:
        issues.append("缺少依赖: beartype (pip install beartype)")
    
    try:
        from rotary_embedding_torch import RotaryEmbedding
    except ImportError:
        issues.append("缺少依赖: rotary-embedding-torch (pip install rotary-embedding-torch)")
    
    try:
        from einops import rearrange
    except ImportError:
        issues.append("缺少依赖: einops (pip install einops)")
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        issues.append("警告: CUDA 不可用（人声分离在 CPU 上会非常慢）")
    
    if issues:
        return False, "\n".join(issues)
    
    return True, "人声分离功能就绪"
