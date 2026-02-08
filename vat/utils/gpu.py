"""
GPU 工具模块

提供 GPU 信息获取和自动选择功能。
遵循项目 GPU 原则：
- 默认必须使用 GPU，禁止静默回退 CPU
- 支持自动选择最低显存占用的 GPU
- 支持指定 GPU 或显式使用 CPU
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Literal

from vat.utils.logger import setup_logger

logger = setup_logger("gpu")


@dataclass
class GPUInfo:
    """单个 GPU 的信息"""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int  # GPU 核心利用率
    
    @property
    def memory_utilization_percent(self) -> float:
        """显存利用率百分比"""
        if self.memory_total_mb == 0:
            return 100.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


def get_available_gpus() -> List[GPUInfo]:
    """
    获取所有可用 GPU 的信息
    
    Returns:
        GPU 信息列表，按 index 排序
        
    Raises:
        RuntimeError: 如果 nvidia-smi 不可用或执行失败
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi 不可用，无法检测 GPU")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi 执行失败: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi 执行超时")
    
    gpus = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        
        try:
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=int(parts[2]),
                memory_used_mb=int(parts[3]),
                memory_free_mb=int(parts[4]),
                utilization_percent=int(parts[5]) if parts[5] else 0
            ))
        except (ValueError, IndexError) as e:
            logger.warning(f"解析 GPU 信息失败: {line}, 错误: {e}")
            continue
    
    return sorted(gpus, key=lambda g: g.index)


def get_gpu_info(gpu_id: int) -> Optional[GPUInfo]:
    """
    获取指定 GPU 的信息
    
    Args:
        gpu_id: GPU 索引
        
    Returns:
        GPUInfo 或 None（如果 GPU 不存在）
    """
    gpus = get_available_gpus()
    for gpu in gpus:
        if gpu.index == gpu_id:
            return gpu
    return None


def select_best_gpu(
    excluded_gpus: Optional[List[int]] = None,
    min_free_memory_mb: int = 2000
) -> Optional[int]:
    """
    选择最佳 GPU（显存利用率最低）
    
    Args:
        excluded_gpus: 排除的 GPU 索引列表
        min_free_memory_mb: 最小空闲显存要求 (MB)
        
    Returns:
        最佳 GPU 索引，如果没有满足条件的 GPU 则返回 None
    """
    excluded = set(excluded_gpus or [])
    
    try:
        gpus = get_available_gpus()
    except RuntimeError as e:
        logger.error(f"无法获取 GPU 信息: {e}")
        return None
    
    # 过滤满足条件的 GPU
    candidates = [
        gpu for gpu in gpus
        if gpu.index not in excluded
        and gpu.memory_free_mb >= min_free_memory_mb
    ]
    
    if not candidates:
        logger.warning(
            f"没有满足条件的 GPU (排除: {excluded}, 最小空闲显存: {min_free_memory_mb}MB)"
        )
        return None
    
    # 按显存利用率排序，选择最低的
    best = min(candidates, key=lambda g: g.memory_utilization_percent)
    logger.info(
        f"自动选择 GPU {best.index} ({best.name}), "
        f"显存占用: {best.memory_utilization_percent:.1f}%, "
        f"空闲: {best.memory_free_mb}MB"
    )
    return best.index


def is_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    try:
        gpus = get_available_gpus()
        return len(gpus) > 0
    except RuntimeError:
        return False


# GPU 设备标识符类型
GPUDevice = Literal["auto", "cpu"] | str  # "auto", "cpu", "cuda:0", "cuda:1", etc.


def resolve_gpu_device(
    device: GPUDevice,
    allow_cpu_fallback: bool = False,
    min_free_memory_mb: int = 2000
) -> tuple[str, Optional[int]]:
    """
    解析 GPU 设备标识符，返回实际使用的设备
    
    Args:
        device: GPU 设备标识符
            - "auto": 自动选择最低显存占用的 GPU
            - "cpu": 显式使用 CPU
            - "cuda:N": 使用指定 GPU
        allow_cpu_fallback: 是否允许 CPU 回退（默认 False，遵循 GPU 原则）
        min_free_memory_mb: 自动选择时的最小空闲显存要求
        
    Returns:
        (device_str, gpu_id) 元组
        - device_str: 用于 PyTorch 的设备字符串 ("cuda" 或 "cpu")
        - gpu_id: GPU 索引（如果使用 GPU），否则为 None
        
    Raises:
        RuntimeError: 如果无法获取 GPU 且不允许 CPU 回退
    """
    device = device.lower().strip()
    
    # 显式 CPU
    if device == "cpu":
        logger.info("显式使用 CPU 模式")
        return ("cpu", None)
    
    # 自动选择
    if device == "auto":
        gpu_id = select_best_gpu(min_free_memory_mb=min_free_memory_mb)
        if gpu_id is not None:
            return ("cuda", gpu_id)
        
        # 没有可用 GPU
        if allow_cpu_fallback:
            logger.warning("WARNING: 没有可用 GPU，回退到 CPU 模式")
            return ("cpu", None)
        else:
            raise RuntimeError(
                "GPU 自动选择失败：没有满足条件的 GPU。"
                "按 GPU 原则禁止 CPU 回退。"
                "如需使用 CPU，请显式指定 --cpu 参数。"
            )
    
    # 指定 GPU (cuda:N 格式)
    if device.startswith("cuda:"):
        try:
            gpu_id = int(device.split(":")[1])
        except (IndexError, ValueError):
            raise ValueError(f"无效的 GPU 设备格式: {device}，应为 cuda:N")
        
        # 验证 GPU 存在
        gpu_info = get_gpu_info(gpu_id)
        if gpu_info is None:
            available = get_available_gpus()
            available_ids = [g.index for g in available]
            raise RuntimeError(
                f"指定的 GPU {gpu_id} 不存在。可用 GPU: {available_ids}"
            )
        
        logger.info(
            f"使用指定 GPU {gpu_id} ({gpu_info.name}), "
            f"显存占用: {gpu_info.memory_utilization_percent:.1f}%"
        )
        return ("cuda", gpu_id)
    
    # 纯数字（兼容旧格式）
    if device.isdigit():
        gpu_id = int(device)
        gpu_info = get_gpu_info(gpu_id)
        if gpu_info is None:
            raise RuntimeError(f"指定的 GPU {gpu_id} 不存在")
        return ("cuda", gpu_id)
    
    raise ValueError(
        f"无效的设备标识符: {device}。"
        "支持的格式: 'auto', 'cpu', 'cuda:N', 或纯数字"
    )


def set_cuda_visible_devices(gpu_id: Optional[int]) -> None:
    """
    设置 CUDA_VISIBLE_DEVICES 环境变量
    
    注意：此函数应在子进程启动前调用，对已加载的 PyTorch 无效。
    
    Args:
        gpu_id: GPU 索引，None 表示不设置
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.debug(f"设置 CUDA_VISIBLE_DEVICES={gpu_id}")


def get_gpu_for_subprocess(
    device: GPUDevice,
    allow_cpu_fallback: bool = False
) -> dict:
    """
    获取用于启动子进程的 GPU 环境配置
    
    用于 ASR 和 Embed 等需要单独进程的 GPU 密集型任务。
    
    Args:
        device: GPU 设备标识符
        allow_cpu_fallback: 是否允许 CPU 回退
        
    Returns:
        包含环境变量的字典，可用于 subprocess.run() 的 env 参数
    """
    device_str, gpu_id = resolve_gpu_device(
        device,
        allow_cpu_fallback=allow_cpu_fallback
    )
    
    env = os.environ.copy()
    
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif device_str == "cpu":
        # CPU 模式：设置为空字符串禁用 GPU
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    return env


def log_gpu_status():
    """打印所有 GPU 的状态信息（用于调试）"""
    try:
        gpus = get_available_gpus()
        logger.info(f"检测到 {len(gpus)} 个 GPU:")
        for gpu in gpus:
            logger.info(
                f"  GPU {gpu.index}: {gpu.name}, "
                f"显存: {gpu.memory_used_mb}/{gpu.memory_total_mb}MB "
                f"({gpu.memory_utilization_percent:.1f}%), "
                f"核心利用率: {gpu.utilization_percent}%"
            )
    except RuntimeError as e:
        logger.error(f"无法获取 GPU 状态: {e}")
