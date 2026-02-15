"""
utils/gpu.py 单元测试

测试 GPU 设备解析、最佳 GPU 选择（全部通过 mock 避免真实 GPU 依赖）。
"""
import pytest
from unittest.mock import patch
from vat.utils.gpu import (
    resolve_gpu_device, select_best_gpu, GPUInfo,
    set_cuda_visible_devices, check_gpu_free_memory,
)


def _gpu(index, free_mb, used_mb=2000, total_mb=8000, util=50):
    return GPUInfo(
        index=index, name=f"GPU {index}",
        memory_total_mb=total_mb, memory_used_mb=used_mb,
        memory_free_mb=free_mb, utilization_percent=util,
    )


class TestGPUInfo:

    def test_memory_utilization_percent(self):
        g = _gpu(0, free_mb=4000, used_mb=4000, total_mb=8000)
        assert abs(g.memory_utilization_percent - 50.0) < 0.1

    def test_memory_utilization_zero_total(self):
        g = GPUInfo(index=0, name="X", memory_total_mb=0,
                    memory_used_mb=0, memory_free_mb=0, utilization_percent=0)
        assert g.memory_utilization_percent == 100.0


class TestResolvGpuDevice:

    def test_cpu_explicit(self):
        device_str, gpu_id = resolve_gpu_device("cpu", allow_cpu_fallback=True)
        assert device_str == "cpu"
        assert gpu_id is None

    def test_invalid_device(self):
        with pytest.raises(ValueError, match="无效"):
            resolve_gpu_device("invalid_device")

    @patch('vat.utils.gpu.select_best_gpu', return_value=0)
    def test_auto_selects_gpu(self, mock_select):
        device_str, gpu_id = resolve_gpu_device("auto")
        assert device_str == "cuda"
        assert gpu_id == 0

    @patch('vat.utils.gpu.select_best_gpu', return_value=None)
    def test_auto_no_gpu_with_fallback(self, mock_select):
        device_str, gpu_id = resolve_gpu_device("auto", allow_cpu_fallback=True)
        assert device_str == "cpu"
        assert gpu_id is None

    @patch('vat.utils.gpu.select_best_gpu', return_value=None)
    def test_auto_no_gpu_without_fallback(self, mock_select):
        with pytest.raises(RuntimeError, match="GPU 自动选择失败"):
            resolve_gpu_device("auto", allow_cpu_fallback=False)

    @patch('vat.utils.gpu.get_gpu_info')
    def test_cuda_n_format(self, mock_info):
        mock_info.return_value = _gpu(1, free_mb=6000)
        device_str, gpu_id = resolve_gpu_device("cuda:1")
        assert device_str == "cuda"
        assert gpu_id == 1

    @patch('vat.utils.gpu.get_gpu_info', return_value=None)
    @patch('vat.utils.gpu.get_available_gpus', return_value=[])
    def test_cuda_n_nonexistent(self, mock_gpus, mock_info):
        with pytest.raises(RuntimeError, match="不存在"):
            resolve_gpu_device("cuda:99")

    def test_cuda_bad_format(self):
        with pytest.raises(ValueError, match="无效"):
            resolve_gpu_device("cuda:abc")


class TestSelectBestGpu:

    @patch('vat.utils.gpu.get_available_gpus')
    def test_selects_lowest_utilization(self, mock_gpus):
        mock_gpus.return_value = [
            _gpu(0, free_mb=2000, used_mb=6000),
            _gpu(1, free_mb=6000, used_mb=2000),
        ]
        assert select_best_gpu(min_free_memory_mb=1000) == 1

    @patch('vat.utils.gpu.get_available_gpus')
    def test_respects_exclusion(self, mock_gpus):
        mock_gpus.return_value = [
            _gpu(0, free_mb=2000, used_mb=6000),
            _gpu(1, free_mb=6000, used_mb=2000),
        ]
        assert select_best_gpu(excluded_gpus=[1], min_free_memory_mb=1000) == 0

    @patch('vat.utils.gpu.get_available_gpus')
    def test_returns_none_if_no_candidate(self, mock_gpus):
        mock_gpus.return_value = [_gpu(0, free_mb=500)]
        assert select_best_gpu(min_free_memory_mb=2000) is None

    @patch('vat.utils.gpu.get_available_gpus', side_effect=RuntimeError("no nvidia-smi"))
    def test_returns_none_on_error(self, mock_gpus):
        assert select_best_gpu() is None


class TestCheckGpuFreeMemory:

    @patch('vat.utils.gpu.get_gpu_info')
    def test_enough_memory(self, mock_info):
        mock_info.return_value = _gpu(0, free_mb=8000)
        assert check_gpu_free_memory(0, 4000) is True

    @patch('vat.utils.gpu.get_gpu_info')
    def test_not_enough_memory(self, mock_info):
        mock_info.return_value = _gpu(0, free_mb=1000)
        assert check_gpu_free_memory(0, 4000) is False

    @patch('vat.utils.gpu.get_gpu_info', return_value=None)
    def test_gpu_not_found(self, mock_info):
        assert check_gpu_free_memory(99, 4000) is False


class TestSetCudaVisibleDevices:

    def test_sets_env_var(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        set_cuda_visible_devices(2)
        import os
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"

    def test_none_does_nothing(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
        set_cuda_visible_devices(None)
        import os
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "original"
