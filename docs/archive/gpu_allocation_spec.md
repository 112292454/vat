# GPU分配策略规范

## 1. 设计目标

在多GPU环境下，VAT项目的GPU使用应遵循以下原则：

### 1.1 核心原则
- **独立任务独立选GPU**：彼此不相关的GPU操作应各自选择最合适的GPU
- **自动负载均衡**：每个任务启动时自动选择当前负载最低的GPU
- **避免资源争抢**：不同进程/线程不应同时抢占同一GPU导致OOM

### 1.2 任务相关性分析

| 场景 | 相关性 | GPU分配策略 |
|------|--------|-------------|
| 多视频并行处理 | **不相关** | 每个视频独立选GPU |
| 同一视频的ASR多chunk | **不相关** | 每个chunk独立选GPU |
| 同一视频的chunk合并 | **相关** | 在同一GPU上完成 |
| 多视频的字幕嵌入 | **不相关** | 每个视频独立选GPU |
| ASR + 人声分离并行 | **不相关** | 各自选GPU |

---

## 2. 当前GPU相关模块

### 2.1 ASR模块 (`vat/asr/whisper_wrapper.py`)
- **GPU使用场景**：Whisper语音识别
- **当前实现**：
  - `device="auto"` 时自动选择最优GPU
  - 支持通过 `CUDA_VISIBLE_DEVICES` 外部指定
  - 使用 `vat/utils/gpu.py` 的 `select_best_gpu()` 选择

### 2.2 人声分离 (`vat/asr/vocal_separator.py`)
- **GPU使用场景**：Mel-Band-Roformer模型推理
- **当前实现**：需检查是否支持自动GPU选择

### 2.3 字幕嵌入 (`vat/embedder/`)
- **GPU使用场景**：FFmpeg硬件编码（可选）
- **当前实现**：主要使用CPU，硬件编码为可选

---

## 3. 期望行为

### 3.1 单视频处理
```
视频A处理流程:
  1. 下载 (CPU)
  2. 人声分离 (GPU-X，自动选择)
  3. ASR识别 (GPU-Y，自动选择，可能与2相同)
     - chunk1 -> GPU-Y1
     - chunk2 -> GPU-Y2
     - chunk合并 (GPU-Y1或Y2)
  4. 翻译 (CPU/远程API)
  5. 嵌入 (CPU或GPU硬件编码)
```

### 3.2 多视频并行处理
```
并行处理视频A、B、C:

视频A: ASR -> GPU-0 (负载最低)
视频B: ASR -> GPU-1 (负载最低)
视频C: ASR -> GPU-2 (负载最低)

如果只有2个GPU:
视频A: ASR -> GPU-0
视频B: ASR -> GPU-1
视频C: 等待，或选择负载较低的GPU
```

### 3.3 GPU选择时机
- **应该**：在实际加载模型前选择GPU
- **应该**：考虑当前GPU显存占用
- **不应该**：在进程启动时就固定GPU
- **不应该**：多个进程同时选中同一GPU导致OOM

---

## 4. 实现要求

### 4.1 GPU选择函数 (`vat/utils/gpu.py`)

```python
def select_best_gpu(
    min_free_memory_gb: float = 8.0,
    exclude_gpus: List[int] = None
) -> int:
    """
    选择最优GPU
    
    选择标准：
    1. 排除显存不足的GPU
    2. 选择当前显存利用率最低的GPU
    3. 如果多个GPU利用率相同，选择索引最小的
    """
```

### 4.2 外部GPU指定

当 `CUDA_VISIBLE_DEVICES` 已设置时：
- 应尊重外部指定，不再自动选择
- 这允许调度器预先分配GPU

### 4.3 并发安全

多进程同时选择GPU时的竞态问题：
- **方案A**：使用文件锁
- **方案B**：在选择后立即预占显存
- **方案C**：调度器预先分配（推荐用于批量任务）

---

## 5. 代码审计结果

### 5.1 whisper_wrapper.py ✓
- **位置**: `vat/asr/whisper_wrapper.py::_resolve_device`
- **状态**: 符合规范
- **行为**:
  - 尊重外部 `CUDA_VISIBLE_DEVICES` 设置（如 scheduler 子进程）
  - `device="auto"` 时使用 `select_best_gpu()` 选择最优GPU
  - 最小显存要求: 8GB
  - `_resolve_device` 不再设置 `CUDA_VISIBLE_DEVICES` 环境变量，避免多线程/多视频环境污染
  - 模型加载通过 faster-whisper 的 `device_index` 参数指定目标 GPU

### 5.2 vocal_separation/separator.py ✓ (已修复)
- **位置**: `vat/asr/vocal_separation/separator.py::_resolve_device`
- **原问题**: 硬编码 `cuda:0`，不使用统一的GPU选择逻辑
- **修复**: 与whisper_wrapper保持一致，尊重外部设置，自动选择最优GPU
- **最小显存要求**: 4GB

### 5.3 pipeline/scheduler.py ✓
- **位置**: `vat/pipeline/scheduler.py::MultiGPUScheduler`
- **状态**: 符合规范
- **行为**: 预先分配GPU给子进程，通过 `CUDA_VISIBLE_DEVICES` 控制

### 5.4 chunked_asr.py ✓ (已重构)
- **位置**: `vat/asr/chunked_asr.py::_asr_chunks_multiprocess`
- **原问题**: 使用 ProcessPoolExecutor + 轮询分配 GPU，不检查显存；多视频并发时同 GPU 叠加多个模型导致 OOM
- **修复**: 重构为 per-GPU worker 模型
  - 每个 GPU 一个持久 worker 进程，模型只加载一次
  - 共享 Queue 天然负载均衡
  - worker 加载模型前检查显存是否满足 `min_free_memory_mb`，不足时等待
  - OOM 时 chunk 重回队列重试（最多 3 次），不丢失内容

---

## 6. 验证检查点

### 6.1 单视频验证
- [ ] ASR自动选择GPU
- [ ] 人声分离自动选择GPU
- [ ] 两者可以选择不同GPU

### 6.2 多视频并行验证
- [ ] 多个视频的ASR分布在不同GPU
- [ ] 没有GPU OOM
- [ ] 负载相对均衡

### 6.3 外部指定验证
- [ ] 设置CUDA_VISIBLE_DEVICES后使用指定GPU
- [ ] 不会覆盖外部设置

---

## 6. 后续工作

1. **审计当前代码**：检查所有GPU使用点是否符合规范
2. **添加日志**：GPU选择时输出选择原因
3. **测试验证**：编写多GPU并行测试用例
4. **文档更新**：在用户文档中说明多GPU配置

---

## 7. 相关文件

- `vat/utils/gpu.py` - GPU选择工具函数
- `vat/asr/whisper_wrapper.py` - ASR GPU使用
- `vat/asr/vocal_separator.py` - 人声分离GPU使用
- `vat/pipeline/executor.py` - 流程调度
- `vat/pipeline/scheduler.py` - 多任务调度

