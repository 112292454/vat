# Phase C Control Plane Implementation Plan

> **For agentic workers:** Execute this plan using the current harness capabilities. Use checkboxes (`- [ ]`) for tracking. If subagents are explicitly requested or clearly beneficial, delegate bounded subtasks; otherwise execute directly.

**Goal:** 收口 VAT 的主控制链，消除 `cli process` 与 `pipeline.scheduler` 的双轨编排，让正常视频处理路径共享同一批处理运行时。

**Architecture:** 本阶段不做企业化重构，也不直接拆 `executor.py`。核心做法是抽出一个共享的批处理运行时，让 `process` 命令与 `scheduler` 共同复用，同时保留 `upload-cron` 等特殊路径继续留在 CLI 层。目标是“唯一化正常处理控制链”，而不是把所有命令逻辑都挤进一个超级入口。

**Tech Stack:** Python, Click, multiprocessing, pytest, unittest.mock

---

## 范围与原则

- 本阶段优先处理：
  - `cli process` 与 `scheduler` 的重复编排逻辑
  - 单 GPU / 单进程批处理的共享运行时
  - `playlist_id` / `concurrency` / `fail_fast` / `delay` / `gpu_id` 等参数在统一运行时中的落点
- 本阶段暂不处理：
  - `upload-cron` / `dtime` 的特殊调度分流
  - 多 GPU 架构的大改
  - `executor.py` 细粒度拆分
- 设计原则：
  - 优先共享运行时，不追求强行统一所有入口
  - 不引入重型抽象
  - `schedule_videos()`、`SingleGPUScheduler`、`MultiGPUScheduler` 仍可保留，但应基于共享批处理运行时

## 当前拟定方案

倾向方案：

1. 在 `vat/pipeline/scheduler.py` 中新增共享批处理运行时，例如 `run_video_batch(...)`
2. `SingleGPUScheduler.run()` 调用该运行时
3. `MultiGPUScheduler._worker()` 也通过该运行时顺序处理分配到本 GPU 的 chunk
4. `cli process` 正常处理路径改为调用该运行时，而不再保留自己的一整套 `process_one_video/_run_batch/retry` 实现

这样做的好处：

- 不需要新增大目录
- 控制面真正共享
- 现有 `schedule_videos()` 仍能保持 API 兼容
- 后续 `Phase D/E` 再继续向上/向下收口时，主链已经只有一套正常运行时语义

## 仍需二次核查的点

- 多 GPU 模式下是否要继续保留“每 GPU 一个进程，GPU 内顺序处理”这一策略
- `process` 当前线程池并发与 scheduler 的多进程模型之间，哪些语义必须保留
- `watch` / `web_jobs` 目前是否依赖 `process` 命令现有日志形态中的某些细节

## 目标文件结构

**主要文件：**
- Modify: `vat/pipeline/scheduler.py`
- Modify: `vat/pipeline/__init__.py`
- Modify: `vat/cli/commands.py`
- Modify: `tests/test_cli_process.py`
- Modify: `tests/test_pipeline.py`
- Optional Modify: `vat/pipeline/readme.md`

---

## Chunk 1: 先锁定共享运行时的目标行为

### Task 1: 为共享批处理运行时写失败测试

**Files:**
- Modify: `tests/test_cli_process.py`
- Modify: `tests/test_pipeline.py`

- [x] **Step 1: 增加 `process` 命令复用共享运行时的测试**
- [x] **Step 2: 增加 `SingleGPUScheduler` 复用同一运行时的测试**
- [x] **Step 3: 增加 `playlist_id/concurrency/fail_fast/gpu_id` 透传测试**

---

## Chunk 2: 实现共享批处理运行时

### Task 2: 在 `scheduler.py` 中引入共享运行时

**Files:**
- Modify: `vat/pipeline/scheduler.py`
- Modify: `vat/pipeline/__init__.py`

- [x] **Step 1: 新增公共批处理运行时函数**
- [x] **Step 2: `SingleGPUScheduler.run()` 改为复用该函数**
- [x] **Step 3: `MultiGPUScheduler._worker()` 改为复用该函数**
- [x] **Step 4: 保持当前单 GPU delay 行为与现有测试兼容**

---

## Chunk 3: 收口 `cli process`

### Task 3: 移除 `process` 命令内部重复批处理逻辑

**Files:**
- Modify: `vat/cli/commands.py`
- Modify: `tests/test_cli_process.py`

- [x] **Step 1: 保留 `upload-cron` / `dtime` 特殊路径**
- [x] **Step 2: 正常处理路径改为调用共享批处理运行时**
- [x] **Step 3: 删掉重复的 `process_one_video/_run_batch` 内联实现**
- [x] **Step 4: 保持 `--force` / `--dry-run` / `--fail-fast` / `--delay-start` 当前契约不回退**

---

## 验证矩阵

至少运行：

```bash
pytest tests/test_cli_process.py -q
pytest tests/test_pipeline.py -q
```

收尾前建议补跑：

```bash
pytest tests/test_database.py tests/test_pipeline.py tests/test_cli_process.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_bilibili_web_api.py tests/test_videos_api.py -q
```

---

## 当前不做

- 不在 Phase C 里重写 `schedule_videos()` 的多 GPU 分配策略
- 不在 Phase C 里拆 `executor.py`
- 不在 Phase C 里调整 `watch` 或 `web_jobs` 的任务模型

---

## 当前结论

`Phase C` 已完成，当前已收口的点：

- `cli process` 与 `pipeline.scheduler` 现在共享同一批处理运行时 `run_video_batch(...)`
- `SingleGPUScheduler` 与 `MultiGPUScheduler` 的 worker 已复用该运行时
- `process` 命令保留了：
  - `upload-cron` / `dtime` 特殊路径
  - `--force` / `--dry-run` / `--fail-fast` / `--delay-start` 契约
- 正常视频处理路径不再保留单独的内联批处理实现

这意味着：

- 主控制链在正常处理路径上已经只有一套底层编排逻辑
- 后续如果要继续收口控制面，重点将转到更高层的 workflow 与更低层的技术子域，而不是继续维护 `commands.py` 和 `scheduler.py` 两套批处理代码

## 验证结果

已运行：

```bash
pytest tests/test_cli_process.py tests/test_pipeline.py -q
pytest tests/test_database.py tests/test_pipeline.py tests/test_cli_process.py tests/test_services.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_bilibili_web_api.py tests/test_videos_api.py -q
```

结果：

- `75 passed in 10.78s`
- `262 passed in 14.51s`

Plan saved to `docs/superpowers/plans/2026-03-24-phase-c-control-plane-implementation-plan.md` and executed.
