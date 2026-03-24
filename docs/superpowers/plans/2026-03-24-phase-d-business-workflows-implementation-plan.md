# Phase D Business Workflows Implementation Plan

> **For agentic workers:** Execute this plan using the current harness capabilities. Use checkboxes (`- [ ]`) for tracking. If subagents are explicitly requested or clearly beneficial, delegate bounded subtasks; otherwise execute directly.

**Goal:** 将 VAT 中最明显的业务 workflow 从错误的层级归位，优先切断 `WatchService -> web.jobs` 反向依赖，并把 Bilibili 高层业务流程从 uploader adapter 文件迁出。

**Architecture:** 本阶段不试图一次性重构全部 `playlist/watch/upload` 业务，而是先处理两个最关键的边界问题：1) Watch 运行时不再直接依赖 Web 基础设施；2) `season_sync / resync_video_info / resync_season_video_infos` 迁到业务层模块，缩小 `uploaders/bilibili.py` 的 adapter 职责。目标是先把 workflow 放回合理层级，而不是继续扩张 route/service/uploader/tools 的混杂。

**Tech Stack:** Python, pytest, SQLite, Click, FastAPI, unittest.mock

---

## 范围与原则

- 本阶段优先处理：
  - `WatchService -> web.jobs` 反向依赖
  - `uploaders/bilibili.py` 中的高层业务流程迁移
  - `cli/tools.py` / `cli/commands.py` / `web/routes/bilibili.py` 对这些 workflow 的调用入口调整
- 本阶段暂不处理：
  - `playlist_service.py` 的整体巨型事务脚本拆分
  - `bilibili.py` 中更深层的多步补偿状态模型
  - `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一模型
- 保守原则：
  - 迁移 workflow，但不一次性改写整个业务语义
  - 能保持现有调用约定的地方优先保守过渡
  - 对尚不清晰的补偿/恢复模型，在文档里标记为后续阶段细化

## 当前拟定方案

1. 新增一个轻量的业务层 submitter 构建模块，让 `WatchService` 依赖“任务提交通道”而不是直接依赖 `JobManager`
2. 新增 `vat/services/bilibili_workflows.py`，承接：
   - `season_sync`
   - `resync_video_info`
   - `resync_season_video_infos`
3. `uploaders/bilibili.py` 中保留兼容 wrapper 或直接改外部调用点，具体以最小破坏为准
4. `cli/tools.py`、`cli/commands.py`、`web/routes/bilibili.py` 改为调用业务层 workflow

## 仍需二次核查的点

- `WatchService` 的默认 submitter 是保守保留在服务层 helper，还是必须所有调用方显式注入
- `season_sync` 等 workflow 是否需要保留在 `uploaders/bilibili.py` 的兼容 wrapper，避免外部调用点一次性全部断开
- `replace_video / sync_season_episode_titles / fix_violation` 是否在本阶段一起迁移，还是继续留到后续补偿设计阶段

## 目标文件结构

**主要文件：**
- Create: `vat/services/process_job_submitter.py`
- Create: `vat/services/bilibili_workflows.py`
- Modify: `vat/services/watch_service.py`
- Modify: `vat/services/__init__.py`
- Modify: `vat/uploaders/bilibili.py`
- Modify: `vat/cli/tools.py`
- Modify: `vat/cli/commands.py`
- Modify: `vat/web/routes/bilibili.py`
- Modify: `tests/test_watch_service.py`
- Modify: `tests/test_tools_job.py`
- Modify: `tests/test_bilibili_batch_resync.py`

---

## Chunk 1: 切断 `WatchService -> web.jobs`

### Task 1: 为可注入的 process submitter 写失败测试

**Files:**
- Modify: `tests/test_watch_service.py`

- [x] **Step 1: 将现有 `JobManager` patch fixture 改为 patch submitter builder**
- [x] **Step 2: 增加 `WatchService` 通过 submitter 提交 process job 的契约测试**
- [x] **Step 3: 确保现有 watch once/retry/full lifecycle 测试仍能覆盖核心行为**

### Task 2: 实现业务层 submitter 边界

**Files:**
- Create: `vat/services/process_job_submitter.py`
- Modify: `vat/services/watch_service.py`
- Modify: `vat/services/__init__.py`
- Modify: `vat/cli/tools.py`
- Modify: `vat/cli/commands.py`

- [x] **Step 1: 新增轻量 `build_process_job_submitter(...)` helper**
- [x] **Step 2: `WatchService` 接收 submitter 或默认通过 helper 构建**
- [x] **Step 3: 删除 `watch_service.py` 对 `vat.web.jobs` 的直接 import**
- [x] **Step 4: `tools_watch` / `commands.watch` 继续保持当前行为**

---

## Chunk 2: 迁移 Bilibili 高层 workflow

### Task 3: 为业务层 workflow 模块写失败测试

**Files:**
- Modify: `tests/test_bilibili_batch_resync.py`
- Modify: `tests/test_tools_job.py`

- [x] **Step 1: 将批量 resync 测试切到新业务层模块**
- [x] **Step 2: 将 `resync_video_info` 相关测试切到新业务层模块**
- [x] **Step 3: 确保 `season_sync` CLI/tools 调用链仍有测试覆盖**

### Task 4: 新建 `bilibili_workflows.py` 并迁移高层流程

**Files:**
- Create: `vat/services/bilibili_workflows.py`
- Modify: `vat/uploaders/bilibili.py`
- Modify: `vat/services/__init__.py`
- Modify: `vat/cli/tools.py`
- Modify: `vat/cli/commands.py`
- Modify: `vat/web/routes/bilibili.py`

- [x] **Step 1: 迁移 `season_sync`**
- [x] **Step 2: 迁移 `resync_video_info`**
- [x] **Step 3: 迁移 `resync_season_video_infos`**
- [x] **Step 4: 更新所有调用点到新模块**
- [x] **Step 5: 视兼容需求决定是否在 `uploaders/bilibili.py` 保留薄 wrapper**

---

## 验证矩阵

至少运行：

```bash
pytest tests/test_watch_service.py -q
pytest tests/test_tools_job.py -q
pytest tests/test_bilibili_batch_resync.py -q
pytest tests/test_bilibili_web_api.py -q
```

收尾前建议补跑：

```bash
pytest tests/test_database.py tests/test_pipeline.py tests/test_cli_process.py tests/test_services.py tests/test_watch_service.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_bilibili_web_api.py tests/test_bilibili_batch_resync.py tests/test_tools_job.py tests/test_videos_api.py -q
```

---

## 当前不做

- 不在 Phase D 里拆 `playlist_service.py`
- 不在 Phase D 里重做 `replace_video / sync_season_episode_titles / fix_violation` 的补偿状态机
- 不在 Phase D 里统一 `watch_sessions/watch_rounds` 与 `web_jobs`

---

## 当前结论

`Phase D` 已完成，当前已落地的边界调整有：

- `WatchService` 不再直接 import `vat.web.jobs`
- 业务层通过轻量 `process_job_submitter` helper 获得 process job 提交通道
- `season_sync / resync_video_info / resync_season_video_infos` 已迁到 `vat/services/bilibili_workflows.py`
- 主调用链（CLI tools / CLI commands / Web bilibili route）已切到新的业务层模块
- `uploaders/bilibili.py` 中保留了薄 wrapper，用于平稳过渡旧引用

这意味着：

- `service -> web` 的直接反向依赖已经切断
- `uploader adapter` 与 `workflow` 的职责开始真正分离
- 后续如果要继续整理 `playlist/watch/upload` 业务链条，重点会转向：
  - `playlist_service.py` 的巨型事务脚本
  - `replace_video / sync_season_episode_titles / fix_violation` 的补偿/恢复模型

## 验证结果

已运行：

```bash
pytest tests/test_watch_service.py tests/test_tools_job.py tests/test_bilibili_batch_resync.py tests/test_bilibili_web_api.py -q
```

结果：

- `77 passed in 10.38s`

Plan saved to `docs/superpowers/plans/2026-03-24-phase-d-business-workflows-implementation-plan.md` and executed.
