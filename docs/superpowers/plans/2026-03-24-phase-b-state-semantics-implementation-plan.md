# Phase B State Semantics Implementation Plan

> **For agentic workers:** Execute this plan using the current harness capabilities. Use checkboxes (`- [ ]`) for tracking. If subagents are explicitly requested or clearly beneficial, delegate bounded subtasks; otherwise execute directly.

**Goal:** 收口 VAT 的核心状态语义，统一 `SKIPPED / unavailable / partial_completed / blocked` 在 database、pipeline、web_jobs、services 和 Web 展示中的解释。

**Architecture:** 本阶段不引入新的大框架，而是先建立一套可测试的“阶段已满足”语义，并把它稳定下沉到 database 与 job 聚合层。重点不是扩张抽象，而是让状态解释从分散实现收敛到少数中心逻辑；同时保留对未来进一步细化的空间，不把当前尚未完全看清的业务含义写死。

**Tech Stack:** Python, SQLite, pytest, FastAPI, Click

---

## 范围与原则

- 本阶段优先统一：
  - `SKIPPED` 是否算阶段语义已满足
  - `unavailable` 是否应继续伪装成“全部 completed”
  - `partial_completed` 的计算口径
  - `blocked` 与 `pending` 的边界
  - `web_jobs` 对任务完成度的判定
- 本阶段暂不处理：
  - 引入新的数据库表
  - 整体 UI 展示重做
  - `watch_sessions/watch_rounds` 的最终统一任务模型
- 保守原则：
  - 只在当前已经有足够证据的语义上收口
  - 对尚未完全确定的业务含义，显式标注“仍需二次核查”

## 当前拟定的目标语义

这是当前实施的暂定目标，不是永远不可修改的教条：

1. `SKIPPED`
   - 视为“该阶段语义已满足”，即 satisfied
   - 应参与：
     - `get_pending_steps()`
     - 视频级进度
     - playlist 进度
     - job 完成度判断
   - 但不必强行等价于“真实执行完成”，因此“completed”与“satisfied”仍需概念区分

2. `unavailable`
   - 不再用“所有阶段 completed”伪装
   - 更准确的编码方式是：阶段状态标记为 `SKIPPED`
   - 视频级 / playlist 级聚合时单独统计为 `unavailable`

3. `partial_completed`
   - 基于“已满足阶段数 > 0 且 < 总阶段数，且无失败/运行中”的统一语义计算

4. `blocked`
   - 仅用于展示：前置阶段失败后，后续未执行步骤应显示为 blocked
   - 不写回任务表，仅作为聚合视图状态

## 仍需二次核查的点

- `is_step_completed()` 是否保留“严格 completed”语义，还是过渡为“已满足”；当前更倾向于新增/优先使用“satisfied”语义，而不是偷换名字
- 某些上传/cron 入口是否必须要求“真正 completed”而不能接受 `SKIPPED`
- `unavailable` 视频在个别 CLI 命令筛选里是否应该被彻底排除，而不是只靠任务状态为空/已满足间接表现

## 目标文件结构

**主要文件：**
- Modify: `vat/models.py`
- Modify: `vat/database.py`
- Modify: `vat/pipeline/executor.py`
- Modify: `vat/services/playlist_service.py`
- Modify: `vat/web/jobs.py`
- Modify: `vat/web/routes/videos.py`
- Test: `tests/test_database.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_services.py`
- Test: `tests/test_web_jobs.py`

---

## Chunk 1: 锁定状态语义测试

### Task 1: 为 `SKIPPED` 作为 satisfied 语义写失败测试

**Files:**
- Modify: `tests/test_database.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_web_jobs.py`

- [x] **Step 1: 为 `get_pending_steps()` 增加 skipped 语义测试**
- [x] **Step 2: 为 `batch_get_video_progress()` 增加 skipped 计入进度测试**
- [x] **Step 3: 为 `batch_get_playlist_progress()` / `get_playlist_progress()` 增加 skipped 计入 satisfied 测试**
- [x] **Step 4: 为 `web_jobs` 中“job 请求步骤为 skipped 也算完成”增加测试**

### Task 2: 为 `unavailable` 独立终态写失败测试

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_database.py`

- [x] **Step 1: 将“unavailable 视频标记全部 completed”的测试改成更准确的目标语义**
- [x] **Step 2: 增加 unavailable 视频不计入 completed / partial，而单独计数的测试**

---

## Chunk 2: database 语义收口

### Task 3: 建立 satisfied 状态辅助语义

**Files:**
- Modify: `vat/models.py`
- Modify: `vat/database.py`

- [x] **Step 1: 在核心模型层定义 satisfied 语义 helper/常量**
- [x] **Step 2: `get_pending_steps()` 改为基于最新记录 + satisfied 状态**
- [x] **Step 3: `invalidate_downstream_tasks()` 支持重置 skipped 下游状态**
- [x] **Step 4: `batch_get_video_progress()` / `get_statistics()` / `batch_get_playlist_progress()` 改为按 satisfied 计数**
- [x] **Step 5: `list_videos_paginated()` 的 SQL 状态过滤同步到 satisfied 语义**

---

## Chunk 3: pipeline / services / web_jobs 适配

### Task 4: pipeline 与 service 适配统一语义

**Files:**
- Modify: `vat/pipeline/executor.py`
- Modify: `vat/services/playlist_service.py`

- [x] **Step 1: `unavailable` 视频改为写 `SKIPPED` 而非 `COMPLETED`**
- [x] **Step 2: `VideoProcessor.process()` 跳过已满足阶段时使用统一语义**
- [x] **Step 3: `PlaylistService.get_pending_videos()` / `get_completed_videos()` / `get_playlist_progress()` 改为统一消费收口后的 database 语义**

### Task 5: `web_jobs` 适配统一语义

**Files:**
- Modify: `vat/web/jobs.py`
- Modify: `tests/test_web_jobs.py`

- [x] **Step 1: `_is_video_completed_in_job()` 接受 skipped 作为 satisfied**
- [x] **Step 2: `_determine_job_result()` 接受 requested step 为 skipped 时判为完成**

---

## Chunk 4: Web API 对齐

### Task 6: 收口 `videos` API 的进度口径

**Files:**
- Modify: `vat/web/routes/videos.py`

- [x] **Step 1: 改为复用 `batch_get_video_progress()`，不再用 `COMPLETED / 7`**
- [x] **Step 2: 确保 API 进度与页面主链保持一致**

---

## 验证矩阵

至少运行：

```bash
pytest tests/test_database.py -q
pytest tests/test_pipeline.py -q
pytest tests/test_services.py -q
pytest tests/test_web_jobs.py -q
pytest tests/test_tasks_api.py -q
pytest tests/test_playlists_api.py -q
pytest tests/test_watch_api.py -q
pytest tests/test_bilibili_web_api.py -q
```

Phase B 收尾前，建议再跑：

```bash
pytest tests/test_web_app_lifespan.py tests/test_database.py tests/test_pipeline.py tests/test_services.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_bilibili_web_api.py -q
```

---

## 当前不做

- 不在 Phase B 中统一 `watch_sessions/watch_rounds` 与 `web_jobs`
- 不在 Phase B 中拆 `playlist_service.py` 或 `bilibili.py` 的 workflow
- 不在 Phase B 中重写 UI 页面展示逻辑

---

## 当前结论

`Phase B` 已完成，当前已统一的核心语义有：

- `SKIPPED` 视为阶段语义已满足（satisfied）
- `get_pending_steps()` 基于最新任务记录 + satisfied 语义
- 视频级 / playlist 级进度与统计统一按 satisfied 计数
- `unavailable` 视频不再伪装成“全部 completed”，而是写为 `SKIPPED` 并在聚合层单独统计
- `web_jobs` 对 job 请求步骤的完成判定已接受 `skipped`
- `videos` API 的进度口径已与 database 主聚合一致

本阶段仍明确留给后续的内容：

- `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一模型
- 更深层的 UI 状态展示语义收口
- `playlist_service.py`、`bilibili.py` 的 workflow 边界整理

## 验证结果

已运行：

```bash
pytest tests/test_database.py tests/test_pipeline.py tests/test_services.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_bilibili_web_api.py tests/test_videos_api.py -q
```

结果：

- `242 passed in 15.58s`

Plan saved to `docs/superpowers/plans/2026-03-24-phase-b-state-semantics-implementation-plan.md` and executed.
