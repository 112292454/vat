# Phase A Web Boundary Implementation Plan

> **For agentic workers:** Execute this plan using the current harness capabilities. Use checkboxes (`- [ ]`) for tracking. If subagents are explicitly requested or clearly beneficial, delegate bounded subtasks; otherwise execute directly.

**Goal:** 收口 VAT 的 Web 控制边界，移除重复 API 和启动副作用，统一 Web 内部的任务/依赖入口，为后续状态语义收口建立稳定基础。

**Architecture:** 本阶段不试图一次性重写 Web 层，而是先消除最危险的边界泄漏：`app.py` 中的旧 API、自动同步线程、页面层直接构造 `JobManager` 的路径，以及明显依赖内存状态的路由。优先做“控制面收口”而不是“页面重构”或“大文件拆分”，确保 Web 退回到“管理层 + HTTP 适配层”的定位。

**Tech Stack:** Python, FastAPI, Jinja2, SQLite, pytest, httpx, unittest.mock

---

## 范围与原则

- 本阶段优先处理：
  - `vat/web/app.py` 的重复 API 与启动副作用
  - Web 层的统一 `JobManager` 入口
  - Web route 中最明显的内存状态真值依赖
- 本阶段暂不处理：
  - `SKIPPED / unavailable / partial_completed` 的最终状态语义收口
  - `WatchService -> web.jobs` 反向依赖
  - `bilibili` 整体大拆分
- 遇到下列情况时不要强行写死实现：
  - 前端页面是否依赖旧 `/api/*` 响应格式
  - `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一模型
  - `bilibili` 路由中哪些动作必须继续直接执行

## 目标文件结构

本阶段不引入新框架，只做有限收口。

**主要文件：**
- Modify: `vat/web/app.py`
- Modify: `vat/web/routes/tasks.py`
- Modify: `vat/web/routes/playlists.py`
- Modify: `vat/web/routes/watch.py`
- Modify: `vat/web/routes/bilibili.py`
- Modify: `vat/web/deps.py`
- Modify: `vat/web/jobs.py`
- Test: `tests/test_web_app_lifespan.py`
- Test: `tests/test_tasks_api.py`
- Test: `tests/test_playlists_api.py`
- Test: `tests/test_watch_api.py`
- Test: `tests/test_bilibili_web_api.py`

---

## Chunk 1: 移除 `app.py` 重复 API 与启动副作用

### Task 1: 清理旧 `/api/*` 入口

**Files:**
- Modify: `vat/web/app.py`
- Test: `tests/test_web_app_lifespan.py`

- [x] **Step 1: 写/改测试，锁定 `app.py` 不再暴露重复 JSON API**

补充测试，直接读取 `FastAPI` 路由表，确认以下旧入口不再注册：
- `/api/videos`
- `/api/video/{video_id}`
- `/api/stats`

Run:
```bash
pytest tests/test_web_app_lifespan.py -q
```

Expected:
- 先失败，显示路由仍存在或当前断言不满足

- [x] **Step 2: 修改 `vat/web/app.py`，删除重复 JSON API**

删除或迁移以下函数：
- `api_list_videos`
- `api_get_video`
- `api_stats`

保留：
- 页面路由
- `/api/thumbnail/{video_id}`
- `/api/videos/upload-file`
- `/database`

- [x] **Step 3: 运行测试验证通过**

Run:
```bash
pytest tests/test_web_app_lifespan.py -q
```

Expected:
- PASS

### Task 2: 去掉 Web 启动自动同步线程

**Files:**
- Modify: `vat/web/app.py`
- Test: `tests/test_web_app_lifespan.py`

- [x] **Step 1: 改测试，确认 lifespan 不再自动启动同步线程**

把当前仅检查 `_start_auto_sync_thread()` 被调用一次的测试，改为验证：
- app lifespan 启动时不再调用自动同步线程入口

Run:
```bash
pytest tests/test_web_app_lifespan.py -q
```

Expected:
- 先失败，因为当前仍会启动线程

- [x] **Step 2: 修改 `vat/web/app.py`，移除 `_start_auto_sync_thread()` 调用**

做法：
- `app_lifespan()` 只保留真正的应用生命周期管理
- `_start_auto_sync_thread()` / `_auto_sync_stale_playlists()` 暂时保留为未启用 helper 或直接删除

注意：
- 不在本阶段替代为新的后台机制
- 先收掉“启动即执行业务副作用”

- [x] **Step 3: 运行测试验证通过**

Run:
```bash
pytest tests/test_web_app_lifespan.py -q
```

Expected:
- PASS

---

## Chunk 2: 统一 Web 层 `JobManager` / `Database` 获取入口

### Task 3: 页面层统一复用 `get_job_manager()`

**Files:**
- Modify: `vat/web/app.py`
- Modify: `vat/web/routes/tasks.py`

- [x] **Step 1: 梳理 `vat/web/app.py` 中直接构造 `JobManager(...)` 的位置**

检查至少这些页面路径：
- `/video/{video_id}`
- `/tasks`
- `/tasks/{task_id}`

记录后续替换点。

- [x] **Step 2: 将 `vat/web/app.py` 页面层改为复用 `vat.web.routes.tasks.get_job_manager()`**

目标：
- 页面层不再手写 `load_config() + log_dir + JobManager(...)`
- `JobManager` 构造逻辑在 Web 层只保留一份

- [x] **Step 3: 回归 `tasks` / `task detail` / `video detail` 相关 API/页面测试**

Run:
```bash
pytest tests/test_tasks_api.py tests/test_web_app_lifespan.py -q
```

Expected:
- PASS

### Task 4: 明确 `Database` 获取入口，减少页面层/route 旁路

**Files:**
- Modify: `vat/web/app.py`
- Modify: `vat/web/routes/watch.py`
- Modify: `vat/web/routes/bilibili.py`
- Modify: `vat/web/deps.py`

- [x] **Step 1: 先做最小替换，不引入新行为**

将能直接替换的 `Database(...)` 改为：
- 页面层：优先 `get_db()`
- route 层：优先 `Depends(get_db)` 或显式复用 `get_db()`

本阶段先处理最直接的旁路，不强行把所有复杂路径一次性收完。

- [x] **Step 2: 运行已有 Web 相关回归**

Run:
```bash
pytest tests/test_watch_api.py tests/test_bilibili_web_api.py tests/test_database_api.py -q
```

Expected:
- PASS

---

## Chunk 3: 去掉最明显的内存状态真值

### Task 5: 收口 `playlists.py` 的内存状态映射

**Files:**
- Modify: `vat/web/routes/playlists.py`
- Modify: `vat/web/jobs.py`
- Test: `tests/test_playlists_api.py`

- [x] **Step 1: 先写测试，描述目标行为**

新增/调整测试，确认：
- `sync-status`
- `refresh-status`

不依赖进程内 `_sync_status` / `_refresh_status` 的残留内容，而是可直接通过持久化 job 查询得到状态。

Run:
```bash
pytest tests/test_playlists_api.py -q
```

Expected:
- 先失败，暴露当前内存状态依赖

- [x] **Step 2: 在 `jobs.py` 增加按 `task_type + task_params` 查最近任务的 helper**

建议新增能力：
- 按 playlist scope 查询最近 `sync-playlist`
- 按 playlist scope 查询最近 `refresh-playlist`
- 按 playlist scope 查询最近 `retranslate-playlist`

注意：
- 本阶段先用 `task_params` 做 scope，不急着上新表/新字段

- [x] **Step 3: 替换 `playlists.py` 的 `_sync_status/_refresh_status` 真值路径**

目标：
- route 每次现查 job
- 不再依赖进程内字典保存状态

- [x] **Step 4: 运行回归**

Run:
```bash
pytest tests/test_playlists_api.py tests/test_web_jobs.py -q
```

Expected:
- PASS

### Task 6: 评估并处理 `bilibili.py` 的内存状态映射

**Files:**
- Modify: `vat/web/routes/bilibili.py`
- Modify: `vat/web/jobs.py`
- Test: `tests/test_bilibili_web_api.py`

- [x] **Step 1: 先补测试，确认哪些状态查询仍依赖 `_fix_tasks/_sync_tasks`**

Run:
```bash
pytest tests/test_bilibili_web_api.py -q
```

Expected:
- 若缺测试，先补；若已有测试不足以覆盖状态真值来源，先补后再改

- [x] **Step 2: 只处理最直接的 job 状态查询旁路**

注意：
- 本任务允许保守推进
- 如果某些 B 站接口已经明显进入“业务 workflow 归位”范围，可在代码注释或文档中标记为“留待 Phase D”
- 不强求本阶段把整个 `bilibili.py` 清干净

- [x] **Step 3: 运行回归**

Run:
```bash
pytest tests/test_bilibili_web_api.py tests/test_web_jobs.py -q
```

Expected:
- PASS

---

## Chunk 4: 为后续 Phase B 留出稳定入口

### Task 7: 在文档中回写 Phase A 落地结果与未决问题

**Files:**
- Modify: `docs/superpowers/plans/2026-03-24-repo-architecture-audit-plan.md`
- Modify: `docs/superpowers/plans/2026-03-18-refactor-phase-prep.md`

- [ ] **Step 1: 回写已完成项**

记录：
- 哪些旧 API 被删掉
- 自动同步线程是否已移除
- `JobManager` / `Database` 获取入口是否收口
- 哪些内存状态真值已移除

- [ ] **Step 2: 回写仍需二次核查点**

重点记录：
- `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一关系
- `bilibili.py` 中哪些长任务要推迟到 Phase D
- 页面是否仍依赖旧响应格式

---

## 验证矩阵

Phase A 每轮至少跑：

```bash
pytest tests/test_web_app_lifespan.py -q
pytest tests/test_tasks_api.py -q
pytest tests/test_playlists_api.py -q
pytest tests/test_watch_api.py -q
pytest tests/test_bilibili_web_api.py -q
pytest tests/test_web_jobs.py -q
```

在 `Chunk 1` 完成后，建议补跑：

```bash
pytest tests/test_database_api.py -q
```

---

## 当前不做

- 不在 Phase A 里统一 `SKIPPED` / `unavailable` 语义
- 不在 Phase A 里切断 `WatchService -> web.jobs`
- 不在 Phase A 里整体拆分 `bilibili.py`
- 不在 Phase A 里拆 `executor.py`

---

## 当前进度

- [x] 清理 `app.py` 中重复的旧 JSON API
- [x] 移除 Web 启动自动同步线程
- [x] 删除 `app.py` 中已失效的自动同步死代码
- [x] 页面层统一复用 `get_job_manager()`
- [x] `playlists.py` 的同步/刷新状态查询已优先走 `web_jobs`
- [x] `watch.py` 的 `JobManager` 构造逻辑已统一入口
- [x] `bilibili.py` 的修复任务 / season sync 状态查询已优先走 `web_jobs`
- [x] route 级别最直接的 `Database(...)` 旁路已做最小替换
- [ ] `bilibili.py` 内部 helper 仍有 `Database(...)` 直连，留待后续业务 workflow 收口时处理
- [ ] `bilibili.py` 中 `_fix_tasks/_sync_tasks` 仍保留为兼容回退映射，后续可继续移除

## 当前结论

`Phase A` 的最小目标已经达成：

- Web 不再在启动时直接执行真实业务同步
- `app.py` 的重复 JSON API 已移除
- 页面层与部分 route 的 `JobManager` / `Database` 获取入口已收口
- `playlists` / `bilibili` 的关键状态查询已优先基于持久化 `web_jobs`

仍然没有在本阶段解决的问题，明确留给后续阶段：

- `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一模型
- `bilibili.py` 的整体 route/workflow 混杂
- `WatchService -> web.jobs` 反向依赖
- 更深层的状态语义统一（`SKIPPED / unavailable / partial_completed`）

Plan saved to `docs/superpowers/plans/2026-03-24-phase-a-web-boundary-implementation-plan.md` and partially executed.
