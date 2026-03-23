# VAT Repository Architecture Audit Plan

> **For agentic workers:** Execute this audit using the current harness capabilities. Use checkboxes (`- [ ]`) for tracking. This document is a living audit record and must be updated as understanding improves.

**Goal:** 对 VAT 全仓进行以架构与设计为核心的深度审查，形成可持续引用的设计判断、问题清单与后续重构方向。

**Architecture:** 本轮工作先不直接进入实现重构，而是先建立全仓结构地图、职责边界图和状态语义基线，再按模块逐步审查代码与测试，识别“过度耦合、抽象不足、抽象过度、状态机不清、边界泄漏”这几类核心设计问题。最终产物不是零散 code review，而是一份能够指导后续多轮重构的架构审查基线。

**Tech Stack:** Python 3.10+, Click, FastAPI, SQLite, yt-dlp, faster-whisper, FFmpeg, biliup, pytest

---

## 1. 审查目标

- [ ] 建立当前仓库的真实结构图，而不是沿用历史文档里的理想化描述
- [ ] 识别当前核心架构边界：CLI、Pipeline、WebUI、Database、Services、Uploaders、LLM/ASR
- [ ] 识别设计问题，而不是停留在局部 bug 或零散代码风格
- [ ] 评估当前抽象程度是否适配“少人开发、个人用户为主、仍需扩展”的项目形态
- [ ] 形成后续重构的优先级和约束条件

## 2. 审查方法

- [ ] 先读现有设计/规划/审查文档，区分“历史宣称”与“当前代码现实”
- [ ] 系统梳理目录、模块、文件体量和调用边界
- [ ] 逐模块阅读核心实现与对应测试，而不是只看入口文件
- [ ] 重点检查状态语义、依赖方向、职责分布、配置单一真值来源、错误处理与旁路调用
- [ ] 每完成一个模块，就把结论回写到本文件，避免丢失中间判断

## 3. 当前状态

- 日期：`2026-03-24`
- 当前分支：`master`
- 当前已知背景：
  - `refactor/test-first-hardening` 已合入主线
  - 契约测试和运行时硬化已经做过一轮
  - 高争议结构重构（尤其是大文件拆分）尚未真正开始
- 当前工作目标：
  - 在现有代码现实上重新建立设计判断
  - 输出比历史 `project_review.md` 更可靠的当前架构结论

## 4. 审查记录

### 4.1 已读取的现有文档

- [x] `README.md`
- [x] `docs/archive/project_review.md`
- [x] `docs/superpowers/plans/2026-03-13-vat-test-first-review-and-refactor-plan.md`
- [ ] 继续补读模块级 readme 与后续相关文档

### 4.2 初步结构判断

- 当前仓库不是“没有设计”，而是“已有一轮阶段语义和契约硬化，但结构性分层未完成”。
- 当前最值得优先怀疑的区域不是单点算法，而是：
  - 状态语义是否真的统一
  - Web 是否仍绕过核心控制面
  - 大文件是否承担过多职责
  - 历史文档里的“已完成”是否与当前代码一致

### 4.3 结构体量底图

- 当前统计到的 Python 文件总量约 `142` 个，其中测试约 `52` 个。
- 各目录 Python 代码量概览：
  - `vat/web`: `14` 文件，约 `4920` 行
  - `vat/asr`: `25` 文件，约 `8805` 行
  - `vat/pipeline`: `5` 文件，约 `2957` 行
  - `vat/uploaders`: `5` 文件，约 `2918` 行
  - `vat/cli`: `4` 文件，约 `3687` 行
  - `vat/services`: `3` 文件，约 `1888` 行
  - `vat/utils`: `9` 文件，约 `1869` 行
  - `tests`: `52` 文件，约 `15120` 行
- 当前最大的实现文件：
  - `vat/cli/commands.py`：`2771` 行
  - `vat/pipeline/executor.py`：`2379` 行
  - `vat/uploaders/bilibili.py`：`2367` 行
  - `vat/database.py`：`1565` 行
  - `vat/services/playlist_service.py`：`1250` 行
  - `vat/downloaders/youtube.py`：`1236` 行
  - `vat/embedder/ffmpeg_wrapper.py`：`1100` 行
  - `vat/web/jobs.py`：`1048` 行
  - `vat/config.py`：`1007` 行
- 初步结论：
  - 复杂度明显集中在少数“控制面 / 副作用中心”大文件，而不是均匀散布。
  - 这更像“关键边界没有拆干净”的架构问题，而不是“整体缺少抽象”。

### 4.4 当前待深读模块

- [x] `vat/models.py`
- [x] `vat/database.py`
- [x] `vat/pipeline/`
- [x] `vat/cli/`
- [x] `vat/web/`
- [x] `vat/services/`
- [x] `vat/uploaders/`
- [x] `vat/translator/`
- [x] `vat/asr/`
- [x] `vat/llm/`
- [x] `vat/downloaders/`
- [x] `tests/` 中与以上模块直接绑定的契约测试

## 5. 当前已确认的风险假设

- `executor.py` 很可能仍是当前最大的结构债中心
- Web 路由与页面层可能仍存在绕过统一依赖注入或统一状态聚合的旁路
- `SKIPPED` / `COMPLETED` / `PENDING` / `FAILED` 的跨层语义可能未完全统一
- 现有文档里“已完成”的结论需要逐条重新验证

## 6. 第一轮架构读图结论

### 6.1 设计意图上的层次

当前仓库从文档和模块命名上，明显试图形成以下结构：

1. `config / models / database`
   - 配置、基础数据结构、状态存储
2. `downloaders / asr / translator / embedder / uploaders`
   - 各阶段的技术实现
3. `pipeline`
   - 处理流程编排、阶段调度、进度追踪
4. `services`
   - Playlist / Watch 级别的业务流程
5. `cli`
   - 命令入口
6. `web`
   - 管理层与任务可视化

这套分层方向本身是合理的，而且很符合本项目“CLI 为核心，Web 为增强管理层”的定位。

### 6.2 代码现实中的主要偏差

当前问题不在于“完全没有分层”，而在于以下偏差已经持续积累：

1. **控制面没有收口**
   - `vat/cli/commands.py`
   - `vat/pipeline/executor.py`
   - `vat/web/jobs.py`
   - `vat/services/playlist_service.py`
   这些文件都同时承担了“流程编排 + 业务判断 + 副作用调用 + 状态写回”的多重职责。

2. **Web 边界未真正收干净**
   - `vat/web/app.py` 不只是页面层，还保留了部分 `/api/*` 端点与后台自动同步逻辑。
   - `vat/web/routes/` 已经存在模块化 API，但 `app.py` 仍保留一套旧式 API / 直接 DB 访问入口。
   - `vat/web/routes/bilibili.py` 与 `vat/web/app.py` 里仍可见直接 `Database(...)`，未完全走 `deps.py`。

3. **状态语义集中在 Database + Pipeline，但未成为真正单一真值**
   - `TaskStep` / `TaskStatus` / `DEFAULT_STAGE_SEQUENCE` 已经集中在 `models.py`
   - 但 `pending/completed/skipped/blocked/partial_completed` 的解释仍分散在 `database.py`、`web/app.py`、`web/routes/videos.py`、`web/jobs.py`
   - 这说明“状态模型存在中心定义”，但“状态语义没有中心解释器”

4. **大技术子模块本身边界相对清楚，但控制层把它们重新粘在一起了**
   - `downloaders/base.py` 的抽象层次是合理的
   - `translator/base.py` / `LLMTranslator` 也有较明确的继承关系
   - `uploaders/template.py`、`uploaders/upload_config.py` 是相对干净的小模块
   - 但这些模块被 `executor.py`、`commands.py`、`playlist_service.py` 以较重的 orchestrator 方式耦合起来

### 6.3 当前最像“架构核心”的文件

从职责密度和外部依赖分布看，当前真正的系统控制面主要是：

- `vat/pipeline/executor.py`
- `vat/cli/commands.py`
- `vat/web/jobs.py`
- `vat/services/playlist_service.py`
- `vat/database.py`

这几处的边界设计，基本决定了整个项目的可维护性上限。

### 6.4 当前最像“边界泄漏”的文件

- `vat/web/app.py`
- `vat/web/routes/bilibili.py`
- `vat/web/routes/videos.py`

原因不是文件大，而是它们同时暴露出：

- 旧接口残留
- 状态口径不统一
- 直接依赖底层数据库或上传器
- 与“Web 只是管理层”的设计定位不完全一致

### 6.5 已确认的典型结构偏差

1. **页面层与 API 层没有完全剥离**
   - `vat/web/app.py` 同时承担：
     - HTML 页面路由
     - 部分 `/api/*` 接口
     - 后台自动同步线程启动
     - 直接数据库访问和聚合逻辑
   - 这说明 `routes/` 的模块化并没有完全替代旧入口。

2. **Database 同时承担了过多角色**
   - 持久化层
   - 查询聚合层
   - 部分状态机解释层
   - Playlist / 视频列表页的过滤与排序承载层
   - 这对单人项目不是绝对错误，但当状态语义变复杂时，会让“数据访问”和“业务状态解释”纠缠在一起。

3. **Config 对象不仅是数据，还会主动改进程环境**
   - `LLMConfig.__post_init__()` 会设置环境变量
   - `ProxyConfig` 同样带有环境层副作用
   - 这让 Config 不再只是“可传递的配置数据”，而变成“加载时即执行副作用的运行时引导器”
   - 这种设计在单进程脚本里方便，但在 Web / 多任务 / 测试环境里会增加全局状态污染风险

4. **CLI 与 Web 在复用统一能力，但仍有两套控制路径**
   - 好的一面：Web 的 JobManager 通过子进程复用 CLI，是对的
   - 问题在于：Web 仍保留若干直接访问 Database / Service / Uploader 的旁路
   - 结果是“CLI 是核心”的原则成立了一半，但没有彻底成为唯一控制面

5. **状态语义中心存在，但状态解释没有真正单点收口**
   - `models.py` 提供 `TaskStep` / `TaskStatus`
   - `database.py` 里实现了一部分视频级聚合语义
   - `web/jobs.py` 又实现了一部分 job 级结果语义
   - `web/routes/videos.py` / `web/app.py` 又各自解释一遍进度与状态
   - 这意味着系统已经有“状态名词表”，但还没有“统一状态解释器”

6. **处理调度存在双控制面**
   - `vat/pipeline/scheduler.py` 提供 `SingleGPUScheduler / MultiGPUScheduler / schedule_videos()`
   - 但 `vat/cli/commands.py` 的 `process` 命令又实现了自己的一套视频收集、并发、重试、延迟、fail-fast、processor 构造逻辑
   - 结果是：
     - `download/asr/translate/embed/pipeline` 这些命令大多走 `schedule_videos()`
     - `process` 走自定义批处理路径
   - 这不是简单的“实现细节不同”，而是系统存在两套编排入口，后续语义修复很容易只修到其中一套

7. **`VideoProcessor` 用“修改配置”表达流程语义**
   - `executor.py` 中的 passthrough 不是通过显式的数据流或阶段结果对象表达
   - 而是通过 `_set_passthrough_config()` 临时修改：
     - `asr.split.enable`
     - `translator.llm.optimize.enable`
     - `translator.skip_translate`
   - 这说明“阶段跳跃/直通”这一流程语义没有成为一等对象，而是借配置开关隐式编码

8. **旧 API 与新 API 已实际重复注册**
   - `vat/web/routes/videos.py` 提供了模块化的 `GET /api/videos`
   - `vat/web/app.py` 又额外保留了 `GET /api/videos`
   - 通过直接检查 FastAPI 路由表，已确认 `GET /api/videos` 被注册了两次：
     - `vat.web.routes.videos.list_videos`
     - `vat.web.app.api_list_videos`
   - 这不是“可能遗留”，而是当前真实存在的重复控制入口

### 6.6 业务服务链路的补充结论

并行审查结果已经确认，`Playlist / Watch / Upload` 这条链路存在以下结构性问题：

1. **`WatchService -> web.jobs` 反向依赖**
   - `vat/services/watch_service.py` 直接依赖 `vat/web/jobs.py::JobManager`
   - 与“service 不依赖 web”这一合理边界相冲突

2. **`PlaylistService.sync_playlist()` 已经演化成巨型事务脚本**
   - 它同时承担：
     - playlist 元数据获取
     - tab ID 修正
     - 新旧视频判定
     - 并发抓详情
     - unavailable 剪枝
     - DB 写入
     - 日期插值
     - 异步视频信息翻译触发
     - 顺序索引分配
     - playlist 统计更新
   - 这使它很难再作为“中等粒度 service”演化

3. **`uploaders/bilibili.py` 同时承载平台适配和业务工作流**
   - 同文件既有 `BilibiliUploader` 平台 API 适配
   - 又有：
     - `season_sync()`
     - `resync_video_info()`
     - `resync_season_video_infos()`
   - 甚至会反向实例化 `PlaylistService`
   - 这是明显的层级反转

4. **多阶段远端副作用缺少明确恢复模型**
   - 典型例子：
     - `replace_video()`
     - `sync_season_episode_titles()`
     - `season_sync()`
   - 当前设计默认允许“远端已经部分成功，本地只能报告失败并靠后续诊断修补”
   - 这在单人项目里可以暂时工作，但已经成为结构性风险，而不只是实现问题

5. **Route 层仍在复制 service 层业务**
   - `vat/web/routes/playlists.py` 自己构造 downloader、算 playlist id、查重、提 job
   - `vat/web/routes/watch.py` 直接查 `watch_sessions/watch_rounds` 并发送进程信号
   - `vat/web/routes/bilibili.py` 直接操作 uploader 和批处理流程
   - 这进一步说明 Web 还没有真正降为“薄管理层”

### 6.7 技术子系统的补充结论

并行审查结果显示，技术子系统整体上并非“完全失控”，但存在两个层面的结构问题：

1. **边界较清楚、值得保留的部分**
   - `vat/downloaders/base.py`
   - `vat/downloaders/local.py`
   - `vat/downloaders/direct_url.py`
   - `vat/translator/base.py`
   - `vat/llm/scene_identifier.py`
   - `vat/llm/prompts/*`
   - 这些模块说明仓库内部并不是缺少模块化意识

2. **已经开始次级膨胀的技术中心**
   - `vat/downloaders/youtube.py`
   - `vat/asr/asr_data.py`
   - `vat/asr/whisper_wrapper.py`
   - `vat/translator/llm_translator.py`
   - `vat/embedder/ffmpeg_wrapper.py`
   - 它们的问题不完全相同，但共性是：单个技术文件同时承载“基础适配 + 策略判断 + fallback/retry + 数据变换 + 副作用”

3. **字幕子域边界没有收稳**
   - 当前“字幕/渲染/格式转换”逻辑分散在：
     - `vat/asr/asr_data.py`
     - `vat/asr/subtitle/*`
     - `vat/subtitle_utils/*`
     - `vat/embedder/ffmpeg_wrapper.py`
   - 这说明字幕相关逻辑目前被挂在 ASR 和 Embed 两侧，没有形成稳定子域边界

4. **统一 LLM 调用语义没有完全收口**
   - `vat/llm/client.py::call_llm()` 已经是明显的统一基础设施中心
   - 但并非所有 LLM 使用都走它
   - 这会让：
     - provider 解析
     - retry
     - cache
     - 错误处理
     - endpoint/proxy 语义
     在不同子系统里继续分叉

5. **技术层错误语义不统一**
   - 有些模块走结构化状态对象
   - 有些直接 `raise`
   - 有些 `return False`
   - 有些仍有 `print(...)`
   - 这意味着技术层没有统一的失败语义模型，后续 orchestration 很难干净

6. **媒体基础操作已有重复苗头**
   - ffprobe 信息提取
   - 音频提取
   - 媒体元数据解析
   当前已经不是唯一实现，这会增加后续一致性维护成本

### 6.8 Web 管理层的补充结论

并行审查结果显示，Web 层当前不是单一管理入口，而是多套入口和多套状态系统并存：

1. **Web 启动即触发真实业务执行**
   - `vat/web/app.py::app_lifespan()` 会启动 `_auto_sync_stale_playlists()`
   - 后者直接构造 `Database` 和 `PlaylistService` 并调用 `sync_playlist()`
   - 这意味着仅启动 Web 进程就会触发 downloader / DB 写入副作用
   - 与“Web 只是管理层”的目标明显冲突

2. **Web 内部至少存在三套任务/状态系统**
   - `web_jobs`
   - `watch_sessions / watch_rounds`
   - `playlists.py` / `bilibili.py` 里的进程内状态字典
   - 这不是单纯实现分散，而是同一产品内没有统一任务状态模型

3. **`app.py` 仍兼任页面层、旧 API 层和部分控制层**
   - 已确认 `GET /api/videos` 被重复注册两次：
     - `vat.web.routes.videos.list_videos`
     - `vat.web.app.api_list_videos`
   - 这表明旧 API 入口并未退出，新的 router 分层也未完全取代旧入口

4. **页面 GET 请求本身带副作用**
   - `/tasks`
   - `/tasks/{task_id}`
   - 等页面会在渲染前主动调用 `update_job_status()`、清理孤儿 task 等逻辑
   - 当前读操作与状态修补耦合在一起

5. **依赖注入只覆盖了部分路径**
   - `deps.py` 存在，但 `app.py`、`watch.py`、`bilibili.py` 等仍可见直接构造 `Database(...)` / `JobManager(...)`
   - 说明依赖注入不是稳定边界，只是部分模块采用的约定

6. **Web 与 CLI/pipeline 的运行语义已经部分重复**
   - `tasks.execute` 不只是提交任务
   - 还在 Web 层做：
     - `upload_cron` 可执行性校验
     - `invalidate_downstream_tasks()`
   - 这意味着 pipeline 状态机语义被复制到 Web route

### 6.9 状态机与主控制链的具体问题

1. **`SKIPPED` 不是一等状态**
   - `VideoProcessor.process()` 会把直通阶段写成 `TaskStatus.SKIPPED`
   - 但 `Database.get_pending_steps()` 只把 `COMPLETED` 当作完成
   - `batch_get_video_progress()` 也只统计 `completed`
   - 这说明系统虽然有 `SKIPPED` 状态名，但没有把它稳定纳入统一状态语义

2. **`unavailable` 被编码为“全部完成”**
   - `VideoProcessor.process()` 对不可用视频会把所有阶段直接标记为 `COMPLETED`
   - 这对下游“少做事”是方便的
   - 但会把“未处理但不需要处理”和“真正处理完成”混为一谈

3. **`process` 命令和 `scheduler` 是两套编排实现**
   - `schedule_videos()` 仍然存在并被多个命令使用
   - `cli process` 却自己实现了并发、重试、delay、fail-fast、processor 构造
   - 这使得“主控制链”没有真正唯一入口

4. **`VideoProcessor` 通过修改 config 表达流程控制**
   - 这会让：
     - 阶段跳跃
     - 直通
     - playlist prompt 覆写
   - 都依赖“运行时临时改配置，再在 finally 恢复”
   - 从设计上说，这是把流程语义编码进配置对象，而不是编码进显式的执行模型

5. **Config 加载即带环境副作用**
   - `LLMConfig` 和代理配置在构造时会设置环境变量
   - CLI 还有全局 `CONFIG` 缓存
   - 这意味着“配置对象”和“进程运行时环境”被耦合在一起
   - 对脚本方便，但对 Web / 多任务 / 测试隔离不够干净

## 7. 当前已经可以成立的设计判断

### 7.1 这不是“过度抽象”的系统

当前仓库更接近“抽象层次存在，但控制面过于肥大”，而不是“到处是抽象工厂和泛型框架”。

因此后续设计方向不应是继续加更多抽象壳，而应是：

- 收紧控制面
- 收拢状态语义
- 明确 orchestrator / service / adapter / pure helper 的职责分界

### 7.2 当前最大的设计问题不是算法，而是职责分配

已有很多模块内部实现并不差，但系统级问题在于：

- 入口层知道太多
- orchestrator 直接操作太多细节
- Web 和 CLI 对同一能力的复用边界不够稳定
- Database 被同时当成“状态机中心”“查询层”“聚合器”“部分业务规则承载体”

### 7.3 这个项目适合“有限分层、强控制面、少量关键抽象”的风格

结合项目形态：

- 单人或极少人开发
- 用户多为个人
- 仍需长期维护和逐步扩展

更适合的风格应当是：

- 避免企业化过度抽象
- 但必须保留清楚的控制面和状态语义中心
- 技术实现可以按阶段模块化
- 业务流程和副作用编排必须减少旁路
- 文档应记录“真实架构”和“当前边界”，而不是只写理想结构图

### 7.4 当前不宜采用的极端方向

不适合继续走的两个极端是：

1. **企业化过度抽象**
   - 为每个动作都抽 `Manager / Factory / Provider / Adapter / Context`
   - 引入过多目录层次与协议壳
   - 这会让单人维护成本上升，不符合项目体量

2. **继续让控制面无限增胖**
   - 把更多判断继续堆进 `commands.py` / `executor.py` / `app.py`
   - 短期方便，长期会让状态语义越来越难验证

当前更合理的方向应当是：

- 保持模块数量克制
- 但让“控制面”与“技术实现”分离得更清楚
- 只在真正跨阶段、跨副作用的地方引入抽象
- 让状态语义集中在极少数中心模块中，而不是分散在 UI/API/DB/Job 各处

### 7.5 已经可以明确的重构方向倾向

基于当前审查，后续更可能需要的不是“大规模新框架”，而是以下几类有限重构：

1. **先收口控制面**
   - 目标是把“谁负责编排、谁负责技术实现、谁负责状态解释”说清楚
   - 先消除双入口和旁路，再谈文件拆分

2. **把平台适配和业务工作流分开**
   - 例如 `BilibiliUploader` 应尽量只保留平台 API 适配
   - `season_sync / resync / 违规修复` 这类流程应进入 service/workflow 层

3. **把状态解释从 UI 和 route 中抽离出来**
   - `completed/skipped/pending/blocked/partial_completed` 不应由页面、route、job、database 各解释一遍

4. **保留相对干净的小模块，不做无差别重写**
   - 如 downloader 抽象层
   - translator base
   - template / upload_config
   - 这类模块更适合围绕边界校正，而不是重做

5. **优先收技术子域边界，而不是拆所有大文件**
   - 技术层优先级更高的不是简单按行数拆文件
   - 而是先收稳：
     - 字幕/渲染子域边界
     - 统一 LLM 调用语义
     - 媒体基础操作入口
     - 技术层错误语义模型

6. **`LLMTranslator`、`WhisperASR`、`FFmpegWrapper` 更适合做“有限解耦”而不是企业化重构**
   - 它们已经是重要技术中心
   - 后续应优先拆责任，而不是堆接口壳

7. **Web 层优先目标不是“页面美化”，而是控制边界收口**
   - 先清掉旧 API 和启动副作用
   - 再统一任务状态模型
   - 最后再谈 route/service/job 的进一步拆分

### 7.6 当前已经可以成立的目标架构约束

基于当前审查，后续架构设计至少应满足以下约束：

1. **唯一执行控制面**
   - CLI 仍应是唯一执行控制面
   - Web 通过任务系统复用 CLI
   - Web 不能继续保留“启动即执行真实业务”“route 直接跑核心流程”的第二通道

2. **状态解释单点收口**
   - `models.py` 保留状态名词表
   - 但视频级 / playlist 级 / job 级状态解释不能继续散在：
     - `database.py`
     - `web/jobs.py`
     - `web/app.py`
     - `web/routes/*`
   - 需要一个更集中、可测试的状态聚合/解释边界

3. **平台适配器与业务工作流分离**
   - `YouTubeDownloader` / `BilibiliUploader` 这类平台适配器可以保留平台细节与网络韧性
   - 但 playlist/watch/upload 后处理这类工作流不应继续堆在 adapter 文件里

4. **技术子域边界先收稳，再拆大文件**
   - 先收稳：
     - 字幕/渲染子域
     - 统一 LLM facade
     - 媒体基础操作入口
     - 技术层错误语义
   - 再拆：
     - `executor.py`
     - `ffmpeg_wrapper.py`
     - `llm_translator.py`
     - `whisper_wrapper.py`

5. **少量关键抽象，避免企业化过度设计**
   - 后续如果引入新的边界层，应优先是：
     - workflow / service
     - state aggregator
     - runtime/task submission port
   - 不应为每一层再套更多 manager/factory/provider 壳

### 7.7 当前最务实的重构顺序

如果进入实现阶段，最稳妥的顺序应当是：

1. 清理 Web 旧入口与启动副作用
2. 统一任务状态模型与状态解释口径
3. 收口主控制链，消除 `cli process` 与 `scheduler` 双轨
4. 拉直 uploader / service / workflow / route 的边界
5. 最后沿稳定边界拆大文件

这个顺序比“先拆 executor.py”更符合当前仓库现实，也更适合单人项目逐步推进

## 8. 目标架构草案

### 8.1 总体目标

目标不是把 VAT 重构成企业化分层框架，而是把当前已经存在的层次真正收稳：

- CLI 作为唯一执行控制面
- Web 作为纯管理层
- 状态解释有单点中心
- 平台适配器不再承载业务工作流
- 技术子域边界稳定后再拆大文件

### 8.2 目标控制面边界

- `vat/cli/commands.py`
  - 目标职责：CLI 参数入口、少量前置校验、调用统一运行时入口
  - 不应继续承载一整套独立于 `scheduler` 的批处理编排实现

- `vat/pipeline/*`
  - 目标职责：执行流程编排、阶段调度、阶段进度追踪
  - `executor.py` 未来应逐步退化为 orchestration 核心，而非技术细节集合

- `vat/web/app.py`
  - 目标职责：FastAPI 应用组合根、页面路由、模板环境、middleware、异常处理
  - 不应继续承载：
    - 重复 `/api/*`
    - 自动同步线程
    - 页面请求中的状态修补

- `vat/web/routes/*`
  - 目标职责：HTTP 控制器
  - 仅做：
    - 参数解析/校验
    - 调 service/workflow 或提交 job
    - 返回 HTTP 响应
  - 不应复制 pipeline/CLI 状态机语义

- `vat/web/jobs.py`
  - 目标职责：唯一的 Web 后台任务引擎
  - 管：
    - `web_jobs`
    - 子进程启动/取消
    - 日志解析
    - 后台任务状态收敛
  - 不负责 playlist/watch/upload 的领域规则

### 8.3 目标业务边界

- `services`
  - 目标职责：本地业务协调
  - 例如：
    - playlist 本地同步语义
    - watch 策略与筛选
  - 不应反向依赖 Web 层

- `workflows`（后续可新增，数量应克制）
  - 目标职责：跨系统、跨副作用、需要恢复能力的业务流程
  - 优先候选：
    - `playlist_sync_workflow`
    - `season_sync_workflow`
    - `video_info_resync_workflow`
    - `violation_fix_workflow`
  - 这层不是为了企业化，而是为了把“多步远端副作用流程”从 adapter/service/route/tools 中拉出来

- `workflows`（未来可新增，数量应克制）
  - 目标职责：跨系统、跨副作用、可恢复的业务流程
  - 适合承接：
    - season sync
    - 视频信息 resync
    - 违规修复
    - 需要多步远端副作用的 upload 后治理流程

- `uploaders`
  - 目标职责：平台适配器
  - `BilibiliUploader` 只保留：
    - 认证
    - 单次远端 API 调用
    - 平台数据格式转换
  - 不再承载 playlist/upload 业务工作流

- `cli/tools.py`
  - 目标职责：长任务入口适配
  - 输出标准化进度
  - 不再自己持有核心业务策略

### 8.4 目标技术子域边界

- `downloaders`
  - 只负责获取原始视频产物和原始平台元数据
  - 不再承载场景识别、视频信息翻译等后置增强

- `asr`
  - 只负责音频/视频到时间轴文本片段
  - 保留：
    - Whisper runtime
    - chunking
    - postprocessing
    - vocal separation
  - 不再承载字幕渲染和样式逻辑

- `subtitle` 子域（未来更适合由 `vat/subtitle_utils` 扩展承接）
  - 负责：
    - 字幕文档模型
    - 编解码
    - 对齐
    - 样式/排版
    - ASS 渲染
  - 当前分散在 `asr/asr_data.py`、`asr/subtitle/*`、`subtitle_utils/*`、`embedder/*` 的逻辑应逐步收拢

- `translator`
  - 负责字幕文本处理策略
  - `LLMTranslator` 未来应拆责任，但不需要企业化重构

- `llm`
  - 负责统一模型调用基础设施
  - 应形成真正统一的 facade，让：
    - `SceneIdentifier`
    - `VideoInfoTranslator`
    - `LLMTranslator`
    走同一套 provider/retry/cache/diagnostics 语义

- `embedder`
  - 只负责视频与现成字幕产物的合成
  - 不再吞掉字幕样式构造和通用媒体基础能力

- `media` 基础层（未来可用小包或小模块形式出现）
  - 统一：
    - ffprobe 解析
    - 音频提取
    - 通用媒体元数据读取
  - 不必一开始就形成庞大新子系统，但应避免继续重复实现

### 8.5 目标状态边界

- `models.py`
  - 保留状态名词表

- 新的目标不是把所有状态解释继续堆进 `database.py`
  - 更合理的是形成一个轻量状态聚合/解释边界
  - 它统一产出：
    - 视频级状态视图
    - playlist 级状态视图
    - job 级状态视图

- `SKIPPED`
  - 需要被明确定义为：
    - 是不是“可下游消费”
    - 算不算“该阶段语义已满足”
    - 在进度、pending 判定、playlist 统计中是否等价于 completed
  - 当前未统一，后续必须先收口

- `unavailable`
  - 不应继续简单编码为“所有阶段 completed”
  - 至少在聚合层需要能区分：
    - 真完成
    - 不可处理但终止

- `web` 任务状态
  - Web 层目标上只保留两层状态：
    - 通用后台任务状态：`web_jobs`
    - 领域运行状态：视频阶段状态 / watch session 状态
  - 应移除以内存 dict 充当真值来源的方式
  - 如果后续只靠 `task_params` 查询太弱，可以为 `web_jobs` 增加轻量 scope 字段，但不应再分裂出更多平行状态系统

### 8.6 目标技术子域落点

- `downloaders`
  - 保留现有抽象层次
  - 未来只需把 `youtube.py` 内部按：
    - metadata 提取
    - 下载重试
    - 直播策略
    做有限拆分

- `asr`
  - 保留语音识别主域
  - `whisper_wrapper.py` 更适合做有限解耦，而不是彻底打散

- `subtitle_utils`
  - 更适合扩展成真正的字幕子域中心
  - 优先承接：
    - 编解码
    - 时间对齐
    - 样式/排版
    - ASS 渲染

- `llm`
  - 目标是形成统一 facade
  - 让：
    - `SceneIdentifier`
    - `VideoInfoTranslator`
    - `LLMTranslator`
    统一走同一调用入口

- `embedder`
  - 专注视频与字幕产物合成
  - `ffmpeg_wrapper.py` 未来可按：
    - ffprobe/media info
    - NVENC/session
    - soft/hard embed
    做有限拆分

- `media` 基础层（形式可以很轻）
  - 当前更像未来的小型共享能力层
  - 先统一：
    - ffprobe
    - extract_audio
  - 不必一开始就扩成大子系统

## 9. 迁移方向建议

### 9.1 第一阶段：控制面收口

- 清理 `app.py` 的旧 `/api/*` 与自动同步线程
- 统一 Web 任务状态模型
- 去掉内存状态真值
- 让页面层停止承担状态修补副作用

### 9.2 第二阶段：主控制链收口

- 消除 `cli process` 与 `scheduler` 双轨
- 把阶段跳跃/直通从“改配置”逐步转向更显式的执行语义
- 统一 `SKIPPED / unavailable / partial_completed` 的状态解释

### 9.2.1 2026-03-24 进展更新

`Phase B` 已实际完成以下收口：

- `SKIPPED` 已被正式纳入“阶段语义已满足（satisfied）”语义
- `Database.get_pending_steps()` 现在基于最新任务记录，并接受 `completed/skipped`
- 视频级 / playlist 级聚合与统计已统一按 satisfied 计数
- `unavailable` 视频在 pipeline 中不再伪装成“全部 completed”，而是写为 `SKIPPED`，并在聚合层单独计数
- `web_jobs` 对请求步骤完成度的判定已接受 `skipped`
- `videos` API 的进度口径已切换到统一的 `batch_get_video_progress()` 聚合

仍然留待后续阶段的点：

- `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一模型
- UI 层状态文案与视觉语义的最终收口
- `playlist_service.py` / `bilibili.py` 中更深层的 workflow 边界整理

### 9.3 第三阶段：业务工作流归位

- 把 `BilibiliUploader` 中的高层业务流程迁出
- 切断 `WatchService -> web.jobs` 反向依赖
- 收掉 route 层的 service 逻辑复制
- 给多阶段远端副作用流程建立最小恢复模型

### 9.4 第四阶段：技术子域收口

- 统一 LLM facade
- 收字幕子域
- 收媒体基础操作
- 之后再按稳定边界拆：
  - `executor.py`
  - `ffmpeg_wrapper.py`
  - `llm_translator.py`
  - `whisper_wrapper.py`

## 10. 实施级迁移计划

这一节不是“最终唯一方案”，而是基于当前审查形成的第一版实施路径。

原则：

- 优先处理控制边界和状态语义
- 不提前做美化式拆文件
- 到达具体模块时，允许根据进一步深读结果调整细节
- 凡是本节明确标记“仍需二次核查”的点，在真正动手前必须重新确认

### 10.1 Phase A: Web 控制边界收口

**目标**

- 让 Web 真正退回管理层
- 去掉重复 API 与启动副作用
- 为后续统一任务状态模型创造稳定入口

**主要文件**

- `vat/web/app.py`
- `vat/web/routes/videos.py`
- `vat/web/routes/playlists.py`
- `vat/web/routes/watch.py`
- `vat/web/routes/bilibili.py`
- `vat/web/jobs.py`
- `tests/test_web_app_lifespan.py`
- `tests/test_tasks_api.py`
- `tests/test_playlists_api.py`
- `tests/test_watch_api.py`

**建议动作**

1. 删除 `app.py` 中重复的旧 `/api/*`
2. 移除 `app.py` 的 `_auto_sync_stale_playlists()` 自动启动
3. 页面层统一复用 router/API 或共享 helper，不再各自实例化 `JobManager`
4. 将 `_sync_status/_refresh_status/_retranslate_status/_fix_tasks/_sync_tasks` 迁出，统一改为 `web_jobs` 查询

**仍需二次核查**

- 删除 `app.py` 旧 API 后，前端页面是否仍隐式依赖旧响应格式
- `watch` 当前是否存在必须依赖 `watch_sessions/watch_rounds` 的前端交互细节，需要在真正收口前逐项比对

### 10.2 Phase B: 状态语义收口

**目标**

- 统一 `completed/skipped/pending/blocked/partial_completed/unavailable` 的解释
- 让 route/page/job 不再各自解释状态

**主要文件**

- `vat/models.py`
- `vat/database.py`
- `vat/pipeline/executor.py`
- `vat/web/jobs.py`
- `vat/web/app.py`
- `vat/web/routes/videos.py`
- `vat/services/playlist_service.py`
- `tests/test_database.py`
- `tests/test_pipeline.py`
- `tests/test_services.py`
- `tests/test_web_jobs.py`

**建议动作**

1. 明确 `SKIPPED` 是否等价于“该阶段语义已满足”
2. 明确 `unavailable` 是否作为独立终态，而不是伪装成全部 `COMPLETED`
3. 建立统一的状态聚合/解释边界
4. Web 页面、API、Job 结果判定都改为消费同一聚合语义

**仍需二次核查**

- `SKIPPED` 的真正业务期望需要在后续改到具体阶段跳跃路径时再验证一遍
- `partial_completed` 对用户展示的含义，需要结合任务列表页和 playlist 页实际使用场景再收口

### 10.3 Phase C: 主控制链收口

**目标**

- 消除 `cli process` 与 `scheduler` 双控制面
- 让执行控制面回到单一路径

**主要文件**

- `vat/cli/commands.py`
- `vat/pipeline/scheduler.py`
- `vat/pipeline/executor.py`
- `tests/test_cli_process.py`
- `tests/test_pipeline.py`

**建议动作**

1. 确定 `process` 与 `schedule_videos()` 的唯一关系
2. 将并发、fail-fast、delay、retry 等控制语义收回单一路径
3. 让 `commands.py` 更多只做参数入口与前置校验

**仍需二次核查**

- 多 GPU 场景下，`process` 现在的线程并发模型与 `scheduler` 的多进程模型是否需要保留两种能力，还是可以统一成一个更小的运行时入口

### 10.4 Phase D: 业务工作流归位

**目标**

- 把 workflow 从 adapter/service/route/tools 的混杂状态中拉出来
- 让 uploader 和 route 重新变薄

**主要文件**

- `vat/services/playlist_service.py`
- `vat/services/watch_service.py`
- `vat/uploaders/bilibili.py`
- `vat/cli/tools.py`
- `vat/web/routes/playlists.py`
- `vat/web/routes/watch.py`
- `vat/web/routes/bilibili.py`
- `tests/test_services.py`
- `tests/test_watch_service.py`
- `tests/test_bilibili_*`
- `tests/test_season_*`
- `tests/test_tools_job.py`

**建议动作**

1. 切断 `WatchService -> web.jobs` 反向依赖
2. 将 `season_sync / resync_video_info / resync_season_video_infos` 迁离 `uploaders/bilibili.py`
3. 让 `route` 只做 HTTP 适配
4. 让 `cli tools` 回到任务入口适配，而不是业务策略中心

**仍需二次核查**

- 是否新增 `workflows/` 目录，还是先以 `services/*_workflow.py` 形式过渡，更适合当前仓库风格
- `playlist_service.py` 中哪些 helper 保留在 service，哪些必须迁到 workflow，需要结合实际代码切面再做一次细分

### 10.5 Phase E: 技术子域收口

**目标**

- 先稳定技术子域边界
- 再按边界有限拆分大文件

**主要文件**

- `vat/llm/client.py`
- `vat/llm/scene_identifier.py`
- `vat/llm/video_info_translator.py`
- `vat/translator/llm_translator.py`
- `vat/asr/asr_data.py`
- `vat/asr/subtitle/*`
- `vat/subtitle_utils/*`
- `vat/embedder/ffmpeg_wrapper.py`
- `vat/asr/whisper_wrapper.py`
- `vat/downloaders/youtube.py`

**建议动作**

1. 建立统一 LLM facade
2. 让 `SceneIdentifier` / `VideoInfoTranslator` / `LLMTranslator` 走统一调用入口
3. 收拢字幕子域到更稳定的中心
4. 统一 ffprobe / extract_audio 等媒体基础操作
5. 最后对膨胀文件做有限拆分

**仍需二次核查**

- 字幕子域最终是扩展 `vat/subtitle_utils/` 还是新建更明确的 `vat/subtitles/`，要在实际迁移前再判断一次，避免过早定目录
- `LLMTranslator` 的拆分粒度要控制，不能把单人项目重构成企业式策略工厂

## 11. 当前已明确不建议的做法

- 不建议现在就先拆 `executor.py`
- 不建议在状态语义未收口前改页面展示逻辑
- 不建议先把所有技术大文件按行数机械拆开
- 不建议引入过多新的抽象层名词
- 不建议把“仍需二次核查”的问题提前写死为最终实现

## 12. 后续文档维护方式

- 当前最详细的全仓架构审查记录保留在本文件
- 如果进入具体实施，应再为每个阶段单独生成更细的实施计划
- 若到某个模块时发现本轮审查未覆盖到位，应直接在对应计划中显式写出“需补查点”，而不是假设当前判断已经充分

## 8. 下一步

- [ ] 输出仓库级模块地图与文件体量分布
- [ ] 从 `models.py -> database.py -> pipeline/ -> cli/ -> web/` 主链开始深读
- [ ] 回写第一轮“当前架构真实样貌”结论

## 9. 目标 Web 架构建议

### 9.1 目标原则

Web 层继续坚持当前项目最合理的总体定位：

- CLI / pipeline 是核心执行面
- Web 是管理层和可视化层
- 长耗时或多副作用操作优先通过 `JobManager -> CLI/tools` 路径执行
- 页面层不主动触发业务副作用
- Web 内部只保留一套任务状态模型和一套进度口径

这不是要把 Web 做成企业化分层系统，而是要把现在已经出现的多控制面和旁路收回来。

### 9.2 各模块的目标职责

#### `vat/web/app.py`

应当只承担：

- FastAPI 应用创建
- middleware / exception handler / template filter 注册
- router 注册
- 页面路由（HTML 页面）
- 少量纯 UI 辅助入口
  - 如确有必要保留：缩略图服务、上传文件临时落盘

不应继续承担：

- 旧 `/api/*` JSON 接口
- 启动即执行的后台业务同步
- 直接 `Database(...)` / `JobManager(...)` 进行控制面逻辑
- 任务状态收敛和孤儿任务清理这类控制动作

结论：
- `app.py` 应成为组合根和页面层，而不是第二套 API 控制器。

#### `vat/web/routes/`

应当承担：

- JSON API 的 HTTP 适配层
- 参数解析、基础校验、错误转 HTTP 响应
- 调用统一依赖：
  - `get_db()`
  - `get_job_manager()`
  - 少量 service / uploader 只读查询

不应承担：

- 长链路业务编排
- 与 CLI/pipeline 重复的状态机语义
- 内存状态中心

结论：
- `routes/` 应是“薄控制器”，但不需要再额外抽象 controller/service framework。

#### `vat/web/jobs.py`

应当承担：

- `web_jobs` 的唯一持久化任务模型
- 命令构建
- 子进程启动 / 取消 / 存活判断
- 日志解析
- 通用任务状态收敛

不应承担：

- playlist/watch/bilibili 领域语义
- 领域级状态命名（如 syncing/refreshing/masking）
- UI 展示口径

结论：
- `jobs.py` 保持“通用后台任务引擎”身份，不再继续吸收领域逻辑。

#### `vat/web/routes/watch.py` + `vat/services/watch_service.py`

目标职责拆分应当是：

- `routes/watch.py`
  - 启动/停止/删除/查询 watch session 的 API 入口
  - 调用 `JobManager` 提交 `vat tools watch`
  - 读取 `watch_sessions/watch_rounds`

- `services/watch_service.py`
  - 真正的 watch 运行时
  - playlist 轮询
  - round 记录
  - session 状态维护
  - 提交 process job

关键边界调整：

- `WatchService` 不应反向 import `vat.web.jobs`
- 更合理的方式是：
  - 在构造时传入 `job_manager`
  - 或传入一个最小的 `submit_process_job` callable

这已经足够，不需要为了它再引入一整套抽象接口层。

#### `vat/web/routes/bilibili.py`

目标上应拆成两类职责：

1. **轻量管理 / 只读查询**
   - 登录状态
   - 合集列表
   - 配置读写
   - 页面渲染

2. **重副作用操作**
   - fix violation
   - season sync
   - 批量 resync info
   - 可能耗时的远端操作

其中第 2 类应优先走：

- `JobManager -> vat tools ...`

而不是 route 中直接：

- 构造 `Database`
- 构造 `BilibiliUploader`
- 顺手修 metadata
- 再直接调用远端 API

结论：
- `bilibili.py` 应保留为“Web 管理入口”，但重副作用逻辑必须逐步退出 route 层。

### 9.3 哪些旧入口应删除或迁移

#### 应删除

- `vat/web/app.py` 中重复的旧 API：
  - `GET /api/videos`
  - `GET /api/video/{video_id}`
  - `GET /api/stats`

这些接口与 `routes/` 模块化 API 并存，已经构成重复控制入口。

#### 应迁移

- `vat/web/app.py::_auto_sync_stale_playlists()`
  - 不应在 Web 启动时自动执行
  - 应迁移为：
    - 显式 CLI/tools 任务
    - 或受配置控制的可选维护任务
  - 默认建议：先移除自动启动，仅保留手动触发

- `vat/web/app.py` 中与任务页相关的 `JobManager` 直接实例化逻辑
  - 迁移为统一 `get_job_manager()`

- `vat/web/routes/bilibili.py` 中直接 `Database(...)` 的路径
  - 先统一改为 `get_db()` 或 service helper
  - 后续再继续把重副作用操作迁到 tools/job 路径

- `vat/web/routes/playlists.py` / `vat/web/routes/bilibili.py` 的内存状态字典：
  - `_sync_status`
  - `_refresh_status`
  - `_retranslate_status`
  - `_fix_tasks`
  - `_sync_tasks`
  - 全部迁移出内存状态模型

#### 可保留但应重定位

- `vat/web/app.py:/api/thumbnail/{video_id}`
- `vat/web/app.py:/api/videos/upload-file`

这两类更像 UI 辅助入口，不属于旧业务 API。
可选方案：

- 短期保留在 `app.py`
- 中期迁到单独的 `vat/web/routes/ui.py` 或 `vat/web/routes/assets.py`

不必为了这个立即做大拆分。

### 9.4 统一任务状态模型如何收口

#### 目标：两层状态，而不是多套平行状态

只保留两层：

1. **通用后台任务状态**
   - 存在 `web_jobs.status`
   - 枚举固定为：
     - `pending`
     - `running`
     - `completed`
     - `partial_completed`
     - `failed`
     - `cancelled`

2. **领域运行状态**
   - 视频阶段状态：`tasks`
   - watch 运行状态：`watch_sessions/watch_rounds`

除此之外，不再引入新的内存态“真值来源”。

#### 表示规则

- Playlist 同步、刷新、重翻译、season-sync、fix-violation 的“当前任务状态”
  - 应来自 `web_jobs`
  - route 只负责把通用 job 状态翻译成页面所需文案

- 视频处理进度
  - 应只来自 `tasks` 聚合

- Watch 详情页
  - watch 进程生命周期来自 `watch_sessions/watch_rounds`
  - watch 启动任务本身仍可在 `web_jobs` 中存在
  - 最好在 `watch_sessions` 中增加 `job_id` 关联，避免两条链断开

#### 如何替代内存 dict

对单人项目，最务实的方案不是上复杂任务引用框架，而是二选一：

1. **优先推荐**
   - 给 `JobManager` 增加“按 `task_type + task_params` 查找最近任务”的查询 helper
   - route 每次现查 `web_jobs`
   - 删除内存 dict

2. **如果后续发现查询太别扭**
   - 再给 `web_jobs` 增加少量 scope 字段，例如：
     - `scope_type`
     - `scope_id`
   - 仍然不需要额外新表

不建议一上来引入通用 task-ref 框架。

#### 状态映射位置

像 `syncing`、`refreshing`、`masking` 这种状态名可以存在，但只能存在于：

- response adapter
- 前端展示层

不能作为新的持久化状态。

### 9.5 务实的迁移顺序

按单人项目的成本控制，建议分 4 步走，每一步都可独立验收：

#### 第一步：清旧入口，不改语义

- 删除 `vat/web/app.py` 中重复的旧 `/api/*` 路由
- 页面层统一改用 `routes/` 的 API
- `app.py` 中直接构造 `JobManager` 的地方改成复用 `get_job_manager()`

目标：
- 不改变功能
- 先把“同一路由有两份实现”的问题消掉

#### 第二步：去掉内存状态真值

- 替换：
  - `_sync_status`
  - `_refresh_status`
  - `_retranslate_status`
  - `_fix_tasks`
  - `_sync_tasks`
- 改为从 `web_jobs` 查询最近任务
- `watch_sessions` 增加与任务的明确关联（如果需要）

目标：
- Web 重启后状态仍可追踪
- 任务状态只剩一套真值

#### 第三步：收紧 Web 副作用边界

- 移除 `app.py` 启动自动同步
- `bilibili.py` 中长耗时 / 重副作用动作优先迁到 tools/job
- route 中尽量不再直接做：
  - DB 修补
  - uploader 复杂编排
  - 多步远端副作用

目标：
- Web 真正退回管理层
- CLI/tools 成为唯一执行面

#### 第四步：再做小规模文件拆分

等前 3 步稳定后，再考虑拆：

- `vat/web/routes/bilibili.py`
- `vat/web/app.py`

拆分方式也应克制：

- 按职责拆，不按技术层空分目录
- 不为拆分而拆分

例如：

- `bilibili.py`
  - `bilibili_page_and_auth.py`
  - `bilibili_jobs.py`

如果拆分收益不明显，也可以只先收边界，不急着改文件名。

### 9.6 对单人项目最重要的取舍

这套迁移建议刻意避免了这些“企业化动作”：

- 不引入复杂 service container
- 不引入 command bus / event bus
- 不引入 repository/interface 泛滥
- 不新增一套通用任务编排框架

真正需要做的只有三件事：

- 删掉重复入口
- 收回状态真值
- 让 Web 不再偷偷执行核心业务

这三件事做完，Web 架构就会明显比现在稳定，而且不会让代码量和抽象层次失控。
