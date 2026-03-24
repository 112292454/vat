# VAT Refactor Phase Prep

## 1. 目的

这份文档不是再重复“测试优先”的总原则，而是作为下一阶段真正进入重构前的执行准备文档。

它回答 4 个问题：

1. 当前测试补强是否已经足够支撑进入重构。
2. 哪些模块已经具备“先改实现、再用测试兜底”的条件。
3. 哪些模块仍然不适合直接重构，因为问题已经进入设计层。
4. 后续重构应按什么顺序推进，避免互相污染。

## 2. 当前判断

当前测试补强工作已经达到“可以进入重构准备”的程度。

更准确地说：

- 大部分高风险模块已经有直接的 contract / unit / regression 护栏。
- 全局 `tests --collect-only` 已成功，说明当前没有新的导入级断裂。
- 多组大子集回归已经稳定通过，足以说明当前补强并非碎片化成功。
- 剩余空白大多是长尾异常分支，或者已经进入“需要明确补偿/原子性设计”的区域。

因此，后续的主线应从“继续大面积补测试”切到“在现有测试护栏上进入重构”。

## 3. 已具备重构条件的模块

这些模块已经有足够的直接测试，可以开始针对实现结构做整理：

- `vat/services/playlist_service.py`
  - 已覆盖 sync / refresh / retranslate / recovery helpers / selection helpers。
- `vat/uploaders/bilibili.py`
  - 已覆盖 upload / season api / edit / replace / sorting / sync / wrappers 的大部分直接契约。
- `vat/translator/base.py`
  - 已覆盖零容忍段映射和 cache fallback。
- `vat/translator/llm_translator.py`
  - 已覆盖 cache key / input context / response validation / optimization loop / realign helpers。
- `vat/llm/video_info_translator.py`
  - 已覆盖 roundtrip / helper / retry / failure / fallback。
- `vat/llm/scene_identifier.py`
  - 已覆盖 config load / prompt build / default scene / detect fallback。
- `vat/uploaders/template.py`
  - 已覆盖 render / context / default template / duration formatting。
- `vat/uploaders/upload_config.py`
  - 已覆盖 load / save / update / convenience wrappers。
- `vat/embedder/ffmpeg_wrapper.py`
  - 已补 wrapper 层第一批运行时契约，足以支持下一阶段继续内聚整理。

## 4. 仍不宜直接大改的区域

这些区域虽然已有测试，但问题已经明显进入设计层，需要先写明目标行为再重构：

- `vat/uploaders/bilibili.py` 的远端补偿语义
  - 尤其是 `replace_video()`、`sync_season_episode_titles()`、`season_sync()`。
  - 当前很多测试只是把“失败后返回 False”钉住了，还没有正式的补偿策略。
- `vat/web/jobs.py` / `vat/web/routes/tasks.py`
  - 生命周期和状态机现在已经保守化，但如果继续做结构整理，必须先明确“取消请求”“终态收敛”“孤儿进程恢复”的正式模型。
- `vat/pipeline/executor.py`
  - 当前已经有较强调度护栏，但阶段间原子性、下载契约、上传前 metadata 真值来源仍需小心收敛。

## 5. 进入重构的推荐顺序

推荐按“由低耦合到高耦合、由本地纯逻辑到远端副作用”的顺序推进：

1. `vat/uploaders/template.py`
2. `vat/uploaders/upload_config.py`
3. `vat/llm/video_info_translator.py`
4. `vat/llm/scene_identifier.py`
5. `vat/translator/base.py`
6. `vat/translator/llm_translator.py`
7. `vat/services/playlist_service.py`
8. `vat/embedder/ffmpeg_wrapper.py`
9. `vat/uploaders/bilibili.py`
10. `vat/pipeline/executor.py`
11. `vat/web/jobs.py`
12. `vat/web/routes/tasks.py`

原因：

- 前 1 到 4 项主要是本地 helper / render / prompt / parsing，最适合先做收敛和简化。
- 第 5 到 8 项会影响 pipeline 的中段行为，但副作用仍以本地文件和缓存为主。
- 第 9 项开始进入 B 站远端副作用，必须带着更明确的补偿设计前进。
- 第 10 到 12 项是控制面，不应在底层 still moving 时过早整理。

## 6. 每个模块重构时的固定检查项

进入具体模块重构时，先逐项确认：

1. 模块对外契约是否已经有直接测试。
2. 模块内部 helper 是否还存在明显重复逻辑。
3. 是否存在多处兜底、静默 fallback、下游补洞。
4. 是否存在共享状态污染。
5. 是否存在“远端副作用已发生，但本地无法恢复”的路径。

如果第 5 项成立，优先先写补偿设计说明，再动实现。

## 7. 推荐的第一批重构目标

### 7.1 模板与配置收敛

目标：

- 收敛上传模板上下文的真值来源。
- 收敛 upload config 的默认值与覆盖行为。
- 减少 `render_upload_metadata()` 调用点周围的重复组装逻辑。

建议文件：

- `vat/uploaders/template.py`
- `vat/uploaders/upload_config.py`
- `vat/cli/commands.py`
- `vat/pipeline/executor.py`
- `vat/uploaders/bilibili.py`

### 7.2 翻译链路收敛

目标：

- 把“零容忍”和“缓存语义”进一步固定成实现结构，而不是依赖零散 helper。
- 明确 `reflect`、`context`、`prompt` 三者对输出和缓存的边界。

建议文件：

- `vat/translator/base.py`
- `vat/translator/llm_translator.py`

### 7.3 Playlist 恢复路径收敛

目标：

- 收敛 `sync / refresh / retranslate` 里的重复视频元信息处理逻辑。
- 统一“缺失日期补抓 / 插值回退 / translated 更新”的状态推进点。

建议文件：

- `vat/services/playlist_service.py`

## 8. 当前不建议做的事

- 不建议现在就大改 `web/jobs.py` 的整体模型。
- 不建议在没有补偿设计前重写 `replace_video()` 或 `sync_season_episode_titles()`。
- 不建议现在做大规模跨模块抽象提炼。
- 不建议把“长尾测试继续做到 100% 完美”作为进入重构的前置条件。

## 9. 进入下一阶段的工作方式

推荐采用下面的节奏：

1. 先选一个模块。
2. 写一个该模块的“小型重构准备笔记”。
3. 先跑该模块直接测试和关联回归。
4. 做最小重构。
5. 重新跑同一组测试。
6. 再决定是否扩大范围。

不要多模块并行重构。

## 10. 当前结论

当前已经不再需要继续以“测试补空白”为主线。

后续主线应切到：

`在现有测试护栏上，按模块逐步进入重构`

同时保留两个原则：

- 继续补测试，但只补“重构过程中新增暴露出来的缺口”。
- 遇到真正的设计问题时，先写设计说明，再改代码。

## 11. 2026-03-24 补充结论

在进入具体模块重构前，已完成一轮更偏架构层的全仓深审。结论与本文件的“逐模块进入重构”判断并不冲突，但对重构顺序做了明显修正。

当前更准确的判断是：

- 仓库不是“缺少模块化”，而是“控制面过肥、状态解释分散、边界没有真正收口”。
- 现在最不应该做的事，是直接按行数去拆大文件，尤其是先拆 `executor.py`。
- 当前最该优先收口的是：
  1. Web 控制边界
  2. 状态语义与状态聚合
  3. 主控制链唯一化
  4. 业务 workflow 归位
  5. 技术子域边界收稳

这意味着本文件第 5 节给出的“模块重构顺序”只能视为旧的局部视角顺序，不再适合作为当前主线顺序。

新的推荐主线顺序应当是：

1. Web 控制边界收口
2. 状态语义收口
3. `cli process` / `scheduler` 主控制链收口
4. `Playlist / Watch / Upload` 业务 workflow 归位
5. 技术子域收口
6. 最后才沿稳定边界拆大文件

补充说明：

- 当前最详细的全仓审查与目标架构草案已写入本地审查文档：
  - `docs/superpowers/plans/2026-03-24-repo-architecture-audit-plan.md`
- 该文件位于 `docs/superpowers/` 下，当前被 `.gitignore` 忽略，因此它是本地活文档，不是仓库跟踪文档。
- 如果后续需要把其中稳定结论正式沉淀到仓库，应再择机将其摘要整理回本文件或新的受跟踪设计文档。

## 12. 2026-03-24 Phase B 完成记录

`Phase B` 已完成，核心是“状态语义收口”，而不是新的结构拆分。

已落地的点：

- `SKIPPED` 现在被正式纳入“阶段语义已满足（satisfied）”语义。
- `Database.get_pending_steps()` 改为基于最新任务记录，并接受 `completed/skipped`。
- 视频级 / playlist 级聚合与统计已统一按 satisfied 计数。
- `unavailable` 视频在 pipeline 中不再伪装成“全部 completed”，而是写为 `SKIPPED`，并在聚合层单独统计。
- `web_jobs` 对请求步骤完成度的判定已接受 `skipped`。
- `vat/web/routes/videos.py` 的进度口径已切换到统一的 database 聚合逻辑。

当前判断：

- `Phase A + Phase B` 完成后，Web 入口边界和核心状态语义已经明显比主线 `master` 收敛。
- 后续优先级应当切到：
  1. 主控制链收口（`cli process` vs `scheduler`）
  2. 业务 workflow 归位（尤其 `playlist/watch/upload`）
  3. 技术子域收口（LLM facade / 字幕子域 / 媒体基础操作）

仍需后续阶段处理的点：

- `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一关系
- `playlist_service.py` / `bilibili.py` 的 workflow 边界整理
- UI 层对状态文案和视觉语义的最终收口

## 13. 2026-03-24 Phase C 完成记录

`Phase C` 已完成，核心是“主控制链唯一化”，而不是继续在 CLI 与 scheduler 两边分别维护批处理逻辑。

已落地的点：

- 新增共享批处理运行时，`cli process` 与 `pipeline.scheduler` 现在复用同一套正常处理路径。
- `SingleGPUScheduler` 与 `MultiGPUScheduler` worker 已复用共享运行时。
- `process` 命令保留了 `upload-cron / dtime` 特殊分流，但正常视频处理路径不再保留内联批处理实现。
- 相关 CLI 与 scheduler 契约测试已补齐并通过。

当前判断：

- `Phase A + Phase B + Phase C` 完成后，Web 边界、核心状态语义、正常处理控制链已经形成较稳定的新基线。
- 后续主线应切到：
  1. 业务 workflow 归位（尤其 `playlist/watch/upload`）
  2. 技术子域收口（统一 LLM facade / 字幕子域 / 媒体基础操作）

仍需后续阶段处理的点：

- `WatchService -> web.jobs` 反向依赖
- `playlist_service.py` / `bilibili.py` 中的高层 workflow 混杂
- 技术层几个关键大文件的有限拆分

## 14. 2026-03-24 Phase D 完成记录

`Phase D` 已完成，核心是“业务 workflow 归位”，而不是继续扩大 Web 或 uploader 的控制面。

已落地的点：

- `WatchService` 不再直接依赖 `vat.web.jobs`，改为通过业务层 submitter helper 获取 process job 提交通道。
- `season_sync / resync_video_info / resync_season_video_infos` 已迁到 `vat/services/bilibili_workflows.py`。
- CLI tools、CLI commands、Web Bilibili 路由的主调用链已切到新的业务层 workflow 模块。
- `uploaders/bilibili.py` 目前仅保留兼容 wrapper，不再是这些高层流程的主入口。

当前判断：

- 到 `Phase D` 为止，控制面、状态语义、正常处理主链、以及最明显的业务 workflow 边界都已经完成第一轮收口。
- 后续主线应优先转向技术子域收口：
  1. 统一 LLM facade
  2. 收字幕子域
  3. 收媒体基础操作

仍需后续阶段处理的点：

- `playlist_service.py` 的巨型事务脚本
- `replace_video / sync_season_episode_titles / fix_violation` 的补偿与恢复模型
- `watch_sessions/watch_rounds` 与 `web_jobs` 的最终统一关系

## 15. 2026-03-24 Phase E 完成记录

`Phase E` 已完成第一轮技术子域收口。

已落地的点：

- 新增 `vat/llm/facade.py`，`SceneIdentifier` 与 `VideoInfoTranslator` 已切到统一文本调用入口。
- 新增 `vat/media/`，统一 ffprobe 与音频提取基础操作。
- `BaseDownloader`、`FFmpegWrapper`、`WhisperASR`、`VideoProcessor` 已开始复用媒体基础 helper。
- 新增 `vat/subtitle_utils/codecs.py`，`ASRData` 的文件编解码与保存职责已迁出到 codec 模块，并通过委托保持兼容。

当前判断：

- 到 `Phase E` 为止，这个分支已经完成了控制面、状态语义、业务 workflow、技术子域第一轮收口。
- 后续如果继续深入，重点将不再是“先理顺边界”，而是更深一层的有限重构与补偿模型设计。

仍需后续阶段处理的点：

- `playlist_service.py` 的巨型事务脚本继续收口
- `replace_video / sync_season_episode_titles / fix_violation` 的补偿状态机
- `LLMTranslator / FFmpegWrapper / ASRData` 的下一轮有限拆分

## 16. 第二轮问题的推荐顺序

在第一轮 A-E 收口完成后，后续不应再按“哪个文件大就先拆哪个”推进。

更合理的第二轮顺序应是：

1. **先做远端副作用恢复模型设计**
   - `replace_video`
   - `sync_season_episode_titles`
   - `fix_violation`
   - 原因：这是当前剩余风险最高的业务语义问题

2. **再收口 `playlist_service.py`**
   - 尤其是 `sync_playlist()` 的阶段拆分
   - 原因：它是后续 watch/upload/metadata 行为的中心业务脚本

3. **再处理 `watch_sessions/watch_rounds` 与 `web_jobs` 的最终关系**
   - 原因：这是长期状态模型统一问题，但当前不比前两项更危险

4. **最后再做技术子域第二轮有限拆分**
   - `LLMTranslator`
   - `FFmpegWrapper`
   - `ASRData`
   - 原因：技术层边界第一轮已经收住，后续更多是结构优化，而不是当前最高风险源

一句话：

`第二轮先解业务语义，再解中心脚本，再解长期状态模型，最后才继续拆技术结构`

## 17. 2026-03-25 第二轮当前进展

第二轮目前已经不只是设计稿，最前两项中的一部分已经开始落地：

- `replace_video`
  - 已新增恢复 workflow，并把最关键的 `file_uploaded` 中间态写入 `video.metadata.bilibili_ops.replace_video`
  - uploader 内部已拆出上下文加载 / 替换文件上传 / edit 提交三个步骤

- `sync_season_episode_titles`
  - 已新增恢复 workflow，并把 `original_order / need_update_aids / readded_aids / failed_aids` 写入 `playlist.metadata.bilibili_ops.sync_season_episode_titles`
  - Web `sync-titles` 路由已经切到该 workflow，通过线程池执行
  - 当前采取保守策略：只有在 `season_id -> playlist` 能唯一映射时才持久化恢复状态；否则仍执行，但不落库

仍待继续深入的核心点：

- `fix_violation`
  - 已新增第一层轮次级恢复 wrapper，并接入 `vat tools fix-violation`
  - 当前已经能把 `all_ranges / masked_path / source / replacement_submitted` 这类轮次级状态写回 `video.metadata.bilibili_ops.fix_violation`
  - 当前又进一步把 uploader 内部拆出了：
    - 违规上下文装载
    - 视频源决策
    - 遮罩渲染
  - 这样后续如果要继续细化恢复模型，已经有稳定切口，不必再从单个大函数硬拆
  - 但更细的恢复模型仍未完成；下一步更合理的是继续围绕这些切口补更细阶段状态，而不是继续只在外层加逻辑

因此，第二轮的当前状态可以概括为：

`replace_video 已初步收口，season 标题同步已开始收口，fix_violation 已有第一层状态可见性和内部阶段切口，但更细粒度恢复仍是下一阶段主难点`

## 18. 2026-03-25 playlist_service 当前进展

在第二轮进入 `playlist_service.py` 后，当前先没有急着大拆 `sync_playlist()`，而是先把它最明显的一段阶段边界单独拿出来：

- 已新增 `_plan_sync_candidates()`
  - 负责扫描 playlist entries
  - 区分：
    - `new_videos`
    - `existing_videos`
    - `videos_needing_refresh`
    - `stale_zero_index_existing_videos`
  - 并返回本轮 sync 的候选计划对象

这一步的目的不是“立刻把 `sync_playlist()` 拆碎”，而是先把：

- 候选集规划
- 信息抓取
- unavailable 剔除
- 最终落库

这四段里的第一段单独稳定下来。

当前判断：

- 这是一刀比较安全的切口，因为它只是在把已有局部状态收成显式 helper。
- 随后又继续拆出了：
  - `_prune_sync_candidates_after_fetch()`
  - `_collect_fetch_results()`
  - `_persist_sync_members()`
  - `_apply_fetch_results()`
- 现在 `sync_playlist()` 的前三段已经开始显式化：
  1. 候选集规划
  2. fetch 结果收集
  3. fetch 后裁剪
  4. 成员落库
  5. fetch 结果应用
- 现在 `sync_playlist()` 主流程已经明显更接近“按阶段编排”，而不是在一个函数里同时揉杂规划、抓取、裁剪、落库和回退细节。
- 后续如果继续深入，下一刀更可能落在“playlist 更新收尾 / 索引分配”与前面的阶段边界之间。

## 19. sync_playlist 功能需求与边界基线

进入 `playlist_service.sync_playlist()` 的后续重构前，需要先把我们真正想保住的功能表现写清楚。下面这些不是“当前代码恰好这样”，而是当前应被视为需求基线的行为。

### 19.1 主功能目标

`sync_playlist()` 的目标不是“重建整个 playlist”，而是：

- 以增量方式同步 playlist 成员
- 保持已有 video 记录尽量稳定
- 对需要补抓信息的成员补全 metadata
- 对永久不可用成员做保守清理
- 最终让 playlist 的成员、日期、顺序索引和后续翻译触发保持一致

### 19.2 应保持的核心行为

1. **增量优先，不做激进删除**
   - 新视频应新增到 DB/playlist 关联中
   - 已存在视频不应因为普通 sync 被重建或重复插入

2. **playlist_index 可以更新，但 video 身份不能漂**
   - 已存在成员的 `playlist_index` 可以按本次 playlist 顺序更新
   - 但 video 记录本身应复用原记录

3. **新视频与旧视频的处理策略不同**
   - 新视频：允许创建 video 记录和 playlist 关联
   - 已存在视频：重点是更新关联、补抓 metadata、必要时清理残留半状态

4. **metadata 补抓必须区分永久失败和暂时失败**
   - `unavailable`：可以标记不可用，并参与清理/跳过逻辑
   - `error`：只能做插值/保守回退，不能直接当成永久不可用

5. **永久不可用的新成员不能进入稳定真值**
   - 新发现但已判定永久不可用的视频，不应作为有效新成员落库

6. **中断残留必须能被识别和清理**
   - 已存在但 `upload_order_index=0`、又被判定永久不可用的残留成员，应视为“上次 sync 半状态遗留”
   - 清理时要保守：如果该 video 还属于其他 playlist，只移除当前 playlist 关联，不删全局 video

7. **日期插值只是保守回退，不是真实日期**
   - 插值日期用于维持排序和后续流程可继续推进
   - 一旦后续拿到真实 `upload_date`，必须覆盖插值值并清掉 `upload_date_interpolated`

8. **翻译触发属于成功 metadata 获取后的下游动作**
   - 只有拿到成功 video_info 后才提交异步翻译
   - 不应把翻译逻辑混进候选规划或 unavailable 清理阶段

9. **upload_order_index 的分配是增量式的**
   - 只给本轮真正需要分配的新成员分配
   - 不应因为一次 sync 就全量重排已有稳定索引

### 19.3 当前明确要覆盖的边界情况

- playlist info 获取失败：直接报错，不进入部分写入
- entry 为 `None` 或缺 `id`：跳过，不污染候选计划
- 已存在 video 但属于新 playlist：允许只新增关联，不重建 video
- 已存在 video 需要补抓 metadata：进入补抓计划，而不是当作普通 existing 直接跳过
- 永久不可用的新成员：禁止进入稳定成员集合
- 永久不可用的 stale zero-index 成员：只清理当前 playlist 残留，必要时保留全局 video
- 暂时性获取失败：保留成员，插值日期，不标记 `unavailable`
- 真实 metadata 回补成功：清除 `unavailable` / `unavailable_reason` / `upload_date_interpolated`
- 显式 `target_playlist_id`：优先于 yt-dlp 返回值

### 19.4 当前重构边界

后续对 `sync_playlist()` 的重构应按下面顺序推进：

1. 候选规划
2. fetch 结果收集
3. unavailable/stale 成员裁剪
4. 最终落库
5. 下游翻译与索引分配

要求：

- 每一步都应能用测试单独描述
- 不允许为了“函数更短”而把这些语义重新混在一起
- 不允许把“暂时失败”和“永久不可用”重新混淆
