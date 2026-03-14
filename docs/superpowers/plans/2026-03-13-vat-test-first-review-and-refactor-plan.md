# VAT Test-First Review And Refactor Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先整理当前工作区，再以“需求与契约先行、测试优先、重构后置”的方式，对 `VAT` 全仓建立稳定约束，并在这些约束之上分阶段完成架构审查与重构。

**Architecture:** 本计划把后续长期工作拆成五个大阶段：`工作区整理 -> 需求/契约基线 -> 测试补强 -> 分块重构 -> 文档与回归收口`。执行中禁止先改代码再补测试；每个子系统都要先明确“它应该提供什么能力、输出什么产物、在哪一层做什么检查”，再把这些内容转成测试，最后才允许进入实现重构。

**Tech Stack:** Python 3.10+, Click, FastAPI, SQLite, yt-dlp, faster-whisper, FFmpeg, biliup, pytest

---

## 1. 文档用途

这不是一次性的 brainstorming 备忘，而是后续多个工作回合都要持续引用、更新和执行的主计划文档。

这个文档要承担五个职责：

1. 记录当前仓库状态与主要风险。
2. 定义各模块“应该是什么”的需求与契约基线。
3. 指导测试优先的补测顺序和验收门槛。
4. 指导分块重构的执行顺序、边界与禁止事项。
5. 作为后续每轮工作结束时的回写位置，持续维护状态。

## 2. 当前上下文

- 日期：`2026-03-14`
- 当前分支：`refactor/test-first-hardening`
- 当前工作区：`clean`
- 当前分支尖端提交：`59491f8 test: assert task routes refresh stale status`
- 当前阶段：`测试补强仍在进行中，尚未进入高争议核心模块的大规模重构`
- 当前已完成的主线：
  - Web job 生命周期第一轮契约收口
  - CLI / pipeline 第一轮阶段语义与 force 失效护栏
  - `tasks` / `playlists` 路由第一轮 API 契约测试
  - `PlaylistService.sync_playlist()` 第一轮基础契约测试
- 当前仓库应用代码与测试代码总量约 `33894` 行。
- 目录体量最大的区域：`vat/web`、`vat/asr`、`vat/llm`、`tests`。

### 当前工作区改动分组

这一步是执行前的硬前置条件。没有完成分组整理，不进入重构。

- Group A: Vertex / LLM provider 相关
  - `vat/config.py`
  - `vat/llm/client.py`
  - `tests/test_llm_client_vertex.py`
  - `requirements.txt`
  - `config/default.yaml`
- Group B: 翻译失败策略与 prompt 调整
  - `vat/translator/llm_translator.py`
  - `tests/test_translator_error_handling.py`
  - `vat/llm/prompts/custom/translate/fubuki.md`
  - `vat/llm/prompts/translate/reflect.md`
- Group C: playlist 标识与同步
  - `vat/services/playlist_service.py`
  - `vat/web/routes/playlists.py`
  - `tests/test_resolve_playlist_id.py`
- Group D: 实验脚本与评测文档
  - `scripts/translation_benchmark.py`
  - `scripts/run_gemini_benchmark.sh`
  - `scripts/batch_sync_info.py`
  - `docs/TRANSLATION_AND_ASR_EVALUATION.md`
- Group E: 小幅 pipeline 配套改动
  - `vat/pipeline/executor.py`
- Group F: 当前计划文档
  - `docs/superpowers/plans/2026-03-13-vat-test-first-review-and-refactor-plan.md`

### 当前工作区改动的暂定标签

这不是最终提交决议，但已经足够指导下一步清理。

| Group | 暂定标签 | 当前判断 |
|---|---|---|
| A | `needs_split` | `vat/config.py` / `config/default.yaml` 同时混入了 Vertex provider、translate fallback 配置和依赖调整，主题不单一；`requirements.txt` 的 FastAPI 升级也应独立处理 |
| B | `needs_split` | 翻译 fallback 开关、prompt 修改、测试补充是相关主题，但配置入口与 pipeline 透传散落在 Group A/E，需要重新并组 |
| C | `ready_to_commit` | playlist tab URL -> DB playlist id 后缀的修复主题相对单一，代码、Web 路由、测试基本同向 |
| D | `experimental_only` | 基准脚本和评测文档偏实验/研究材料，默认不进入当前主线重构工作区 |
| E | `needs_split` | `executor.py` 里的 `enable_fallback` 透传应并入翻译 fallback 主题；额外日志改动需要单独判断是否保留 |
| F | `ready_to_commit` | 当前计划文档本身可保留在主线执行空间，用于后续工作 |

### 当前工作区整理建议

- 先不要提交 Group A、B、E。
- 先把翻译 fallback 相关改动重新并成一个主题组：
  - `vat/translator/llm_translator.py`
  - `vat/config.py` 中的 `enable_fallback`
  - `config/default.yaml` 中的 `enable_fallback`
  - `vat/pipeline/executor.py` 中的 `enable_fallback` 透传
  - `tests/test_translator_error_handling.py`
- 把 Vertex provider 相关改动单独成组：
  - `vat/llm/client.py`
  - `vat/config.py` 中的 provider/location/project_id
  - `config/default.yaml` 中的 `llm.provider/location/project_id`
  - `tests/test_llm_client_vertex.py`
- `requirements.txt` 中 `fastapi>=0.115.0` 应单独审查，不与 LLM/provider 主题捆绑。
- Group C 可作为优先候选提交组。
- Group D 默认留在本地或单独文档/实验分支，不混入后续主线重构。

### 工作区整理原则

- 不自动把上述所有改动合并成一个提交。
- 每组都要先判断：这是功能完成中的真实改动、试验性改动、还是仅供本地调研的中间状态。
- 不在脏工作区上直接开重构分支。
- 整理完成后，再创建新的 review/refactor worktree 或新分支。

### 工作区整理完成的硬标准

只有同时满足下面条件，才算“工作区整理完成”：

- 已对每个改动组给出 `ready_to_commit / needs_split / experimental_only` 标签。
- 已明确未跟踪文件各自归属，不存在“不知道是不是要保留”的文件。
- 已明确 `master` 上是否需要先落一个清理基线提交。
- 若不准备把某组带入主线，已经明确其保留策略或放弃策略。
- 准备进入新 worktree 前，至少已能运行一次最小基线测试集：
  - `pytest tests/test_models.py tests/test_database.py -q`
- 进入新 worktree 时，新的执行空间中不应再混入与本轮目标无关的历史实验改动。

## 3. 总体执行原则

### 3.1 核心原则

- 需求先于实现。
- 契约先于防御性编码。
- 测试先于重构。
- fail-fast 优先于静默兼容。
- 单一真值来源优先于多处兜底。
- 阶段原子性优先于“先做再补”。
- WebUI 是管理层，不应复制或绕过核心能力。

### 3.2 本项目中的“测试优先”含义

本计划中的测试优先，不是传统意义上“每次先写一个很小的单测”就结束，而是分三层：

1. 先定义模块需求和阶段契约。
2. 再写覆盖这些契约的约束测试。
3. 最后在测试保护下重构实现。

也就是说，后续任何模块的工作顺序都应当是：

`理解需求 -> 定义输入/输出/不变量/失败语义 -> 写或补测试 -> 跑红/确认缺口 -> 修改实现 -> 跑绿 -> 文档同步`

### 3.3 禁止事项

- 禁止在契约不清的情况下直接重构。
- 禁止因为下游能兜底，就放任上游输出不完整产物。
- 禁止新增无来源配置项。
- 禁止用更多 if/else 继续堆叠历史兼容层。
- 禁止在没有测试护栏的情况下大面积改动状态机。

## 4. 已确认的高优先级问题地图

这部分是后续进入逐文件 review 时的优先级依据。

### P0: 核心控制面

- `translate` 阶段组语义与 CLI 文案/调用意图漂移，影响命令真实行为。
- `cli/commands.py` 仍有旧字段访问和字段命名漂移。
- `VideoProcessor` 运行时修改共享 `config`，且 `cli process` 多线程复用同一配置对象。
- GPU 选择存在多入口，CLI 覆写无法稳定穿透所有执行点。
- `SKIPPED`、`force`、下游失效、依赖判定没有被统一成严格状态机。
- `scheduler` 对子进程失败的汇总与退出码传递不足。

### P0: Web / 外部交互

- `web/jobs.py` 的结果判定过于乐观，早崩子进程可能被误判成功。
- `cancel_job()` 的旧实现既会把 zombie 误判为“仍存活”，也会在 async 路由中同步阻塞等待。
- Web 层实际存在大量绕过 CLI/JobManager 的旁路。
- `BilibiliUploader` 多步远程副作用缺少事务包装、补偿逻辑和统一错误分型。
- `PlaylistService.sync_playlist()` 缺少 playlist 级原子性。

### P0: ASR / LLM / 字幕 / 嵌字

- `async_embedder.py` 调 `embed_subtitle_hard()` 的签名疑似已经错位。
- “翻译零容忍”设计目标与实现不一致，仍存在原文补洞逻辑。
- 人声分离分支与主 GPU fail-fast 策略不一致。
- 缓存键覆盖不完整，存在“换配置但吃旧产物”的风险。

### P1: 测试 / 文档 / hygiene

- `pytest` 入口本身不稳定，裸跑会被仓库根目录符号链接带出边界。
- 测试强项集中在 DB/状态机，弱项集中在跨层契约与真实运行时。
- 文档与默认配置已经出现多处相互冲突。

## 5. 目标架构与需求基线

这一节不是描述当前实现，而是定义后续审查与重构要对齐的“应该如此”的标准。

### 5.0 当前就生效的核心契约表

下面三张表从现在开始就是执行基线，而不是后面才补的占位内容。后续若修订，需要显式更新本节。

#### 5.0.1 阶段契约表

| 阶段 | 前置依赖 | 主要输入 | 主要输出 | 完成条件 | 失败条件 | 允许警告完成 | 允许复用旧产物 |
|---|---|---|---|---|---|---|---|
| `download` | 无 | `source_url` / `source_type` / 配置 | 本地视频主文件、基础 metadata、必要时字幕/场景/翻译标题等 | DB 中视频可查询，主视频产物存在，供下游使用的关键字段已统一校验 | 主视频产物不存在；关键字段无法判定且未显式标记不可得 | 可以，但 warning 必须可追踪 | 可以，但必须由明确缓存/快照规则决定 |
| `whisper` | `download` | 本地视频或音频产物 | `original_raw.srt` 及等价 ASR 结构 | 原始 ASR 结果完整可读，时间轴单调合法 | 无法产生可消费的原始 ASR 产物 | 仅限非关键清洗 warning | 可以，但必须绑定 ASR 相关配置 |
| `split` | `whisper` | `original_raw.srt` | `original_split.srt`、`original.srt`、`original.json` | 句子级结构完整，对齐合法，可直接供 optimize/translate 使用 | 编号或时间对齐被破坏；输出缺段 | 原则上不依赖 warning 完成 | 可以，但必须绑定 split 相关配置 |
| `optimize` | `split` | `original.srt` | `optimized.srt` | 不改变编号和关键对齐契约，只优化文本内容 | 输出缺段、编号漂移、系统性失败超阈值 | 可以，但必须有明确阈值与记录 | 可以，但必须绑定优化模型/提示词/配置 |
| `translate` | `optimize` 或显式定义的直达模式 | `optimized.srt` 或 `original.srt` | `translated.srt`、`translated.ass` | 所有片段都有有效翻译，编号与对齐不破坏 | 任意片段静默漏翻、以原文补洞却未声明、输出不完整 | 原则上不允许 | 可以，但必须绑定模型、prompt、reflect、context 等 |
| `embed` | `translate` | `translated.srt` / `translated.ass` / 视频主文件 | `final.*`、嵌字日志 | 目标视频产物可播放且与字幕契约一致 | 目标视频不存在、嵌字失败、错误回退掩盖失败 | 原则上不允许 | 仅允许基于明确文件新旧判定 |
| `upload` | `embed` | `final.*`、上传配置、标题模板、playlist/season 信息 | 远端稿件、远端 metadata、必要时合集状态 | 远端对象状态与本地记录一致且可追踪 | 远端副作用部分成功但无补偿；远端与本地状态不一致 | 仅对非关键附属动作允许 warning | 原则上不复用“半完成”远端状态 |

#### 5.0.2 状态机不变量表

| 主题 | 当前执行基线 |
|---|---|
| `completed` | 阶段数据库状态、阶段对外产物、阶段关键 metadata 三者一致才算完成 |
| `failed` | 阶段未形成可消费产物，或形成了不满足契约的产物 |
| `skipped` | 仅用于明确定义的“不需要执行但后续语义仍成立”情形，不能掩盖未知状态 |
| `warning` | warning 不能改变阶段是否可被下游安全消费；只能附着在已满足核心契约的完成态上 |
| `force` | 强制重跑上游阶段时，必须重新评估并失效所有受其产物影响的下游阶段 |
| 下游失效 | 下游若消费了被重算或被改配置影响的上游产物，就不得继续保持 `completed` |
| 依赖满足 | 不以“文件似乎还在”为准，而以状态与产物契约共同满足为准 |
| 单一真值来源 | 阶段是否完成以数据库状态为主判据；文件存在性是校验依据，不是平行真值来源 |

#### 5.0.3 Web / Job 契约表

| 主题 | 当前执行基线 |
|---|---|
| 长任务入口 | 默认必须通过 `JobManager -> CLI 子进程 -> 核心 pipeline/services` 路径执行 |
| 允许的旁路 | 仅允许明确标记的只读查询、极薄协调逻辑或文档已声明的例外 |
| `submit_job` 成功 | 只有在 DB 记录成功且子进程成功启动时才算成功 |
| `cancel_job` 成功 | 只表示取消请求已被接受并已向目标进程组发出终止信号；不得在 Web 请求线程里阻塞等待退出 |
| `cancelled` 终态 | 只能在 `update_job_status()` 等状态收敛路径确认子进程已结束后写入，不能在发出取消请求时乐观提前写入 |
| `update_job_status` | 在证据不足时必须保守，宁可停留在不确定/失败，也不能乐观判完成 |
| process job 完成 | 必须能证明目标视频集合对应阶段的任务状态满足完成契约 |
| tools job 完成 | 必须能证明外部副作用或日志结果与任务类型契约一致 |
| Web 与核心边界 | Web 不能复制核心状态机；若必须旁路，后续重构优先把它收回统一入口 |

## Chunk 1: 核心状态机与控制面需求基线

### 5.1 `models.py` 应承担的职责

- 定义全项目的阶段枚举、状态枚举、依赖图、默认顺序。
- 明确定义“单阶段”和“阶段组”的语义，不允许 CLI 与模型层各说各话。
- 明确定义 `Video`、`Task`、`Playlist` 这些模型的持久化字段和运行时字段边界。

### 5.2 `config.py` 应承担的职责

- 只负责配置解析、默认值装配、配置对象构建。
- 可以做基本校验，但不应承担过多运行时副作用。
- 不应把“配置文件解析”和“进程全局状态修改”混成一个不可分操作，除非这是明确设计并有测试约束。

### 5.3 `database.py` 应承担的职责

- 成为任务状态与阶段完成性的单一事实来源。
- 提供明确的状态推进、失效、重算、查询接口。
- 原子性边界要清晰：一旦一个事务声称“某阶段完成”，对应的数据库状态与产物状态必须一致。

### 5.4 `pipeline/executor.py` 应承担的职责

- 它是单视频编排层，不是隐式配置修补层。
- 它负责按阶段顺序调度各组件，并在恰当边界更新状态。
- 它可以协调依赖，但不能靠下游补洞来掩盖上游契约不清。
- 它不应把外部共享配置对象当作可随手改写的临时上下文容器。

### 5.5 这一块的必须测试

- `TaskStep` 与阶段组的语义测试。
- `force`、`SKIPPED`、下游失效、依赖满足的状态机测试。
- CLI 到 executor 的阶段映射测试。
- 多线程 `process` 下共享配置不被污染的测试。
- 多进程调度失败时退出码与聚合结果测试。

## Chunk 2: 下载 / 同步 / 上传需求基线

### 5.6 download 阶段完成时必须承诺的事情

- 视频记录存在且可查询。
- 下载产物已落地，且路径与 DB/metadata 一致。
- 必要元数据字段已在这一阶段补齐，或明确标记为不可得。
- 若该阶段结束时要为下游提供场景、翻译标题、字幕、上传日期等字段，则必须在此阶段完成统一校验。

### 5.7 playlist sync 应承担的职责

- 明确“同步”是新增、更新索引、补元数据，还是还包括删除/清理。
- 明确同步完成的原子性边界。
- 明确失败时允许的中间状态，以及这些中间状态如何可恢复。

### 5.8 upload / bilibili 相关应承担的职责

- 任何“修改现有远端对象”的操作都必须明确原子性模型。
- 对“删除再添加”“替换视频”“合集同步”这类多步远程副作用，需要定义补偿或至少定义中断后可恢复策略。
- 明确可重试错误和不可重试错误的分界。

### 5.9 这一块的必须测试

- 下载阶段产物契约测试。
- playlist sync 的新增/重复/中断恢复测试。
- season/合集相关幂等测试。
- 上传模板渲染与上传配置读写测试。
- 网络错误分型与重试策略测试。

## Chunk 3: ASR / Split / Optimize / Translate / Embed 需求基线

### 5.10 ASR 主线应承担的职责

- `WHISPER` 输出的原始产物、`SPLIT` 输出的标准化产物必须边界清晰。
- 每个阶段的输入文件、输出文件、可见中间文件必须有固定契约。
- GPU 使用策略应统一，不能主线 fail-fast、分支 silently fallback。

### 5.11 split / optimize / translate 应承担的职责

- split 负责把 ASR 原始片段整理成适合阅读和翻译的句子级结构。
- optimize 负责在同语种内部做纠错与统一，不应改变下游所依赖的关键编号/对齐契约。
- translate 负责完整翻译，不允许静默漏翻；若设计允许保底原文，则必须在契约层明确，而不能在实现中暗补。

### 5.12 embed 应承担的职责

- 明确软字幕、硬字幕、异步嵌字服务三条路径的职责与一致性要求。
- 明确 `translated.srt`、`translated.ass`、`final.*` 各自产物的优先级与重建规则。
- 嵌字失败时要么 fail-fast，要么明确规定可接受的回退路径并有测试覆盖。

### 5.13 这一块的必须测试

- ASR 产物契约测试。
- split/optimize/translate 编号与对齐一致性测试。
- 翻译零容忍契约测试。
- 缓存键覆盖测试。
- async embedder 与主 pipeline embed 一致性测试。
- GPU fail-fast 与禁止 CPU fallback 测试。

## Chunk 4: Web 管理层需求基线

### 5.14 Web 层应承担的职责

- 管理任务、展示状态、提供交互入口。
- 默认应通过 CLI/JobManager 间接调用核心能力。
- 如确需旁路，必须是明确授权的只读查询或非常薄的协调层，并写入文档。

### 5.15 JobManager 应承担的职责

- Web 侧长任务的持久化、启动、取消、状态更新、日志关联。
- 结果判定必须保守，不能在信号不足时乐观判成功。
- 取消语义必须对应“整个任务树停止”，而不是“向父 PID 发过信号”。

### 5.16 这一块的必须测试

- `POST /api/tasks/execute` 约束测试。
- `POST /api/watch/start` 契约测试。
- playlist sync / refresh / retranslate 路由测试。
- JobManager 子进程生命周期集成测试。
- 路由是否绕过 CLI 的契约测试。

## Chunk 5: 文档 / 默认值 / repo hygiene 需求基线

### 5.17 文档应满足的最低要求

- README、模块 readme、专项文档、默认配置必须一致。
- 用户可见默认值只能有一个真值来源，其他文档都应引用或同步它。
- 文档不能宣传代码中未被测试或未被支持的行为。

### 5.18 repo hygiene 应满足的最低要求

- 测试入口稳定，不会越过项目边界。
- 实验脚本、一次性脚本、生产路径代码边界清楚。
- worktree / branch / docs 结构稳定，适合长期执行。

### 5.19 这一块的必须测试

- `pytest tests --collect-only -q` 稳定性测试或至少流程约束。
- 默认端口、watch 配置、job db 路径等关键默认值的一致性测试。
- 文档示例命令与配置字段的回归校验测试。

## 5.20 环境与高风险流程门槛

这一节是阻断条件，不满足就不推进对应阶段。

### 5.20.1 环境就绪门槛

| 项目 | 阻断条件 | 备注 |
|---|---|---|
| CUDA / GPU | 任何需要 GPU 的阶段若检测不到满足要求的 GPU，则默认阻断，不允许静默 CPU fallback | 人声分离当前是已知不一致点，后续要纳入统一策略 |
| FFmpeg | 缺失或关键编码器不可用时，阻断 embed 相关真实验证 | 包括 NVENC 能力校验 |
| yt-dlp | 缺失时阻断 downloader / playlist sync 相关真实验证 | 单元测试可 mock，契约测试要明确层级 |
| biliup / 上传依赖 | 缺失时阻断 upload/season 相关真实验证 | 不阻断纯本地契约测试 |
| LLM API key / endpoint | 缺失时阻断依赖真实远端 LLM 的验证 | 本地契约测试仍可先做 |

### 5.20.2 并发与重入门槛

- 同一视频在同一时间只能存在一条有效的处理链，除非测试明确验证并允许并发语义。
- 同一 playlist 的 sync / refresh / watch / season-sync 不允许无约束重入。
- 任何“重复提交”保护都必须有持久化依据，不能只靠进程内字典。

### 5.20.3 远端副作用恢复门槛

- upload/season/sync 相关流程若是多步远端副作用，必须先定义“部分成功后如何恢复”，再允许重构。
- cancel 或中断后，必须能回答三件事：
  - 本地文件处于什么状态
  - 本地数据库处于什么状态
  - 远端对象处于什么状态
- 如果回答不了，就不能声称该流程具备可靠取消/恢复语义。

## 6. 测试优先执行策略

这一节是后续所有工作的主顺序，不允许跳过。

## Chunk 6: 测试工作流

### 6.1 测试分层

- Layer 1: 契约测试
  - 模块职责、输入输出、不变量、错误分型。
- Layer 2: 跨层集成测试
  - CLI -> scheduler -> executor
  - Web route -> JobManager -> CLI -> 状态回写
  - playlist/service -> DB -> downloader/uploader
- Layer 3: 回归测试
  - 已知缺陷点。
  - 文档/默认值一致性。
  - 历史修复不倒退。

### 6.2 测试优先级

先补这些，再动实现：

1. 核心状态机与阶段语义。
2. Web Job 生命周期与取消语义。
3. 翻译零容忍与阶段产物契约。
4. 上传/season/playlist sync 的原子性与幂等性。
5. 默认值与文档一致性。

### 6.3 当前测试薄弱点清单

- Web 路由契约覆盖不足。
- CLI 到子进程链路覆盖不足。
- 上传模板和上传配置基本无直接契约测试。
- video info translator / scene identifier 基本无直接测试。
- ASR 主运行链、embedder 运行时覆盖不足。
- 现有一些大型测试文件 mock 面过大，难以保证真实 wiring。

### 6.4 后续测试命令约定

- 收集测试：
  - `pytest tests --collect-only -q`
- 单模块回归：
  - `pytest tests/test_models.py -q`
  - `pytest tests/test_database.py -q`
  - `pytest tests/test_pipeline.py -q`
  - `pytest tests/test_web_jobs.py -q`
  - `pytest tests/test_tools_job.py -q`
  - `pytest tests/test_watch_service.py -q`
- 任何需要全量测试时，优先使用：
  - `pytest tests -q`

禁止在仓库结构未整理前用裸 `pytest` 作为成功依据。

### 6.5 测试任务固定模板

从本节开始，Phase 2 的所有测试任务统一遵循这个模板，不允许退化成“边改实现边补测试”。

1. 写失败测试，只表达目标契约，不改实现。
2. 运行目标测试，确认出现预期失败。
3. 在计划文档或状态栏中记录失败信号。
4. 在该测试任务完成前，不改实现文件。
5. 当一组关键失败测试写齐后，才允许进入对应重构任务。

固定记录格式：

- `测试文件`
- `目标契约`
- `预期失败信号`
- `是否已进入实现修改`

固定通过门槛：

- 目标测试先红过。
- 修改后目标测试转绿。
- 关联回归测试转绿。
- 文档同步后，才能勾选任务完成。

### 6.6 文件 / 函数级测试审计法

从这一轮开始，测试策略进一步收紧：不只按“模块”补测试，还要按“文件 -> 函数 -> 上层调用链”自底向上审计。

执行顺序固定为：

1. 先扫纯函数和数据转换函数。
2. 再扫单文件内状态推进函数。
3. 再扫 service / manager 层的组合逻辑。
4. 再扫 CLI / Web route / JobManager 这类入口层。
5. 最后补跨层集成与回归测试。

每个文件都必须建立一个最小审计记录，至少回答：

- 这个文件的职责是什么。
- 这个文件里哪些函数是纯逻辑、哪些是状态推进、哪些是 I/O 边界。
- 哪些函数已经有直接测试。
- 哪些函数目前只被上层间接覆盖。
- 哪些函数完全没有测试。
- 哪些测试是在“验证需求/契约”，哪些只是“贴实现”。

从现在开始，后续每轮新增测试时，都要在文档或状态回写中至少标出下面四项：

- `文件`
- `函数/方法`
- `测试层级`：`unit / contract / integration / regression`
- `当前状态`：`covered / indirectly_covered / missing / needs_rewrite`

禁止把“被某个大集成测试顺便走到”当成充分覆盖。对于高风险函数，默认要求直接测试，而不是只靠上层路由或命令间接经过。

优先要做函数级审计的文件：

- `vat/models.py`
- `vat/database.py`
- `vat/pipeline/executor.py`
- `vat/web/jobs.py`
- `vat/web/routes/tasks.py`
- `vat/web/routes/playlists.py`
- `vat/services/playlist_service.py`
- `vat/uploaders/bilibili.py`
- `vat/translator/llm_translator.py`
- `vat/embedder/async_embedder.py`

## 7. 工作区整理计划

这一步先执行，未完成前不进入后续重构。

## Chunk 7: Workspace Cleanup

### Task 1: 确认每组改动意图

**Files:**
- Review: `vat/config.py`
- Review: `vat/llm/client.py`
- Review: `vat/translator/llm_translator.py`
- Review: `vat/services/playlist_service.py`
- Review: `vat/web/routes/playlists.py`
- Review: `vat/pipeline/executor.py`
- Review: `requirements.txt`
- Review: `config/default.yaml`
- Review: `tests/test_llm_client_vertex.py`
- Review: `tests/test_resolve_playlist_id.py`
- Review: `tests/test_translator_error_handling.py`
- Review: `scripts/translation_benchmark.py`
- Review: `scripts/run_gemini_benchmark.sh`
- Review: `scripts/batch_sync_info.py`
- Review: `docs/TRANSLATION_AND_ASR_EVALUATION.md`

- [ ] **Step 1: 逐组复核差异，给每组打标签**

标签只允许三类：`ready_to_commit`、`needs_split`、`experimental_only`

- [ ] **Step 2: 列出每组是否应进入主线**

输出格式：
`Group A -> ready_to_commit`

- [ ] **Step 3: 标出绝不能自动提交的文件**

至少包括不确定用途的脚本、实验文档、中途测试草稿。

- [ ] **Step 4: 形成工作区整理建议摘要**

需要回答：
- 哪些可以先提交？
- 哪些需要拆分？
- 哪些只应保留本地？
- 哪些必须在进入新 worktree 前移出主线执行空间？

### Task 2: 整理完成后的分支策略

**Files:**
- Reference: `.git`
- Reference: `docs/superpowers/plans/2026-03-13-vat-test-first-review-and-refactor-plan.md`

- [ ] **Step 1: 确认当前 `master` 是否要承接工作区清理提交**
- [ ] **Step 2: 若承接，则先提交清理性提交**
- [ ] **Step 3: 基于干净状态创建 review/refactor worktree**
- [ ] **Step 4: 后续所有重构都在新 worktree 中进行**
- [ ] **Step 5: 在新 worktree 中跑最小基线测试集**

## 8. 分阶段执行计划

## Chunk 8: Phase 1 - 契约定义

这一阶段只做阅读、归纳、测试设计，不做大规模实现修改。

### Task 3: 核心状态机契约文档化

**Files:**
- Modify: `docs/superpowers/plans/2026-03-13-vat-test-first-review-and-refactor-plan.md`
- Review: `vat/models.py`
- Review: `vat/database.py`
- Review: `vat/pipeline/executor.py`
- Review: `vat/cli/commands.py`

- [ ] **Step 1: 写出阶段表**

表格应包含：
- 阶段名
- 前置依赖
- 产物
- 完成条件
- 失败条件
- 是否允许警告完成
- 是否允许复用旧产物

- [ ] **Step 2: 写出状态机不变量**

至少包括：
- 什么叫 `completed`
- 什么叫 `failed`
- 什么叫 `skipped`
- `force` 应如何影响下游

- [ ] **Step 3: 给每个不变量映射到测试文件**

### Task 4: Web / Job 契约文档化

**Files:**
- Modify: `docs/superpowers/plans/2026-03-13-vat-test-first-review-and-refactor-plan.md`
- Review: `vat/web/jobs.py`
- Review: `vat/web/routes/tasks.py`
- Review: `vat/web/routes/watch.py`
- Review: `vat/web/routes/playlists.py`
- Review: `vat/web/routes/bilibili.py`

- [ ] **Step 1: 明确哪些路径必须经 CLI**
- [ ] **Step 2: 标出当前旁路路径**
- [ ] **Step 3: 给出哪些旁路要删除、哪些要保留、哪些要包装**
- [ ] **Step 4: 定义 Job 生命周期契约**

## Chunk 9: Phase 2 - 测试补强

### Task 5: 核心状态机测试补强

**Files:**
- Modify: `tests/test_models.py`
- Modify: `tests/test_database.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 先补 `translate` 阶段语义测试**

Run: `pytest tests/test_models.py -q`
Expected: 新增语义测试初始失败或暴露不一致

- [ ] **Step 2: 补 `force/skipped/invalidate_downstream` 状态机测试**

Run: `pytest tests/test_database.py tests/test_pipeline.py -q`
Expected: 至少出现当前状态机闭环不完整对应的失败信号

- [ ] **Step 3: 补 CLI -> executor 映射测试**

Run: `pytest tests/test_pipeline.py -q`
Expected: 暴露阶段映射或命令语义不一致

- [ ] **Step 4: 补共享 config 污染测试**

Run: `pytest tests/test_pipeline.py -q`
Expected: 在现状下能区分“共享配置被污染”和“隔离良好”

### Task 6: Web 生命周期测试补强

**Files:**
- Modify: `tests/test_web_jobs.py`
- Modify: `tests/test_tools_job.py`
- Modify: `tests/test_watch_api.py`
- Create: `tests/test_tasks_api.py`
- Create: `tests/test_playlists_api.py`

- [ ] **Step 1: 给 `JobManager` 补结果判定保守性测试**
- [ ] **Step 2: 跑红并记录“早崩子进程被误判成功”的失败信号**
- [ ] **Step 3: 给 `cancel_job()` 补“取消请求 / 最终收敛”分离测试**
- [ ] **Step 4: 跑红并记录 zombie 误判与 async 阻塞的失败信号**
- [ ] **Step 5: 给 `POST /api/watch/start` 补 API 级测试**
- [ ] **Step 6: 给 tasks/playlists 关键路由补契约测试**

### Task 7: 翻译 / ASR / 嵌字契约测试补强

**Files:**
- Modify: `tests/test_translator_error_handling.py`
- Create: `tests/test_translator_contracts.py`
- Create: `tests/test_embedder_runtime.py`
- Create: `tests/test_video_info_translator.py`
- Create: `tests/test_scene_identifier.py`
- Create: `tests/test_upload_config.py`
- Create: `tests/test_upload_template.py`

- [ ] **Step 1: 先补“翻译零容忍”测试**
- [ ] **Step 2: 补 cache key 覆盖测试**
- [ ] **Step 3: 补 async embedder 与 FFmpegWrapper 签名一致性测试**
- [ ] **Step 4: 补 video info translator / scene identifier 测试**
- [ ] **Step 5: 补 upload config / template 契约测试**
- [ ] **Step 6: 统一跑红并记录每类失败信号，再进入实现修改阶段**

### Task 8: 文档与默认值一致性测试

**Files:**
- Create: `tests/test_docs_and_defaults.py`
- Review: `README.md`
- Review: `README_EN.md`
- Review: `docs/webui_manual.md`
- Review: `vat/web/readme.md`
- Review: `config/default.yaml`
- Review: `vat/config.py`

- [ ] **Step 1: 选定关键默认值检查项**

至少包括：
- web 默认端口
- watch 限制语义
- job db 位置
- playlist prompt API

- [ ] **Step 2: 写一致性测试**

Run: `pytest tests/test_docs_and_defaults.py -q`

## Chunk 10: Phase 3 - 分块重构

这一步只有在相关测试已经补齐并稳定通过后才开始。

### Task 9: 重构顺序

**Files:**
- Modify: `vat/pipeline/executor.py`
- Modify: `vat/cli/commands.py`
- Modify: `vat/database.py`
- Modify: `vat/web/jobs.py`
- Modify: `vat/uploaders/bilibili.py`
- Modify: `vat/services/playlist_service.py`
- Modify: `vat/translator/llm_translator.py`
- Modify: `vat/embedder/async_embedder.py`

- [ ] **Step 1: 先修阶段语义与状态机**
- [ ] **Step 2: 再修 Web Job 生命周期**
- [ ] **Step 3: 再修上传/season/playlist sync 原子性**
- [ ] **Step 4: 再修翻译/嵌字契约与缓存**

### Task 10: 每个重构子块的固定流程

- [ ] **Step 1: 明确本子块目标与不改范围**
- [ ] **Step 2: 先写失败测试**
- [ ] **Step 3: 跑目标测试确认失败**
- [ ] **Step 4: 做最小实现修改**
- [ ] **Step 5: 跑目标测试确认通过**
- [ ] **Step 6: 跑关联回归测试**
- [ ] **Step 7: 更新文档**
- [ ] **Step 8: 单独提交**

## Chunk 11: Phase 4 - 文档与回归收口

### Task 11: 文档校准

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `docs/webui_manual.md`
- Modify: `vat/web/readme.md`
- Modify: 其他受影响子模块 readme

- [ ] **Step 1: 统一默认值描述**
- [ ] **Step 2: 统一 API 路径说明**
- [ ] **Step 3: 标出实验能力与正式支持能力的边界**
- [ ] **Step 4: 删除或修正文档中的失真陈述**

### Task 12: 最终验证

**Files:**
- Test: `tests/`

- [ ] **Step 1: 收集测试**

Run: `pytest tests --collect-only -q`
Expected: 成功收集，无越界收集

- [ ] **Step 2: 运行核心子集**

Run: `pytest tests/test_models.py tests/test_database.py tests/test_pipeline.py tests/test_web_jobs.py tests/test_tools_job.py tests/test_watch_service.py -q`

- [ ] **Step 3: 运行新增契约测试**

Run: `pytest tests/test_docs_and_defaults.py tests/test_translator_contracts.py tests/test_embedder_runtime.py -q`

- [ ] **Step 4: 运行全量项目测试**

Run: `pytest tests -q`

## 9. 模块级 review 顺序

后续逐文件详细 review 的顺序固定如下：

1. `vat/pipeline/executor.py`
2. `vat/cli/commands.py`
3. `vat/database.py`
4. `vat/web/jobs.py`
5. `vat/web/routes/tasks.py`
6. `vat/web/routes/playlists.py`
7. `vat/web/routes/bilibili.py`
8. `vat/uploaders/bilibili.py`
9. `vat/services/playlist_service.py`
10. `vat/translator/llm_translator.py`
11. `vat/embedder/async_embedder.py`
12. `vat/llm/client.py`
13. `vat/config.py`
14. `vat/pipeline/scheduler.py`
15. `vat/services/watch_service.py`
16. `vat/web/routes/watch.py`
17. `vat/downloaders/youtube.py`
18. `vat/downloaders/direct_url.py`
19. `README.md` / `README_EN.md` / `docs/`

## 9.1 风险项到 owner / 测试 / 门槛映射表

| 风险项 | Owner 文件 | 首要契约测试 | 进入重构门槛 |
|---|---|---|---|
| `translate` 语义漂移 | `vat/models.py` `vat/cli/commands.py` | `tests/test_models.py` `tests/test_pipeline.py` | 能稳定区分“单阶段 translate”和“阶段组 translate” |
| 共享 `config` 污染 | `vat/pipeline/executor.py` `vat/cli/commands.py` | `tests/test_pipeline.py` | 已有并发场景下配置不串扰测试 |
| 子进程失败被误判成功 | `vat/web/jobs.py` | `tests/test_web_jobs.py` | 已有早崩/无 task 记录场景测试 |
| 取消语义不完整 | `vat/web/jobs.py` `vat/web/routes/tasks.py` | `tests/test_web_jobs.py` `tests/test_tasks_api.py` | 已有“取消请求 / 最终收敛 / route 非阻塞”测试 |
| playlist sync 非原子 | `vat/services/playlist_service.py` | `tests/test_services.py` 或新增 sync 契约测试 | 已定义中断恢复语义 |
| season / upload 多步副作用 | `vat/uploaders/bilibili.py` | `tests/test_scheduled_upload.py` `tests/test_bilibili_violation.py` 及新增幂等测试 | 已定义部分成功补偿或恢复规则 |
| 翻译零容忍失效 | `vat/translator/llm_translator.py` `vat/translator/base.py` | `tests/test_translator_contracts.py` | 已有漏翻即失败测试 |
| async embedder 坏路径 | `vat/embedder/async_embedder.py` | `tests/test_embedder_runtime.py` | 已有签名一致性与最小运行路径测试 |
| GPU 策略不一致 | `vat/asr/whisper_wrapper.py` `vat/asr/vocal_separation/separator.py` | `tests/test_gpu_utils.py` 及新增 GPU 契约测试 | 已定义何处允许/不允许 fallback |
| watch 重入/冲突 | `vat/services/watch_service.py` `vat/web/routes/watch.py` | `tests/test_watch_api.py` `tests/test_watch_service.py` | 已有重复 session / 冲突提交测试 |
| download 契约不清 | `vat/pipeline/executor.py` `vat/downloaders/youtube.py` | 新增 download 产物契约测试 | 已写明 download 完成时的必要字段与产物 |
| 文档默认值失真 | `README.md` `README_EN.md` `docs/` `config/default.yaml` | `tests/test_docs_and_defaults.py` | 已确定唯一真值来源 |

## 10. 每轮工作都要回填的状态栏

每次实际执行完一个子块，都必须回写以下四项：

- `当前处理子块`
- `本轮新增测试`
- `本轮修复的问题`
- `下一步`

### 10.1 当前状态

- `当前处理子块`: `Chunk 8: Runtime Contract Hardening`
- `本轮新增测试`:
  - `tests/test_async_embedder.py`
  - `tests/test_cli_process.py`
  - `tests/test_pipeline.py` 中新增配置隔离与早退恢复契约测试
  - `tests/test_web_jobs.py` 中新增 process job 结果判定、取消请求收敛与 zombie contract 测试
  - `tests/test_tasks_api.py`
  - `tests/test_playlists_api.py`
  - `tests/test_tools_job.py` 中新增 tools job lifecycle 收敛测试
  - `tests/test_database_api.py`
  - `tests/test_cli_process.py` 中新增 `parse_stages` / `process -s` 阶段语义与 `--force` 下游失效契约测试
  - `tests/test_services.py` 中新增 `sync_playlist()` 基础契约测试
  - `tests/test_database.py` 中新增连接回滚、锁重试、运行时 output_dir 解析、字段过滤与空 video_id 约束测试
  - `tests/test_pipeline.py` 中新增 passthrough config 恢复、playlist prompt 自动应用/恢复、`_is_no_speech`、`_is_shorts_video` 的函数级测试
  - `tests/test_season_sync.py` 中新增 `season_sync()` 的成功、诊断、不一致修复失败路径测试
  - `tests/test_scheduled_upload.py` 中新增 `_auto_season_sync()` 的“无待同步直接返回 / 失败后自动重试一次”测试
  - `tests/test_season_title_sync.py` 中新增 `sync_season_episode_titles()` 的无变更跳过、删除失败、部分重加失败、成功后恢复原顺序测试
  - `tests/test_translator_contracts.py` 中新增 `_set_segments_translated_text()` 的“索引映射正常 / 缺段立即失败”契约测试
  - `tests/test_translator_contracts.py` 中新增 `LLMTranslator._get_cache_key()` 的 prompt / reflect / context 变更影响缓存键测试
  - `tests/test_season_sorting.py` 中新增 `_extract_title_index()`、`sort_season_episodes()`、`auto_sort_season()` 的函数级契约测试
- `本轮修复的问题`:
  - 已完成工作区清理，按 task 提交现有改动，并在 `refactor/test-first-hardening` 分支开始正式修复。
  - 修复 `VideoProcessor` 直接持有共享 `config` 的问题；现在在初始化时深拷贝配置，每个 processor 都拥有独立配置副本，避免 `passthrough` 和自动 playlist prompt 覆写跨视频串扰。
  - 修复 `VideoProcessor.process()` 早退路径的 prompt 泄漏问题；自动 playlist prompt 应用已经纳入同一 `try/finally`，`unavailable` 和 “无待执行步骤” 两类早退也会正确恢复配置。
  - 修复 `AsyncEmbedderQueue._process_task()` 对 `FFmpegWrapper.embed_subtitle_hard()` 的参数错位；现在通过关键字参数传递 `gpu_device`，不再把 `gpu_id` 错塞进 `progress_callback` 槽位。
  - 修复 `vat.embedder.async_embedder` 仍依赖旧版数据库接口的问题；异步嵌字队列现已改为使用当前 `Database` / `TaskStep` / `TaskStatus` 回写 `embed` 任务状态，并在 `embed_service` 初始化时显式传入数据库路径与输出根目录。
  - 修复 `cli process` 对缓存全局配置对象的命令级污染；命令入口现在先复制调用级配置，再应用 playlist prompt，上层 `CONFIG` 缓存不会被单次命令改脏。
  - 修复 `translate --backend` 的陈旧 CLI 契约；命令现在只接受 `local/online`，并正确映射到 `translator.backend_type`，不再写入不存在的 `default_backend` 字段，同时同样具备调用级配置隔离。
  - 修复 `cli process --force` 未失效下游阶段的问题；现在会按目标阶段中最早的那个阶段统一调用 `invalidate_downstream_tasks()`，与 Web execute 路径保持一致。
  - 收紧 `JobManager._determine_job_result()`：进程结束后若某视频缺少任务记录或步骤未完成，不再被乐观判成成功；存在部分完成时返回 `partial_completed`。
  - 修复 `JobManager.cancel_job()` 的同步阻塞设计；取消现在改为“接受请求并向进程组发送 `SIGTERM`”，不再在 Web 请求线程里等待退出。
  - 修复 `cancel_job()` / `cancel_task` 的 zombie 与事件循环问题；取消状态通过 `cancel_requested` 持久化，由 `update_job_status()` 在确认子进程结束后收敛为 `cancelled`，`POST /api/tasks/{id}/cancel` 也改为 `asyncio.to_thread(...)` 非阻塞调用。
  - 收敛 `update_job_status()` 的进程活性判断；僵尸进程现在视为已结束并尝试回收，不再把 zombie 当成“仍在运行”。
  - 修复 `JobManager.submit_job()` 的原子性缺口；当 `_start_job_process()` 失败时，现在会回滚 `web_jobs` 记录，避免残留脏的 `pending` job。
  - 修复 `vat/web/routes/database.py` 的 FastAPI 弃用项；`Query(..., regex=...)` 已迁移为 `pattern=`，并补了 API 参数校验测试。
  - 修复 `vat/web/routes/tasks.py` 的陈旧状态问题；`delete` 与 `retry` 现在都会先 `update_job_status()` 再判定 running，避免已结束任务因陈旧状态被误拦截。
  - 补齐 `vat/web/routes/tasks.py` 的关键 API 契约测试：覆盖 `parse_steps`、`execute`、`get/list`、`cancel`、`delete`、`retry`，把 stage group 展开、force 下游失效、冲突拒绝、CLI 预览、异步取消与状态刷新语义固定下来。
  - 补齐 `vat/web/routes/playlists.py` 的第一批 API 契约测试：覆盖 `add/sync/refresh/retranslate` 的后台任务提交、重复提交保护，以及 `sync-status/refresh-status` 的状态映射。
  - 补齐 tools job 生命周期测试：`update_job_status()` 现在对 `[SUCCESS] / [FAILED] / cancel_requested` 三条收敛路径都有直接测试。
  - 补齐 `PlaylistService.sync_playlist()` 的基础契约测试：覆盖显式 target playlist ID 归属、已有视频复用关联、不重建视频记录、已有关联 playlist_index 更新等底层语义。
  - 已完成本轮回归：
    - `pytest tests/test_pipeline.py tests/test_async_embedder.py tests/test_cli_process.py tests/test_web_jobs.py tests/test_watch_api.py tests/test_scheduled_upload.py tests/test_models.py tests/test_database_api.py -q`
    - `pytest tests/test_cli_process.py tests/test_models.py tests/test_pipeline.py tests/test_scheduled_upload.py -q`
    - `pytest tests/test_web_jobs.py tests/test_tasks_api.py tests/test_watch_api.py tests/test_database_api.py -q`
    - `pytest tests/test_tools_job.py tests/test_playlists_api.py -q`
    - `pytest tests/test_pipeline.py tests/test_async_embedder.py tests/test_cli_process.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_watch_api.py tests/test_scheduled_upload.py tests/test_models.py tests/test_database_api.py -q`
    - `pytest tests/test_pipeline.py tests/test_async_embedder.py tests/test_cli_process.py -q`
    - `pytest tests/test_playlists_api.py -q`
    - `pytest tests/test_tools_job.py -q`
    - `pytest tests/test_services.py -q`
    - `pytest tests/test_tasks_api.py tests/test_playlists_api.py tests/test_web_jobs.py tests/test_watch_api.py tests/test_database_api.py -q`
    - `pytest tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_tools_job.py tests/test_watch_api.py tests/test_database_api.py tests/test_pipeline.py tests/test_async_embedder.py tests/test_cli_process.py -q`
    - `pytest tests/test_services.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_web_jobs.py tests/test_watch_api.py tests/test_database_api.py -q`
    - `pytest tests/test_pipeline.py tests/test_async_embedder.py tests/test_cli_process.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_database_api.py -q`
    - `HOME=/tmp pytest tests/test_database.py -q`
    - `HOME=/tmp pytest tests/test_pipeline.py -q`
    - `HOME=/tmp pytest tests/test_models.py tests/test_database.py tests/test_pipeline.py -q`
    - `HOME=/tmp pytest tests/test_season_sync.py -q`
    - `HOME=/tmp pytest tests/test_scheduled_upload.py tests/test_season_sync.py -q`
    - `HOME=/tmp pytest tests/test_tools_job.py tests/test_scheduled_upload.py tests/test_season_sync.py -q`
    - `HOME=/tmp pytest tests/test_season_title_sync.py -q`
    - `HOME=/tmp pytest tests/test_bilibili_violation.py tests/test_tools_job.py tests/test_scheduled_upload.py tests/test_season_sync.py tests/test_season_title_sync.py -q`
    - `HOME=/tmp pytest tests/test_translator_contracts.py tests/test_translator_error_handling.py -q`
    - `HOME=/tmp pytest tests/test_translator_contracts.py tests/test_translator_error_handling.py tests/test_vertex_translation_flow.py -q`
    - `HOME=/tmp pytest tests/test_season_sorting.py -q`
    - `HOME=/tmp pytest tests/test_bilibili_violation.py tests/test_tools_job.py tests/test_scheduled_upload.py tests/test_season_sync.py tests/test_season_title_sync.py tests/test_season_sorting.py -q`
- `下一步`: 继续自底向上补剩余高风险模块测试，优先是阶段语义漂移、playlist/upload/season 的原子性，以及翻译零容忍契约

### 10.2 规划文档当前状态

这份规划文档相较最初版本，已经不再是“待完善草案”，而是当前正式执行中的主控文档。

当前判断：

- `契约基线`: 已基本成型，可以直接指导后续实现与 review。
- `测试优先总策略`: 已成型，并已在多个子块实际执行验证。
- `状态回写机制`: 已成型，但后续还要继续按轮次更新。
- `文件 / 函数级测试审计台账`: 已立项并写入本计划，下一轮开始要按此方式持续落地。
- `高争议重构规划`: 仍未完成，必须等更底层测试护栏补齐后再进入。

因此，本规划文件当前状态应视为：

- `已完成 70% 的长期执行基线定义`
- `已进入持续维护状态，而不是继续大改结构的草案阶段`
- `后续新增内容以状态更新、测试审计台账、风险收口记录为主`

后续记录方式固定为三类：

1. `状态回写`
   - 写进 `10.1 当前状态`
2. `测试审计进展`
   - 按 `6.6 文件 / 函数级测试审计法` 追加
3. `高风险模块收口记录`
   - 当某个模块进入“可重构”状态时，在风险映射表和当前状态中同时更新

### 10.3 当前文件 / 函数级测试审计进展

这一节从 `models.py -> database.py -> executor.py` 的底层链开始维护，只记录“已经实际审过并补过测试”的文件。

| 文件 | 本轮已审函数/方法 | 测试层级 | 当前状态 | 备注 |
|---|---|---|---|---|
| `vat/models.py` | `expand_stage_group` `get_required_stages` `Video.__post_init__` `Task.__post_init__` `Playlist.__post_init__` | `unit / contract` | `partially_covered` | 阶段语义主路径已有测试；后续仍可补显式时间戳保留与枚举 coercion 边界 |
| `vat/database.py` | `get_connection` `_retry_on_locked` `add_video` `get_video` `_row_to_video` `update_video` | `unit / contract` | `covered_this_round` | 已补事务回滚、锁重试、运行时 output_dir 解析、字段过滤、空 video_id 约束 |
| `vat/pipeline/executor.py` | `VideoProcessor.process` `_resolve_stage_gaps` `_set_passthrough_config` `_restore_passthrough_config` `_auto_apply_playlist_prompts` `_restore_playlist_prompts` `_is_no_speech` `_is_shorts_video` | `unit / contract / regression` | `covered_this_round` | 仍有大量 stage 实现函数未做直接函数级测试，但辅助控制逻辑这一轮已下探 |
| `vat/uploaders/bilibili.py` | `season_sync` `sync_season_episode_titles` | `contract / regression` | `covered_this_round` | 已补成功同步、upload 已完成但无 aid 诊断、aid 查无、DB/合集不一致修复失败回写，以及删后重加标题同步的主要成功/失败路径；后续继续下探排序/删除组合原子性与真正补偿策略 |
| `vat/translator/base.py` | `_set_segments_translated_text` | `contract / regression` | `covered_this_round` | 已收紧为“缺少任何翻译段即立即失败”，不再允许静默漏翻继续落盘 |
| `vat/translator/llm_translator.py` | `_get_cache_key` | `contract / regression` | `covered_this_round` | 已补 prompt / reflect / context 维度进入缓存键，避免不同翻译语义错误复用旧缓存 |
| `vat/uploaders/bilibili.py` | `_extract_title_index` `sort_season_episodes` `auto_sort_season` | `unit / contract` | `covered_this_round` | 已补标题编号提取、缺失 aid 直接失败、未列出视频自动补尾、顺序已正确时跳过排序、新增视频已在末尾时跳过排序 |

本节的维护规则：

- 只有真正新增或重写了测试后，才更新状态。
- `partially_covered` 不等于“可以停止”，而是表示已经进入函数级审计。
- `covered_this_round` 仅表示本轮新增了直接测试，不表示该文件已无缺口。

## 11. 进入下一阶段的门槛

### 从工作区整理进入契约定义的门槛

- 脏工作区已按主题分组。
- 已确认哪些改动要留、哪些要拆、哪些不要。
- 新 worktree 或新分支策略已确定。
- 已满足“工作区整理完成的硬标准”。

### 从契约定义进入测试补强的门槛

- 每个高优模块都已写出输入、输出、不变量、失败语义。
- 已知争议语义已记录，不再边做边猜。

### 从测试补强进入重构的门槛

- 对应子块的关键契约测试已存在。
- 新旧行为边界已能被测试区分。
- 有明确的最小重构目标。
- 对应测试任务已经经历过“先红后绿”的完整流程。

### 从重构进入收口的门槛

- 目标测试和关联回归全部通过。
- 文档已同步。
- 无新增未解释的配置项或旁路。

## 12. 当前建议的立即执行项

- [x] 先做工作区分组确认，不提交、不重构。
- [x] 整理并提交当前工作区，确保重要脚本和文档进入 git。
- [x] 切换到新的开发分支开始正式实现。
- [x] 先补最小 runtime 契约测试，修复 async embedder 参数错位与批量 process 配置复用。
- [ ] 继续补核心状态机测试和 Web Job 生命周期测试。
- [ ] 在 `web/jobs.py` 与上传/同步链路进入重构前，先补早崩、取消、远端副作用恢复等约束测试。

## 13. 备注

- 本计划明确提高测试优先级，并把它从“辅助验证”提升为“重构前置条件”。
- 本计划假设后续会持续多轮执行，因此文档必须在每轮之后更新，而不是只在开头写一次。
- 若后续执行中发现某个 chunk 规模仍然过大，应继续拆成更小的可验证子任务，但不得绕开本计划的顺序原则。
