# VAT 已知问题

本文档记录**已知但尚未修复**的问题，以及 LLM 成本参考信息。已修复的问题归档在 [docs/archive/](archive/) 目录。

---

## 一、LLM 成本策略

### 当前方案

各阶段按重要性分配不同模型，控制成本：

| 阶段 | 模型 | 相对成本 | 说明 |
|------|------|---------|------|
| Split | gpt-4o-mini | 1x | 断句对模型要求低，后有时间戳修正兜底 |
| Optimize | gpt-5-nano | 6x | 术语纠错靠 custom prompt 引导 |
| Translate | gemini-3-flash | 40x（开 reflect 约 60-80x） | 翻译质量要求最高，直接面向用户 |

### 效果对比

- **gpt-4o-mini 全流程**：明显不通顺，不可用于正式翻译
- **gpt-5-nano（optimize+translate）**：基本通顺，但专有名词误报多，缺乏个人风格
- **gemini-3-flash 全流程**：接近人工翻译质量，术语处理正确，风格自然

### Reflect 开销说明

Reflect 不是多次 LLM 调用，而是单次调用中输出 3 个字段（initial_translation + reflection + native_translation）。
实际开销增量：output token 约 2.5-3x，综合成本约 1.5-2x（取决于 batch_size 对 prompt 开销的摊薄）。

---

## 二、ASR（语音识别）

### ASR-1: 偶发漏句

- **现象**：频率很低，但会漏掉整句
- **原因**：faster-whisper 本身的局限
- **状态**：暂无解决方案，等待上游改进

### ASR-2: BGM/歌唱场景识别差

- **现象**：BGM 较大的歌唱、音调明显偏离正常讲话的语调，ASR 会漏检
- **补充**：一旦 ASR 能识别到，翻译效果依然理想
- **状态**：暂无解决方案。可考虑 two-pass 方案（常规识别 + 歌唱专用参数），但开发成本高

### ASR-3: 多讲话人问题

- **现象**：多人同时讲话时 ASR 识别质量下降
- **状态**：暂无解决方案。测试过 kotoba-whisper + diarizers，效果不如 faster-whisper large-v3
- **注意（dedup 相关）**：当前 `dedup_adjacent_segments` 基于时间重叠比判定重复段。由于 faster-whisper 是单流输出（无说话人分离），不会产生多讲话人的合法时间重叠，因此 dedup 不会误删。但若未来启用 diarization（说话人分离），ASR 可能输出不同说话人的合法重叠段，届时 dedup 需结合 `speaker_id` 区分——仅对同一说话人的重叠去重，不同说话人的保留

### ASR-4: 漏字、错字与同音异字

- **现象**：日语场景尤其严重——片假名、平假名、汉字三种写法读音可能相同但含义不同
- **影响**：下游 optimize 阶段的 diff 校验可能将合法的同义替换误判为非法修改。已通过片假名→平假名归一化缓解
- **状态**：部分缓解，无法完全解决

### ASR-5: 幻觉输出

- **现象**：VTuber 直播无声序幕（只有画面、无语音）容易产生幻觉文本
- **缓解**：默认关闭 `condition_on_previous_text`，防止幻觉蔓延
- **状态**：已有后处理检测（幻觉检测模块），但无法 100% 消除

### ASR-6: initial_prompt 无通用有效方案

- **现象**：测试了多种写法，绝大部分情况效果变差
- **状态**：暂不使用。详见 [ASR 参数指南](asr_parameters_guide.md)

### ASR-7: Whisper 分块重叠导致内容重复（已修复）

- **现象**：ChunkedASR 分块处理时，相邻 chunk 的重叠窗口（默认 每600s重叠10s）使 Whisper 对同一音频区域产生两次识别结果。表现为：
  1. 内容重复：如原文 "スタンプ" 在 split 后变成 "スタンプスタンプ"
  2. 误听污染：重叠副本可能包含不同的错误识别（如 "趣味" → "スミ"）
  3. 1ms 残渣段：重叠段经 split + optimize_timing 后被压缩为极短段
  4. ASS 字幕崩溃：1ms 段在 ASS centisecond 截断后变成 start==end，触发 assert crash
- **影响范围**：实测 367 个视频中 243 个（66%）存在分块重叠，共 780 对
- **根因**：`dedup_adjacent_segments` 原先只做精确文本匹配，无法捕获子串/后缀重复和同一音频的不同识别结果
- **修复**（`asr_data.py`）：
  1. **`dedup_adjacent_segments` 三级增强**：
     - Case 0: 文本完全相同 → 移除（原有逻辑）
     - Case 1: 文本包含关系（子串）+ 时间重叠 → 移除被包含段
     - Case 2: 时间重叠 ≥50%（同一音频区域）→ 移除较短段
     - Case 3: 时间重叠 <50%（不同内容，时序微偏）→ 保留两段，调整边界
  2. **`ASRData.__init__` 退化段过滤**：过滤 duration < 50ms 的 whisper 幽灵段
- **验证**：367 个视频全量扫描，修复后残余重叠 = 0，ASS crash 风险段 = 0
- **状态**：已修复。已有视频需从 split 阶段重跑才能消除内容重复（ASS crash 已自动修复）

---

## 三、Split（智能断句）

### Split-1: 断句后偶发时间错位

- **现象**：经 split 后字幕偶尔出现轻微时间错位（延后），但无法稳定复现
- **状态**：已修复主要问题（字符级时间插值 + 重叠消除），不排除边缘情况仍有微小偏差

### Split-2: Whisper 分块重叠导致 split 输入含重复段

- **现象**：当 whisper 原始输出包含分块边界重叠段时，LLM split 收到重复内容作为输入，导致断句结果中出现内容重复和 1ms 残渣段
- **状态**：已修复。根因在 ASR 层 dedup 不充分，详见 [ASR-7](#asr-7-whisper-分块重叠导致内容重复已修复)

---

## 四、Translate（翻译）

### Translate-1: chunk 间上下文传递

- **现象**：翻译阶段按 chunk 并行处理，chunk 之间无上下文传递（Optimize 阶段已改为带上下文的线性处理）
- **状态**：理论上添加 chunk 间上下文可提升连贯性，但会将并行改为串行。收益与成本待评估

---

## 五、B 站上传

### Upload-1: 添加合集不稳定（已修复）

- **现象**：视频上传成功，但添加到合集始终失败（API 返回 code=0 但实际未添加）
- **根因**：B站创作中心 `episodes/add` 接口的参数格式与常规 REST API 不同——需要 `sectionId`（驼峰）+ `episodes`（含 aid/cid/title 的对象数组）+ JSON body + csrf 同时在 query string 和 body 中。之前使用的 `{id, aids}` 格式返回"空成功"（code=0 但不生效）
- **修复**：通过逆向 B站创作中心前端 JS 找到正确格式，重写 `add_to_season`、`remove_from_season`。详见 `docs/bilibili_season_api.md`
- **排序功能**：`sort_season_episodes` 已修复。关键是 `section` 对象必须包含 `id/type/seasonId/title` 四个字段，且 body 中不能有 `csrf`

### Upload-2: 视频编号（#N）与时间顺序不一致（已修复）

- **现象**：上传标题中的 `#N` 编号与视频的实际发布时间顺序不符。例如倒数第 2 新的视频显示 `#3` 而非 `#29`
- **根因**：4 个 bug 叠加导致
  1. `database.update_video_playlist_info` 使用 `INSERT OR REPLACE`，每次 sync 都**删除旧行再插入新行**，丢失已分配的 `upload_order_index`
  2. `_assign_upload_order_index` 只处理本次 sync 新增的视频，已有视频即使索引丢失也不会重新分配
  3. `executor` 上传时 `upload_order_index` 缺失会回退到 `playlist_index`（YouTube 的 1=最新逆序），语义与 `upload_order_index`（1=最旧正序）**完全相反**
  4. `backfill_upload_order_index` 保留已有的错误索引，不做全量修正
- **修复**：
  1. `update_video_playlist_info` 改用 `ON CONFLICT UPDATE`，只更新 `playlist_index`，保留 `upload_order_index`
  2. 重写为 `_reassign_upload_order_indices`：每次 sync 全量按 `upload_date` 排序分配 `1~N`
  3. 移除 executor 中对 `playlist_index` 的错误回退
  4. `backfill` 改为全量重分配
- **验证**：12 个新增测试 + 8 个 playlist 全量数据验证通过
- **状态**：已修复
