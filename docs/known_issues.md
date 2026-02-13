# VAT 已知问题与待办

## 一、LLM 成本策略

### 当前方案（2025-02）

各阶段按重要性分配不同模型，控制成本：

| 阶段 | 模型 | API | 相对成本 | 说明 |
|------|------|-----|---------|------|
| Split | gpt-4o-mini | 中转站 | 1x | 断句对模型要求低，后有时间戳修正兜底 |
| Optimize | gpt-5-nano | 中转站 | 6x | 术语纠错靠 custom prompt 引导 |
| Translate | gemini-3-flash | Google 官方 | 40x（开 reflect 约 60-80x） | 翻译质量要求最高，直接面向用户 |

### 成本参考

- 全 gemini-3-flash + reflect：约 $1000+（所有视频）
- 当前方案（split=4o, opt=nano, trans=gemini+reflect）：约 $350-$670
- 关闭 reflect 可再降约 40%（output token 减少 2.5-3x，综合约 1.5-2x）

### 效果对比

- **gpt-4o-mini 全流程**：明显不通顺，不可用于正式翻译
- **gpt-5-nano（optimize+translate）**：基本通顺，但专有名词误报多，缺乏个人风格
- **gemini-3-flash 全流程**：接近人工翻译质量，术语处理正确，风格自然

### Reflect 开销说明

Reflect 不是多次 LLM 调用，而是单次调用中输出 3 个字段（initial_translation + reflection + native_translation）。
实际开销增量：output token 约 2.5-3x，综合成本约 1.5-2x（取决于 batch_size 对 prompt 开销的摊薄）。

---

## 二、ASR 相关问题（当前无法解决）

### ASR-1: 偶发漏句

- **现象**：频率很低，但会漏掉整句
- **原因**：faster-whisper 本身的局限
- **状态**：暂无解决方案，等待上游改进

### ASR-2: BGM/歌唱场景漏检

- **现象**：BGM 较大的歌唱、音调明显偏离正常讲话的语调，ASR 会漏检
- **补充**：一旦 ASR 能捡到，gemini 翻译效果依然理想
- **状态**：暂无解决方案

### ASR-3: 多讲话人问题

- **现象**：多人同时讲话时 ASR 识别质量下降
- **状态**：暂无解决方案，影响断句和翻译的上游数据质量

---

## 三、字幕嵌入问题

### Embed-1: 双层字幕碰撞检测错位 ✅ 已修复

- **现象**：中文字幕使用两个不同 Layer 的事件（发光底层 + 主文字层）实现视觉效果。当中文换行或多条字幕同时出现时，碰撞检测移动了某一层但另一层不动 → 二者错位
- **根因**：
  1. 日文只有 1 层事件，中文有 2 层 → Layer 间碰撞检测不对称
  2. _Base 样式的 Outline/Shadow 比主样式大 → 两层 bounding box 不一致，碰撞偏移量不同
- **修复（3 步）**：
  1. 日文也做两层（Secondary_Base + Secondary），使 Layer 0/1 事件数量和顺序完全对称
  2. _Base 样式的 Outline/Shadow 与主样式**完全一致**（保证 bounding box 相同），发光效果改用 `\blur` 内联标签（后处理特效，不影响碰撞布局）
  3. 对话行顺序调整为"译文在前、原文在后"，碰撞时译文保持原位，原文被推开
- **涉及文件**：`vat/asr/asr_data.py`（`to_ass` 方法）、`vat/resources/subtitle_style/default.txt`
- **技术细节**：见 `docs/subtitle_style_guide.md`

### Embed-2: 字幕尺寸偏小 ✅ 已调整

- **现象**：横屏全屏观看时尺寸合适，但非全屏时偏小
- **调整**：中文字号 55→65，MarginV 60→75；日文 MarginV 125→150
- **涉及文件**：`vat/resources/subtitle_style/default.txt`

### Embed-3: 字幕颜色优化 ✅ 已调整

- **现象**：原配色（淡青 + 深蓝描边）偏"办公风"，不适合直播/动画内容
- **调整**：中文主色改为薄荷蓝 #44DFC4，描边改为近黑 #0A1A1A。发光层使用偏冷白 #DDEEFF 避免色彩对比错觉导致的"发黄"感
- **涉及文件**：`vat/resources/subtitle_style/default.txt`

---
