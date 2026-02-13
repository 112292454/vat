# TODO

## 待完成

- [ ] Translate 阶段缺少进度输出（低优先级）
- [ ] Upload 模块集成（使用 biliup 作为胶水层）

## 已完成

- [x] 翻译场景提示词优化（split/optimize/translate 的场景提示词）
- [x] ASR 识别效果优化（temperature=0.0、no_speech_threshold=0.6、后处理幻觉清理；人声分离对游戏直播有害，默认关闭）
- [x] 代码结构清理（GalTransl 移除、pipeline 重构、阶段细粒度拆分）
- [x] GPU 自动选择（whisper/ffmpeg/人声分离自动选择空闲显存最多的 GPU）

## 搁置

### 多说话人上下文优化

**状态**：无限期搁置。测试了 kotoba-whisper + diarizers 方案，效果不如 faster-whisper large-v3。

**背景**：当前 Split 和翻译按说话人独立处理，上下文孤立。VTuber 联动场景中不同说话人的对话存在关联性，跨说话人的上下文可能提升翻译质量。

**潜在方案**：
1. Split 保持独立，翻译时提供全局对话历史作为上下文
2. 滑动窗口策略：为每个片段提供时间相邻的 N 条对话（不限说话人）
3. 后处理术语统一：翻译完成后跨说话人统一关键术语

**评估要点**：上下文 token 成本增加、翻译质量提升幅度、是否引入误翻。待当前方案效果验证后再决定。