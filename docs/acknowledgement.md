# 致谢

## 核心参考项目

### [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)

VAT 的翻译引擎（反思翻译）、字幕优化、智能断句（分块 + LLM）、ASS 渲染的实现均基于此项目，在其基础上做了修改和扩展。

### [GalTransl](https://github.com/xd2333/GalTransl)

VideoCaptioner 的翻译引擎基于 GalTransl。VAT 间接受益于其翻译思路。

### [VoiceTransl](https://github.com/shinnpuru/VoiceTransl)

项目最初的想法来源。VoiceTransl 面向 GUI（Gradio）的用户侧设计不适合服务器端批量处理，因此创建了 VAT。早期版本的 ASR 和翻译部分复用过其代码，后已完全替换。

## 依赖库

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — 语音识别引擎
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频下载
- [biliup](https://github.com/biliup/biliup) — B 站上传（合集管理基于其 API 自行实现）

## 模型与参考实现

- [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) — 人声分离模型
- [WhisperJAV](https://github.com/meizhong986/WhisperJAV.git) — ASR 后处理参考（幻觉检测、重复清理、日语标点处理）
- [faster-whisper-GUI #159](https://github.com/CheshireCC/faster-whisper-GUI/issues/159) — Whisper 参数调优参考