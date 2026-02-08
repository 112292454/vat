
本项目想法来源：https://github.com/shinnpuru/VoiceTransl?tab=readme-ov-file
因为这个自动翻译的实现主要面向gui的web开发（gradio base），并且是用户侧，难以实现在服务器上后台的、自动的进行大批量的翻译工作，所以我创建了这个仓库

其中，最初版的翻译部分和asr部分都是复用它的代码，不过后来完全改动删除了（ast部分目前是ai完成，翻译部分是复用了VideoCaptioner的实现）

====================

目前，翻译部分、ast之后的源语言字幕增强、翻译增强部分，均来自：https://github.com/WEIFENG2333/VideoCaptioner

视频上传来自于：https://github.com/dreammis/social-auto-upload 上传完全没有改动逻辑，仅写了一个pipeline 封装，提供了进度管理，以及视频信息的llm优化（翻译部分均将上述原仓库代码提取复制重构到了我们的仓库中，因此无需克隆他们，但是这个上传是要克隆的）
—— update：我个人对此仓库的b站部分实现表示谴责
纯粹就是写了一个ai一分钟能完成的cookie获取（还tm是by exe），然后无脑调biliup这个库的api  我还以为是他自己照着那个b站 api copllect写的呢。这还能800star？
所以我现在是用biliup的库解决上传功能，然后参考api自己实现了合集的管理

================

asr部分，关于fast whisper的参数参考了：https://github.com/CheshireCC/faster-whisper-GUI/issues/159

复用了：https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model.git 的人声分离模型

参考了https://github.com/meizhong986/WhisperJAV.git 的后处理实现（幻觉检测、重复清理、日语特殊处理、语音动态分块）