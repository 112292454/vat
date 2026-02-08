# ASR 参数调优指南

## 1. 日文语音识别评估数据集调研

### 可用数据集

| 数据集 | 规模 | 特点 | 适用性 |
|--------|------|------|--------|
| **ReazonSpeech** | 35,000+小时 | 日本电视节目，自然对话 | ⭐⭐⭐ 最佳 |
| Common Voice JA | ~50小时 | 众包朗读 | ⭐⭐ 一般 |
| JSUT | 10小时 | 单人朗读 | ⭐ 差异大 |
| CSJ | 660小时 | 自发语音 | ⭐⭐⭐ 但需申请 |

### 推荐方案

**方案A: kotoba-whisper评估集** (最快)
```python
from datasets import load_dataset
ds = load_dataset("japanese-asr/ja_asr.reazonspeech_test")
```

**方案B: 自建测试集** (最贴近场景)
- 从已测试视频选3-5分钟片段
- 人工校对作为ground truth

**方案C: 使用ReazonSpeech** 
```python
from datasets import load_dataset
ds = load_dataset("reazon-research/reazonspeech", "tiny", 
                  split="train[:100]", trust_remote_code=True)
```

**方案D: JTubeSpeech** ⭐ 最贴近应用场景
- **来源**: 东京大学 sarulab-speech
- **内容**: 带人工字幕的YouTube日文视频列表
- **GitHub**: https://github.com/sarulab-speech/jtubespeech
- **特点**:
  - 视频有 `sub=True` 标记表示有人工字幕
  - 提供下载脚本，可获取音频+字幕
  - 包含各类视频（对话、独白等）
  - CSV格式列表，可筛选需要的视频

```bash
# 下载视频列表
git clone https://github.com/sarulab-speech/jtubespeech.git
# 查看日文视频列表
cat jtubespeech/data/ja/*.csv | head
# 下载带人工字幕的视频
python scripts/download_video.py ja {csv文件}
```

**方案E: VTuber切り抜き频道**
- Hololive/Nijisanji的切り抜き（剪辑）频道
- 很多有日文字幕（CC字幕）
- 搜索方式: `{VTuber名} 切り抜き 字幕`
- 可用yt-dlp下载字幕: `yt-dlp --write-sub --sub-lang ja {URL}`

---

## 2. ASR 参数详细说明

### 2.1 hallucination_silence_threshold

**作用**: 当检测到超过此秒数的静音时，触发幻觉检测机制

| 值 | 效果 |
|----|------|
| null | 禁用 |
| 2.0 | 2秒静音触发检测 |
| **3.0** (当前) | 3秒静音触发检测 |
| 5.0 | 较宽松 |

**建议**: 2-3秒适合直播场景，过低可能误删正常停顿

---

### 2.2 initial_prompt

**作用**: 给模型提供上下文提示，引导识别风格

**示例**:
```yaml
initial_prompt: "This is a Japanese VTuber livestream with gaming commentary."
```

| 场景 | 推荐prompt |
|------|-----------|
| VTuber直播 | "Japanese VTuber livestream" |
| 游戏实况 | "Japanese gaming commentary" |
| 动漫解说 | "Japanese anime discussion" |
| 通用 | "" (空) |

**注意**: 
- 建议使用英文（Whisper训练时使用英文prompt）
- 不宜过长，通常1-2句即可
- 可能提高特定领域词汇识别率

---

### 2.3 no_speech_threshold

**作用**: 判定音频片段为"无语音"的概率阈值

| 值 | 效果 |
|----|------|
| 0.1 | 极敏感，容易保留噪音 |
| **0.2** (当前) | 较敏感 |
| 0.4 | 中等 |
| 0.6 | 严格，可能丢失轻声 |

**建议**: 
- 有BGM的视频: 0.3-0.4（减少BGM误识别）
- 安静录音: 0.2-0.3

---

### 2.4 condition_on_previous_text

**作用**: 是否基于前文预测当前文本

| 值 | 优点 | 缺点 |
|----|------|------|
| true | 上下文更连贯 | 幻觉可能传播 |
| **false** (当前) | 幻觉不传播 | 上下文可能断裂 |

**建议**: 保持false，因为：
- 开头静音产生的幻觉不会影响后续
- 直播内容本身话题跳跃，不需要强连贯

---

### 2.5 temperature

**作用**: 采样温度，控制输出多样性

**当前配置**: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`

**机制**: 
- 从0.0开始尝试
- 如果输出质量不佳（重复/压缩比过高），自动提高温度重试
- 直到找到满意结果或用完所有温度

**建议**: 保持默认的温度回退序列

---

### 2.6 log_prob_threshold

**作用**: 对数概率阈值，过滤低置信度输出

| 值 | 效果 |
|----|------|
| -0.5 | 严格，只保留高置信度 |
| **-1.0** (当前) | 中等 |
| -2.0 | 宽松，保留更多内容 |

**建议**: 
- 高质量录音: -0.5 到 -1.0
- 嘈杂音频: -1.0 到 -1.5

---

### 2.7 compression_ratio_threshold

**作用**: 压缩比阈值，检测重复文本

**原理**: 重复文本压缩后体积小，压缩比高

| 值 | 效果 |
|----|------|
| 2.0 | 严格，减少重复 |
| **2.4** (当前) | 默认 |
| 3.0 | 宽松，保留更多 |

**建议**: 
- 如果输出有大量重复: 降低到2.0
- 如果内容被过度过滤: 提高到2.6-3.0

---

### 2.8 repetition_penalty

**作用**: 重复惩罚，降低重复token的概率

| 值 | 效果 |
|----|------|
| 1.0 | 无惩罚 |
| **1.2** (当前) | 轻度惩罚 |
| 1.5 | 中度惩罚 |
| 2.0 | 强惩罚（可能影响正常重复） |

**注意**: 日文中有很多正常的重复表达（如「いいね、いいね」），惩罚过重会影响

---

## 3. 参数调优建议

### 当前配置评估

```yaml
# 当前配置
hallucination_silence_threshold: 3    # ✓ 合理
condition_on_previous_text: false     # ✓ 推荐
temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # ✓ 默认最佳
compression_ratio_threshold: 2.4      # ✓ 默认
log_prob_threshold: -1.0              # ✓ 默认
no_speech_threshold: 0.2              # 可尝试0.3
initial_prompt: ""                    # 可尝试添加场景提示
repetition_penalty: 1.2               # ✓ 合理
```

### 可尝试的优化

1. **initial_prompt**: 添加 `"Japanese VTuber livestream commentary"`
2. **no_speech_threshold**: 提高到 0.3（减少BGM误识别）
3. **compression_ratio_threshold**: 如有重复问题，降低到 2.2

---

## 4. 已知问题与陷阱

### 4.1 temperature=0.0 + condition_on_previous_text=False 组合问题

**问题描述**: 当同时设置 `temperature: 0.0`（固定温度）和 `condition_on_previous_text: false` 时，Whisper 在遇到某些音频片段时会产生**大量重复字符的幻觉**。

**典型表现**:
```
[30.1s -> 51.8s] はい、はい、ああうううううううううううううううううう...（数百个重复字符）
```

**根因分析** :
- `temperature=0.0` 使输出完全确定性
- `condition_on_previous_text=false` 使每个片段独立解码
- 两者组合时，模型在遇到模糊/困难音频时无法通过上下文或随机性"逃逸"，陷入重复循环

**测试结果**:
| 组合 | 结果 |
|------|------|
| condition=False + temp=0.0 | ❌ 产生重复幻觉 |
| condition=False + temp=[0.0,0.2,...] | ✓ 正常 |
| condition=True + temp=0.0 | ✓ 正常 |

**解决方案**: 使用温度回退列表 `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` 代替固定 `0.0`

**权衡**:
- 温度回退可能导致结果略有不稳定（不同运行可能有细微差异）
- 但相比大量幻觉，这是可接受的代价
- 后处理模块可进一步清理残留问题

---

## 5. 评估指标

### CER (Character Error Rate)
```
CER = (S + D + I) / N

S = 替换数
D = 删除数  
I = 插入数
N = 参考文本字符数
```

### 计算示例
```python
import jiwer

# 日文需要先分词/分字
reference = "こんにちは世界"
hypothesis = "こんにちわ世界"

# 字符级CER
cer = jiwer.cer(list(reference), list(hypothesis))
```
