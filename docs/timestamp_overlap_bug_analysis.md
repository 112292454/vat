# VAT 时间戳重叠问题分析报告

> **状态：已解决**。本文档描述的是旧版 `_realign_timestamps` 使用 segment 边界时间的问题。该算法已重写为基于 `difflib.SequenceMatcher` 的 diff 容错对齐，从根本上消除了此问题。此外，后续又修复了 diff 对齐算法中相邻段重叠导致的时间戳倒挂问题（在生成段落后增加了重叠消除步骤）。详见 `vat/asr/readme.md` 第 13 节。

## 1. 问题确认（历史记录）

员工报告的问题**确实存在**。经代码分析和实验验证，问题已定位。

### 1.1 现象描述

- **raw 阶段**（Whisper ASR 输出）：存在少量时间重叠（这是 Whisper VAD 的正常行为，多人说话时会产生重叠片段）
- **split/optimized 阶段**：出现**大量**相邻条目时间重叠，量级为秒级

### 1.2 影响范围

- `vat/pipeline/executor.py` 中的 `_realign_timestamps()` 方法
- `vat/asr/chunked_split.py` 中的 `_realign_timestamps()` 方法（相同逻辑）

---

## 2. 根本原因

### 2.1 问题代码

```python
# vat/pipeline/executor.py:533-543 和 vat/asr/chunked_split.py:151-161
char_to_time = []
for seg in original_asr.segments:
    text = seg.text.strip()
    duration = seg.end_time - seg.start_time
    for i, char in enumerate(text):
        progress = i / len(text) if len(text) > 1 else 0
        char_time = seg.start_time + duration * progress
        # BUG: 存储的是整个 segment 的边界，而不是字符级时间
        char_to_time.append((char, char_time, seg.start_time, seg.end_time))

# 分配时间戳时
start_time = char_to_time[char_idx][2]   # ← 使用整个 segment 的 start_time
end_time = char_to_time[end_idx][3]       # ← 使用整个 segment 的 end_time
```

### 2.2 问题分析

当 LLM 将一个原始 ASR segment 断句成多个部分时，所有部分都被分配了**相同的时间戳边界**。

**示例**：
- 原始 segment: `"ABCDEFGH"` (start=0ms, end=10000ms)
- LLM 断句结果: `["ABC", "DEFGH"]`
- **错误的**时间戳分配:
  - `"ABC"`: start=0, end=10000
  - `"DEFGH"`: start=0, end=10000 ← **完全重叠！**

### 2.3 正确逻辑

应该使用字符级插值时间 (`char_time`) 来计算精确的时间边界：

**正确的**时间戳分配:
- `"ABC"`: start=0, end=3750
- `"DEFGH"`: start=3750, end=10000 ← **无重叠**

---

## 3. 修复方案

### 3.1 修改 `char_to_time` 数据结构

存储每个字符的**精确起止时间**，而不是整个 segment 的边界：

```python
char_to_time = []
for seg in original_asr.segments:
    text = seg.text.strip()
    if not text:
        continue
    duration = seg.end_time - seg.start_time
    text_len = len(text)
    for i, char in enumerate(text):
        # 计算字符的起始时间
        char_start = seg.start_time + duration * (i / text_len)
        # 计算字符的结束时间
        char_end = seg.start_time + duration * ((i + 1) / text_len)
        char_to_time.append((char, char_start, char_end))
```

### 3.2 修改时间戳分配逻辑

使用字符级时间而不是 segment 边界：

```python
start_time = char_to_time[char_idx][1]   # 第一个字符的 char_start
end_time = char_to_time[end_idx][2]       # 最后一个字符的 char_end
```

---

## 4. 受影响的文件

| 文件 | 方法 | 说明 |
|------|------|------|
| `vat/pipeline/executor.py` | `_realign_timestamps()` | 全文断句时的时间戳重分配 |
| `vat/asr/chunked_split.py` | `_realign_timestamps()` | 分块断句时的时间戳重分配 |

---

## 5. 验证方案

修复后运行以下验证：

```python
# 检查相邻字幕时间戳是否重叠
for i in range(1, len(segments)):
    prev_end = segments[i-1].end_time
    curr_start = segments[i].start_time
    assert curr_start >= prev_end, f"时间戳重叠: #{i} start={curr_start} < prev_end={prev_end}"
```

---

## 6. 结论

问题报告**准确**。根本原因是 `_realign_timestamps()` 方法在重新分配时间戳时，错误地使用了原始 segment 的边界时间，而不是基于字符位置插值计算的精确时间。

修复后，断句产生的字幕条目将拥有正确的、不重叠的时间戳。
