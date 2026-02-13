# 字幕样式与双层渲染说明

## 一、双层渲染架构

本项目使用 ASS 字幕的双层（Layer）机制实现"发光 + 本体"视觉效果。每条字幕在 ASS 文件中生成两个 Dialogue 事件：

| Layer | 样式 | 作用 | 内联覆盖 |
|-------|------|------|----------|
| **Layer 0**（后方） | `X_Base` | 发光/光晕底层 | `\blur3` |
| **Layer 1**（前方） | `X` | 主文字层，清晰可读 | 无 |

渲染顺序：Layer 0 先渲染（在后），Layer 1 后渲染（在前，覆盖在 Layer 0 上方）。

### 为什么需要两层

单层字幕只能有一种描边效果。双层可以实现：
- Layer 0：白色/亮色文字 + 白色描边，经 `\blur` 模糊后形成柔和光晕
- Layer 1：彩色文字 + 深色描边，保持清晰锐利

两层叠加的视觉效果：彩色文字周围有一圈柔和的白色发光，提升在各种背景下的可读性。

---

## 二、碰撞检测机制

### ASS Layer 与碰撞

ASS 中，**同一 Layer 内**的事件互相参与碰撞检测（时间重叠时自动排开），**不同 Layer 之间**不参与碰撞检测。

### 对称性约束

为保证发光层和主文字层在碰撞排开后仍然完全重合，必须满足：

1. **事件数量对称**：Layer 0 和 Layer 1 的事件数量、顺序完全一致
2. **Bounding box 一致**：`_Base` 样式与主样式的所有影响布局的属性（Fontname、Fontsize、Spacing、Outline、Shadow、MarginL/R/V、ScaleX/Y）必须**完全相同**
3. **发光效果只能用 `\blur`**：`\blur` 是后处理特效，不影响碰撞 bounding box。而 Outline/Shadow 差异会导致 bounding box 不同 → 碰撞偏移量不一致 → 两层错位

### 对话行顺序

同一时间段的多条字幕，ASS 中先出现的事件在碰撞检测中有优先权（保持原位）。

当前策略：**译文在前、原文在后** → 碰撞时译文（中文）保持原位不动，原文（日文）被推开。

每个双语片段生成 4 个事件：

```
Dialogue: 0,...,Default_Base,...,{\blur3}译文    ← Layer 0 译文发光
Dialogue: 1,...,Default,...,译文                  ← Layer 1 译文本体
Dialogue: 0,...,Secondary_Base,...,{\blur3}原文   ← Layer 0 原文发光
Dialogue: 1,...,Secondary,...,原文                ← Layer 1 原文本体
```

---

## 三、样式文件字段说明（双层语境）

样式文件位于 `vat/resources/subtitle_style/`，ASS V4+ Styles 格式。

以 `default.txt` 为例，包含 4 个样式：

| 样式名 | 角色 | 说明 |
|--------|------|------|
| `Secondary_Base` | 原文（日文）发光层 | 白色文字 + 白色描边，经 `\blur` 产生光晕 |
| `Secondary` | 原文（日文）主层 | 白色文字 + 深色描边，清晰可读 |
| `Default_Base` | 译文（中文）发光层 | 偏冷白文字 + 白色描边，经 `\blur` 产生光晕 |
| `Default` | 译文（中文）主层 | 薄荷蓝文字 + 近黑描边，清晰可读 |

### 关键字段含义

```
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,
        Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,
        BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
```

**影响碰撞 bounding box 的字段**（_Base 和主样式必须完全一致）：
- `Fontname`、`Fontsize`：字体和大小
- `Bold`、`Italic`：加粗、斜体
- `ScaleX`、`ScaleY`：水平/垂直缩放
- `Spacing`：字间距
- `Outline`：描边宽度
- `Shadow`：阴影偏移
- `MarginL`、`MarginR`、`MarginV`：边距

**仅影响视觉、不影响碰撞的字段**（_Base 和主样式可以不同）：
- `PrimaryColour`：文字填充色（BGR 格式，`&H00BBGGRR`）
- `OutlineColour`：描边颜色
- `BackColour`：阴影/背景色
- `SecondaryColour`：卡拉OK效果色（正常字幕不生效）

**内联覆盖标签**（对话行内，不影响碰撞）：
- `\blur`：高斯模糊，发光效果的核心。值越大越模糊，当前使用 `\blur3`

### 颜色格式

ASS 使用 BGR 格式：`&H00BBGGRR`（注意不是 RGB）。

示例：
- `&H00C4DF44` → R=0x44, G=0xDF, B=0xC4 → RGB #44DFC4（薄荷蓝）
- `&H001A1A0A` → RGB #0A1A1A（近黑）
- `&H00FFFFFF` → RGB #FFFFFF（白）

---

## 四、修改样式时的注意事项

1. **_Base 和主样式的布局属性必须同步修改**：改了主样式的 Fontsize，_Base 也要改成一样的值
2. **不要在 _Base 上加粗描边/阴影来做发光**：这会导致 bounding box 不一致，碰撞时两层错位
3. **发光强度只能通过 `\blur` 值调整**：在 `asr_data.py` 的 `to_ass()` 方法中修改 `dlg_base` 格式字符串
4. **颜色可以自由调整**：PrimaryColour 和 OutlineColour 不影响碰撞
5. **_Base 自动降级**：如果样式文件中没有 `X_Base`，代码会自动用 `X` 本身作为发光层样式（视觉上无发光，但碰撞检测仍然正确）
6. **多讲话人模式**：`_generate_speaker_styles()` 会自动生成带 `_Base` 后缀的样式，遵循相同的双层架构

---

## 五、当前 default.txt 配色方案

```
中文（Default）：薄荷蓝 #44DFC4 + 近黑描边 #0A1A1A
日文（Secondary）：白色 #FFFFFF + 深蓝灰描边 #203050
发光层：偏冷白 #DDEEFF + \blur3
```

参考分辨率：720p（PlayResX=1280, PlayResY=720），高分辨率视频通过 `_scale_ass_style` 自动缩放。
