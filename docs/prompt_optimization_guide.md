# VAT 提示词优化完整指南（零代码改动）

本文档详细列出 VAT 项目中所有可以通过修改提示词（prompt）来优化 vtuber 直播字幕质量的位置，包括：
- 提示词模板文件（`.md`）
- 配置文件中的自定义提示词字段
- 各环节的优化策略和示例

---

## 📋 目录

1. [提示词位置总览](#提示词位置总览)
2. [环节 1：Whisper ASR 阶段](#环节-1whisper-asr-阶段)
3. [环节 2：智能断句（Split）阶段](#环节-2智能断句split阶段)
4. [环节 3：字幕优化（Optimize）阶段](#环节-3字幕优化optimize阶段)
5. [环节 4：翻译（Translate）阶段](#环节-4翻译translate阶段)
6. [vtuber 场景优化示例](#vtuber-场景优化示例)

---

## 提示词位置总览

VAT 项目中涉及 LLM 提示词的位置：

| 环节 | 提示词位置 | 类型 | 可修改性 |
|------|-----------|------|---------|
| **Whisper ASR** | `config/default.yaml` → `asr.initial_prompt` | 配置字段 | ✅ 零代码 |
| **智能断句** | `vat/llm/prompts/split/sentence.md` | 模板文件 | ✅ 零代码 |
| **智能断句** | `vat/llm/prompts/split/semantic.md` | 模板文件 | ⚠️ 需改代码激活 |
| **智能断句（场景）** | `vat/llm/scenes.yaml` → `scenes[i].prompts.split` | 配置字段 | ✅ 零代码 |
| **字幕优化** | `vat/llm/prompts/optimize/subtitle.md` | 模板文件 | ✅ 零代码 |
| **字幕优化** | `config/default.yaml` → `translator.llm.optimize.custom_prompt` | 配置字段 | ✅ 零代码 |
| **字幕优化（场景）** | `vat/llm/scenes.yaml` → `scenes[i].prompts.optimize` | 配置字段 | ✅ 零代码 |
| **翻译（标准）** | `vat/llm/prompts/translate/standard.md` | 模板文件 | ✅ 零代码 |
| **翻译（反思）** | `vat/llm/prompts/translate/reflect.md` | 模板文件 | ✅ 零代码 |
| **翻译（降级）** | `vat/llm/prompts/translate/single.md` | 模板文件 | ✅ 零代码 |
| **翻译自定义** | `config/default.yaml` → `translator.llm.custom_prompt` | 配置字段 | ✅ 零代码 |
| **翻译（场景）** | `vat/llm/scenes.yaml` → `scenes[i].prompts.translate` | 配置字段 | ✅ 零代码 |

---

本文档是总结了该视频翻译项目中，用到LLM各处的提示词写法

并且目前尚未实施针对场景（Vtuber直播）的优化

因此，下面的各个小节的格式为：

```
part1： 某环节当前的提示词是怎么写的（来自某个经典项目）

part2：优化建议——我们要对于这部分做怎样的修改，优化，需要你去完成对上述环节当前提示词的修改。或：大致的一个优化方向——只是说明了一个大概的想法，但是具体的优化写法尚未规划，需要你首先规划上述提示词怎么改，或额外添加的custom提示词应该怎么设计。然后与用户商讨这个思路是否可行，最后在完成具体的提示词内容
```

请你在后续完善时，根据上述说明理解，并协助完成工作。


## 环节 1：智能断句（Split）阶段

### 位置 1：断句长度配置
**配置字段**：`config/default.yaml` → `asr.split.*`

```yaml
asr:
  split:
    enable: true
    mode: "sentence"  # 当前代码固定使用 sentence，semantic 模式需改代码激活
    max_words_cjk: 24      # 中文/日文每句最大字符数
    max_words_english: 18  # 英文每句最大单词数
    model: "gpt-4o-mini"   # 断句使用的 LLM 模型
    
    # 分块配置（新增）- 用于处理长视频
    enable_chunking: true          # 是否启用分块处理
    chunk_size_sentences: 50       # 每块的句子数
    chunk_overlap_sentences: 5     # 块间重叠的句子数
    chunk_min_threshold: 30        # 小于此值不分块，直接全文处理
```

#### vtuber 场景优化建议

| 场景 | max_words_cjk | max_words_english | 原因 |
|------|--------------|------------------|------|
| **游戏直播** | 16-20 | 12-15 | 快节奏，需要短句快速显示 |
| **闲聊直播** | 24-28 | 18-22 | 正常语速，可以稍长保持语义完整 |
| **ASMR** | 12-18 | 10-14 | 轻声细语，短句营造氛围 |
| **教学/解说** | 28-32 | 20-25 | 信息密度高，长句保持完整性 |

### 位置 1：断句提示词模板
**模板文件**：`vat/llm/prompts/split/sentence.md`（当前使用）

#### 当前模板结构
```markdown
你是一位专业的字幕分段专家。你的任务是将未分段的连续文本按语义断点拆分,使字幕便于阅读和理解。

<instructions>
1. 在语义自然断点处插入 <br>(可在句内、句间灵活分段)
2. 字数限制:
   - CJK语言(中文、日语、韩语等):每段≤ $max_word_count_cjk 字
   - 拉丁语言(英语、法语等):每段≤ $max_word_count_english 词
3. 每段需包含完整语义,避免过短碎片
4. 原文保持不变:不增删改,仅插入 <br>
5. 直接输出分段文本,无需解释
</instructions>

<output_format>
直接输出分段后的文本,段与段之间用 <br> 分隔,不要包含任何其他内容或解释。
</output_format>

<examples>
<example>
<input>
大家好今天我们带来的3d创意设计作品是进制演示器我是来自中山大学附属中学的方若涵我是陈欣然我们这一次作品介绍分为三个部分第一个部分提出问题第二个部分解决方案第三个部分作品介绍当我们学习进制的时候难以掌握老师教学 也比较抽象那有没有一种教具或演示器可以将进制的原理形象生动地展现出来
</input>
<output>
大家好<br>今天我们带来的3d创意设计作品是<br>进制演示器<br>我是来自中山大学附属中学的方若涵<br>我是陈欣然<br>我们这一次作品介绍分为三个部分<br>第一个部分提出问题<br>第二个部分解决方案<br>第三个部分作品介绍<br>当我们学习进制的时候难以掌握<br>老师教学也比较抽象<br>那有没有一种教具或演示器<br>可以将进制的原理形象生动地展现出来
</output>
</example>

<example>
<input>
the upgraded claude sonnet is now available for all users developers can build with the computer use beta on the anthropic api amazon bedrock and google cloud's vertex ai the new claude haiku will be released later this month
</input>
<output>
the upgraded claude sonnet is now available for all users<br>developers can build with the computer use beta<br>on the anthropic api amazon bedrock and google cloud's vertex ai<br>the new claude haiku will be released later this month
</output>
</example>
</examples>

```

代码实现：
```python
# 分块
def _create_chunks(
    self, segments: List[ASRDataSeg]
) -> List[Tuple[List[ASRDataSeg], int, int]]:
    """
    创建带 overlap 的分块
    
    Returns:
        List[(chunk_segments, start_idx, end_idx)]
    """
    chunks = []
    total = len(segments)
    i = 0
    
    while i < total:
        end = min(i + self.chunk_size, total)
        chunk_segs = segments[i:end]
        chunks.append((chunk_segs, i, end - 1))
        
        # 下一块的起点：当前块末尾 - overlap
        i = end - self.overlap
        
        # 如果剩余片段太少（不足 overlap 的一半），合并到当前块
        if i > 0 and total - i < self.overlap // 2:
            break
    
    return chunks

…………………………

chunks = self._create_chunks(segments)
logger.info(f"将 {total} 个片段分为 {len(chunks)} 块处理")

# 逐块断句（串行，因为需要前块结果）
all_split_segments = []

for i, (chunk_segs, start_idx, end_idx) in enumerate(chunks, 1):
    if progress_callback:
        progress_callback(f"断句进度: {i}/{len(chunks)} 块")
    
    logger.info(f"处理第 {i}/{len(chunks)} 块 (片段 {start_idx}-{end_idx})")
    
    # 合并该块的文本
    chunk_text = "".join(seg.text for seg in chunk_segs)
    
    # 调用 LLM 断句
    split_texts = split_by_llm(
        chunk_text,
        model=self.model,
        max_word_count_cjk=self.max_word_count_cjk,
        max_word_count_english=self.max_word_count_english,
        scene_prompt=self.scene_prompt,
    )
    
    # 重新分配时间戳
    chunk_asr = ASRData(chunk_segs)
    split_asr = self._realign_timestamps(chunk_asr, split_texts)
    
    # 合并结果（处理 overlap）
    if i == 1:
        # 第一块，全部保留
        all_split_segments.extend(split_asr.segments)
    else:
        # 后续块：跳过 overlap 部分，但保留最后一句的优化版本
        # 策略：A的最后一句丢弃（无下文），用B的第一句（有上下文）
        # 但如果 overlap=1，就还是用 A 的
        if self.overlap == 1:
            # overlap=1 时用前块的结果
            all_split_segments.extend(split_asr.segments[self.overlap:])
        else:
            # overlap>1 时，丢弃前块最后一句，用当前块的第一句
            # 先移除前块的最后一句
            all_split_segments = all_split_segments[:-1]
            # 当前块跳过前 overlap-1 句，从第 overlap 句开始（即保留当前块的第一句）
            all_split_segments.extend(split_asr.segments[self.overlap - 1:])
```


#### 优化建议：针对 vtuber 直播特点

**修改点 1**：在 `<instructions>` 部分增加直播特有规则

```markdown
<instructions>
1. 在句子边界处插入 <br> (句号、逗号、分号等标点符号应出现的位置)
2. 分割段的字数限制:
   - CJK语言(中文、日语、韩语等):每段≤ ${max_word_count_cjk} 字
   - 拉丁语言(英语、法语等):每段≤ ${max_word_count_english} 词
3. 在遵循字数限制的同时，保持每个分句的意思完整
4. 原文保持不变:不增删改,不要翻译，仅插入 <br>
5. 倒计时（每个数字进行分割）、关键信息揭示前及需要强调的位置需要进行适当分割

<!-- 新增：vtuber 直播专用规则 -->
6. 口癖和语气词（如"啊"、"嗯"、"哈"、"ww"）通常与前句合并，除非单独成句作为强调
7. 游戏术语、技能名称保持完整，不在中间断句
8. 重复感叹（如"啊啊啊"、"草草草"）可独立成段以体现情绪强度
9. 主播与观众互动（如"谢谢xx的SC"）优先独立成句
</instructions>
```

注意 上述新增内容不是最终的修改版，但是可以参考采纳。请你根据vtuber直播情景，完成具体的修改内容

需要参考的信息如下：

默认可以处理不同的任务（包括各种语言的各个vtuber、油管博主等），但是更倾向于日本人vtuber（不要显式写出来）
默认的
分词应该灵活参考语义，结合上下文与上述背景，但不得修改内容（即便可能原文看起来有误——例如吧人名识别成某个名词导致不连贯）。因为后续还有字幕内容优化环节

### 位置 2：场景特定的断句提示词（新增）
**配置文件**：`vat/llm/scenes.yaml` → `scenes[i].prompts.split`

每个场景可配置独立的断句提示词，补充全局的 sentence.md 模板。执行流程：
1. 先加载全局模板 `split/sentence.md`
2. 在下载时识别视频场景（由 LLM 根据标题和简介判断）
3. 翻译前从 `scenes.yaml` 加载对应场景的 `split` 提示词
4. 场景提示词将与全局模板合并后发送给 LLM

#### 示例：游戏直播的断句规则

（此处第一次出现场景特定的提示词，顾展示一个完整的yml结构。后续将只展示相关位置）

```yaml

scenes:
  - id: "gaming"
    name: "游戏直播"
    description: "Playing video games with commentary and reactions. Includes competitive gaming, casual gaming, game tutorials."
    keywords:
      - "game"
      - "gameplay"
      - "boss"
      - "level"
      - "play"
      - "ゲーム"
      - "プレイ"
    
    prompts:
      # 断句阶段的场景特定提示词
      split: |
        ## 游戏直播断句规则
        - 战斗/紧张时刻使用更短的分句，突出节奏感
        - 感叹词和情绪爆发（"啊啊啊"、"草"）单独成句或紧跟前句
        - 游戏术语不要从中间断开（如"Boss战"、"技能CD"保持完整）
        - 连续快速反应可以稍微超出字数限制，保持紧凑感
      
      # 翻译阶段的场景特定提示词（补充全局 custom_prompt）
      translate: |
        ## 游戏直播场景特点
        - 快节奏对话，术语准确性优先于文学性
        - 战斗/紧张时刻：短句、有力、感叹词保留情绪强度
        - 游戏术语统一：Boss/HP/MP/Skill 等保持社区习惯
        - 失败/成功的情绪表达要强烈（"寄了"、"草"、"牛"、"绝杀"）
        - 数值直接保留（"10% HP" → "10%血" 或 "没血了"）
      
      # 优化阶段的场景特定提示词
      optimize: |
        ## 游戏直播优化规则
        - 保留游戏术语的原文或社区通用译名
        - 保留情绪化重复（"啊啊啊"、"草草草"）
        - 移除犹豫词（"あの"、"ええと"），但保留战斗时的语气词

```

那么此处，我们主要是完成上述的split部分



#### 应用场景：

system prompt即上一节的示例内容

此节内容将这样被应用：

```python
system_prompt = get_prompt(
    prompt_path,
    max_word_count_cjk=max_word_count_cjk,
    max_word_count_english=max_word_count_english,
)

# 插入场景特定提示词（如果有）
if scene_prompt:
    # 在 </instructions> 之后插入场景提示词
    insert_marker = "</instructions>"
    if insert_marker in system_prompt:
        scene_block = f"\n\n<scene_specific>\n{scene_prompt.strip()}\n</scene_specific>"
        system_prompt = system_prompt.replace(
            insert_marker, 
            insert_marker + scene_block
        )
    else:
        # 如果没有找到标记，追加到末尾
        system_prompt = f"{system_prompt}\n\n<scene_specific>\n{scene_prompt.strip()}\n</scene_specific>"

user_prompt = (
    f"Please use multiple <br> tags to separate the following sentence:\n{text}"
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

```
（注意，此处即后续所有位置的scene_prompt均不包含下面yml中的id、关键词等。仅包含prompts.split或者其他阶段缩写的内容）

那么，请你针对此环节，完成下面多个场景的split custom提示词优化：

```yaml
# vtuber 直播场景配置
# 每个场景包含：ID、名称、描述、关键词、各阶段的额外提示词

scenes:
  - id: "gaming"
    name: "游戏直播"
    description: "Playing video games with commentary and reactions. Includes competitive gaming, casual gaming, game tutorials."
    keywords:
      - "game"
      - "gameplay"
      - "boss"
      - "level"
      - "play"
      - "ゲーム"
      - "プレイ"
    （上述内容再分词阶段不会）
    prompts:
      # 断句阶段的场景特定提示词
      split: |
        ## 游戏直播断句规则
        - 战斗/紧张时刻使用更短的分句，突出节奏感
        - 感叹词和情绪爆发（"啊啊啊"、"草"）单独成句或紧跟前句
        - 游戏术语不要从中间断开（如"Boss战"、"技能CD"保持完整）
        - 连续快速反应可以稍微超出字数限制，保持紧凑感

(其他阶段略，后面再处理，下同)

  - id: "chatting"
    name: "闲聊直播"
    description: "Casual conversation, Q&A session, daily life sharing, viewer interaction."
    keywords:
      - "chat"
      - "talk"
      - "雑談"
      - "freetalk"
      - "zatsudan"
    
    prompts:
      split: |
        ## 闲聊直播断句规则
        - 保持自然的口语节奏，不要过度切分
        - 语气词（"嗯"、"啊"、"呢"）可以跟随前句，不必单独成句
        - 对观众的互动句保持完整（如"大家好"、"谢谢xxx"）
        - 话题转换处是自然断点

  - id: "asmr"
    name: "ASMR放松"
    description: "ASMR content with soft speaking, whispering, relaxation, sleep aid, trigger sounds."
    keywords:
      - "asmr"
      - "relax"
      - "sleep"
      - "whisper"
      - "耳かき"
      - "癒し"
    
    prompts:
      split: |
        ## ASMR 断句规则
        - 短句为主，营造轻柔节奏
        - 拟声词（ささ、ふわふわ）可单独成句或紧跟动作描述
        - 停顿和呼吸音标记是自然断点
        - 不要把舒缓的长句强行拆散，保持意境完整


  - id: "singing"
    name: "歌回直播"
    description: "Singing stream, karaoke, music performance, song covers."
    keywords:
      - "sing"
      - "song"
      - "karaoke"
      - "歌"
      - "カラオケ"
      - "utawaku"
    
    prompts:
      split: |
        ## 歌回断句规则
        - 歌词部分按歌曲原有节奏断句（参考原曲分句）
        - 说话部分按正常口语断句
        - 歌曲名、歌手名保持完整不要拆开
        - 唱完后的感叹/评论可以稍长，保持情绪完整


  - id: "teaching"
    name: "教学解说"
    description: "Tutorial, educational content, knowledge sharing, explanation streams."
    keywords:
      - "tutorial"
      - "how to"
      - "guide"
      - "explain"
      - "教学"
      - "講座"
    
    prompts:
      split: |
        ## 教学断句规则
        - 按逻辑步骤断句（"首先"、"然后"、"最后"是自然断点）
        - 专业术语和概念名词保持完整
        - 数字、公式、代码片段不要从中间断开
        - 因果关系句可以稍长，保持逻辑完整性
      

# 默认场景（无法判断时使用）
default_scene: "chatting"

```
请你考虑我们此前描述的内容，确定这部分我们分别要为每个场景做怎样的分词额外提示（依然注意此阶段原则：不改变内容，仅将原本asr可能零碎或者大段的文本做出恰当长度、语法结构、习惯的分词）。

上述示例内容可以修改。

首先对位置一做出修改建议，然后对位置二各个情景给出讨论

---

## 环节 2：字幕优化（Optimize）阶段


注意，此处对于语言的假设和之前一样，仍然默认是日语直播的转录字幕

下述环节在代码中的调用如下：

```python

    def _optimize_subtitle(self, asr_data: ASRData) -> ASRData:
        """
        内部方法：优化字幕内容
        复用 BaseTranslator 的并发框架
        """
        assert asr_data is not None, "调用契约错误: asr_data 不能为空"
        
        if not asr_data.segments:
            logger.warning("字幕内容为空，跳过优化")
            return asr_data

        # 转换为字典格式
        subtitle_dict = {str(i): seg.text for i, seg in enumerate(asr_data.segments, 1)}
        
        # 分批处理（使用基类的批量大小）
        items = list(subtitle_dict.items())
        chunks = [
            dict(items[i : i + self.batch_num])
            for i in range(0, len(items), self.batch_num)
        ]

        # 并行优化（复用线程池）
        optimized_dict: Dict[str, str] = {}
        futures = []
        
        if not self.executor:
            raise ValueError("线程池未初始化")
        
        for chunk in chunks:
            future = self.executor.submit(self._optimize_chunk, chunk)
            futures.append((future, chunk))

        # 收集结果
        for future, chunk in futures:
            if not self.is_running:
                break
            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                logger.error(f"优化批次失败: {e}")
                optimized_dict.update(chunk)  # 失败时保留原文

        # 验证数量一致性
        assert len(optimized_dict) == len(subtitle_dict), \
            f"逻辑错误: 优化后字幕数量 ({len(optimized_dict)}) 与原文数量 ({len(subtitle_dict)}) 不一致"

        # 创建新 segments
        new_segments = [
            ASRDataSeg(
                text=optimized_dict.get(str(i), seg.text),
                start_time=seg.start_time,
                end_time=seg.end_time,
                translated_text=seg.translated_text
            )
            for i, seg in enumerate(asr_data.segments, 1)
        ]
        
        assert len(new_segments) == len(asr_data.segments), \
            f"逻辑错误: 生成的 segments 数量 ({len(new_segments)}) 与原文数量 ({len(asr_data.segments)}) 不一致"
        
        return ASRData(new_segments)

    def _optimize_chunk(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        """
        优化单个字幕批次
        使用 Agent Loop 自动验证和修正
        """
        start_idx = next(iter(subtitle_chunk))
        end_idx = next(reversed(subtitle_chunk))
        logger.debug(f"正在优化字幕：{start_idx} - {end_idx}")
        
        prompt = get_prompt("optimize/subtitle")
        
        user_prompt = (
            f"Correct the following subtitles. Keep the original language, do not translate:\n"
            f"<input_subtitle>{json.dumps(subtitle_chunk, ensure_ascii=False)}</input_subtitle>"
        )

        if self.optimize_prompt:
            user_prompt += f"\nReference content:\n<reference>{self.optimize_prompt}</reference>"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_result = subtitle_chunk
        
        # Agent Loop
        for step in range(self.MAX_STEPS):
            try:
                response = call_llm(messages=messages, model=self.model, temperature=0.2)
                
                result_text = response.choices[0].message.content
                if not result_text:
                    raise ValueError("LLM返回空结果")
                
                result_dict = json_repair.loads(result_text)
                if not isinstance(result_dict, dict):
                    raise ValueError(f"LLM返回结果类型错误，期望dict，实际{type(result_dict)}")
                
                last_result = result_dict
                
                # 验证结果
                is_valid, error_message = self._validate_optimization_result(
                    original_chunk=subtitle_chunk,
                    optimized_chunk=result_dict
                )
                
                if is_valid:
                    return result_dict
                
                # 验证失败，添加反馈
                logger.warning(f"优化验证失败，开始反馈循环 (第{step + 1}次尝试): {error_message}")
                messages.append({"role": "assistant", "content": result_text})
                messages.append({
                    "role": "user",
                    "content": f"Validation failed: {error_message}\n"
                              f"Please fix the errors and output ONLY a valid JSON dictionary.DO NOT REPLY ANY ADDITIONEL EXPLANATION OR OTHER PREVIOUS TEXT."
                })
                
            except Exception as e:
                logger.warning(f"优化批次尝试 {step+1} 失败: {e}")
                if step == self.MAX_STEPS - 1:
                    return last_result
        
        return last_result

    def _validate_optimization_result(
        self, original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
    ) -> Tuple[bool, str]:
        """验证优化结果"""
        expected_keys = set(original_chunk.keys())
        actual_keys = set(optimized_chunk.keys())

        # 检查键匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_parts = []
            
            if missing:
                error_parts.append(f"Missing keys: {sorted(missing)}")
            if extra:
                error_parts.append(f"Extra keys: {sorted(extra)}")

            error_msg = (
                "\n".join(error_parts) + f"\nRequired keys: {sorted(expected_keys)}\n"
                f"Please return the COMPLETE optimized dictionary with ALL {len(expected_keys)} keys."
            )
            return False, error_msg

        # 检查改动是否过大
        excessive_changes = []
        for key in expected_keys:
            original_text = original_chunk[key]
            optimized_text = optimized_chunk[key]

            original_cleaned = re.sub(r"\s+", " ", original_text).strip()
            optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()

            matcher = difflib.SequenceMatcher(None, original_cleaned, optimized_cleaned)
            similarity = matcher.ratio()
            similarity_threshold = 0.3 if count_words(original_text) <= 10 else 0.5

            if similarity < similarity_threshold:
                excessive_changes.append(
                    f"Key '{key}': similarity {similarity:.1%} < {similarity_threshold:.0%}. "
                    f"Original: '{original_text}' → Optimized: '{optimized_text}'"
                )

        if excessive_changes:
            error_msg = ";\n".join(excessive_changes)
            error_msg += (
                "\n\nYour optimizations changed the text too much. "
                "Keep high similarity (≥70% for normal text) by making MINIMAL changes."
            )
            return False, error_msg

        return True, ""

```


### 位置 1：优化提示词模板
**模板文件**：`vat/llm/prompts/optimize/subtitle.md`

#### 作用
在翻译前修正原语言字幕的错别字、语音识别错误、去除无意义口癖（um/uh/ah）、无意义拟声词（重复的啊啊啊->啊——）、统一术语。

#### 当前模板核心规则

```markdown
You are a professional subtitle correction expert. Your task is to fix errors in video subtitles while preserving the original meaning and structure.

<context>
Subtitles often contain recognition errors, filler words, and formatting inconsistencies that reduce readability. Your corrections should maintain the original expression while fixing technical errors and improving clarity.
</context>

<input_format>
You will receive:

1. A JSON object with numbered subtitle entries
2. Optional reference information containing:
   - Content context
   - Important terminology
   - Specific correction requirements
</input_format>

<instructions>
1. Fix errors while preserving original sentence structure (no paraphrasing or synonyms)
2. Remove filler words and non-verbal sounds: um, uh, ah, laughter markers, coughing sounds, etc.
3. Standardize formatting:
   - Correct punctuation
   - Proper English capitalization
   - Mathematical formulas in plain text (use ×, ÷, =, etc.)
   - Code syntax (variable names, function calls)
4. Maintain subtitle numbering (no merging or splitting entries)
5. Use reference information to correct terminology when provided
6. Keep original language (English stays English, Chinese stays Chinese)
7. Output only the corrected JSON, no explanations
</instructions>

<output_format>
Return a pure JSON object with corrected subtitles:

{
"0": "[corrected subtitle]",
"1": "[corrected subtitle]",
...
}

Do not include any commentary, explanations, or markdown formatting.
</output_format>

<examples>

<example>
<input_subtitles>
{
  "0": "the formula is ah x squared plus y squared equals uh z squared",
  "1": "this is called the pathagrian theorem *laughs*",
  "2": "it's um used in geometry and trigonomatry"
}
</input_subtitles>
<reference>
Content: Mathematics - Pythagorean theorem
Terms: Pythagorean theorem, geometry, trigonometry
</reference>
<output>
{
  "0": "The formula is x² + y² = z²",
  "1": "This is called the Pythagorean theorem",
  "2": "It's used in geometry and trigonometry"
}
</output>
</example>

<example>
<input_subtitles>
{
  "0": "大家好呃今天我们来学习机器学习",
  "1": "首先介绍一下神经网络的几本概念",
  "2": "它使用反向传播算法来训练模型嗯"
}
</input_subtitles>
<reference>
Content: 机器学习基础
Terms: 机器学习, 神经网络, 反向传播算法
</reference>
<output>
{
  "0": "大家好,今天我们来学习机器学习",
  "1": "首先介绍一下神经网络的基本概念",
  "2": "它使用反向传播算法来训练模型"
}
</output>
</example>
</examples>

<critical_notes>

- Preserve meaning and structure - only fix errors
- Use reference information to correct misrecognized terms
- Output pure JSON only, no explanations or markdown
- Maintain original language throughout
  </critical_notes>

```

那么，这一阶段，就是非常关键的环节了：前面我们的字幕是asr从直播视频中提取的，那么势必存在很多误报漏报。
比如我举几个我观察到的经典问题：
- 漏字：日语容易连读，可能缺失字符
- 错词：识别出的结果从字面上看是对的，但是其不符合上下文语境：比如主播是白上吹雪（shirakami fubuki），那么她自称自己的时候就有可能被识别成“白发”“白色”。导致“asr没错、translate也没错，但是结果质量就是很差”的问题。类似的，还有她对于粉丝群体的称呼sukonbu也容易被错误识别然后错误的翻译——并且注意，这里我们无论如何都是不可能穷尽所有的专有词汇的正确翻译形式的，所以我们应该举几个代表性的、不可能被llm思考联系到的例子（比如白上这个主播对于粉丝的称呼，只能我们先验的指定）。然后，通过提示词说明，要求他多思考，尽可能将相近的、读音可能有小幅度差别的原文字幕在上下文语境中灵活判断。（比如描述这个情景，然后要求结合我们的例子，灵活根据上下文判断，不要机械的pass）
- 错字：和上面的错词类似，但是不同于它的是，这里指的是对于非专有词汇/称呼的误报。比如两个在日语中读音相近的假名词汇（vrchat和河豚xxx），可能被错误的识别。尽管这个可以在下文翻译时纠正，但是我们也要尽可能在这里改变正确。还有一种情况，就是读音相同，但是写法不同（比如平、片假名和汉字写法，可能一个读音对应五六种文字表达，但是显然只有一两种的意思是对的）。所以这里也要要求llm灵活结合语境，判别出“看起来和上下文对不上”的词到底是不是误报？有没有必要修改成什么其他的词汇？但是这里要特别谨慎——比如可能上下文是在讲游戏做饭，然后提到了河豚——这个时候虽然可能看起来突兀，出现了一个没见过的词汇，但是未必不是正确的。类似的，在打游戏的时候提到vrchat也是很合理的。所以到底要不要修改？这个就不能只看一两句话是上下文来判断。而是应该结合这次输给大模型的整个上下文，来判断某个可疑的词汇到底是什么意思。此外，因为代码里面会检查修改率，防止大模型“抽风”，所以在修改的时候，最好保持原有的写法（比如原本是片假名的某个外来词，llm确定是误报，要修改，那么如果这个修改的结果确实某种表示（假名？汉字？）看起来和原本的字符很类似，那么还是最好对齐一下，不要彻底重写。对于语句结构的重写我们放在最后的translate环节，而不是这里）


### 位置 2：优化自定义提示词（配置）
**配置字段**：`config/default.yaml` → `translator.llm.optimize.custom_prompt`

```yaml
translator:
  llm:
    optimize:
      enable: true
      custom_prompt: ""  # 在这里填入术语表、主播特点、常见错误纠正规则
```

这里，就是要填入我们上面讨论的：specify 主播（翻译场景）的提示词了。

也就是说，上面应该指定“你要根据xx去联系上下文灵活判断”而这里就是“到底这个主播是什么，有什么约定的称呼“之类的场景特定提示词。

这里的代码中应用形式如下：

```python
        self.optimize_prompt = custom_optimize_prompt

…………
        if self.optimize_prompt:
            user_prompt += f"\nReference content:\n<reference>{self.optimize_prompt}</reference>"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]
```

即，填在最开始的user prompt末尾

我们这里以前面举的例子（白上吹雪）来完成这部分的提示词编写

这里请你多检索一下她的相关推文、文案、视频、介绍等。尽可能灵活、全面、富有代表性的编写这个提示词

（注意，我们交流的时候还是用中文。最后定稿再翻译成llm比较好接受的英文）

后面我们还要针对具体直播的题材编写特定的额外提示词。不过那个是下一阶段的任务

### 位置 3：场景特定的优化提示词（新增）
**配置文件**：`vat/llm/scenes.yaml` → `scenes[i].prompts.optimize`

类似于断句，每个场景也可配置独立的优化提示词。执行流程与断句相同，场景优化提示词会与全局提示词合并后使用。

#### 示例：闲聊直播的优化规则
```yaml
scenes:
  - id: "chatting"
    name: "闲聊直播"
    prompts:
      optimize: |
        ## 闲聊直播优化规则
        - 保留所有语气词和口癖（"哈"、"嗯"、"呢"、"啦"）
        - 对观众的称呼要多样化（"大家"、"各位"、"朋友们"）
        - 保留重复词强调情绪（"好好好"、"对对对"）
        - 删除明显的 filler："啊"（犹豫词）但保留"啊"（惊讶）
```

---

## 环节 3：翻译（Translate）阶段

因为这个翻译是整个流程最后的环节了（输出将会直接拿去嵌入视频），所以这个地方的优化可以稍微激进一点——比如，中日语言的结构语法显然不同，所以必然有语句的重构（这个几乎都不用强调了，是默认的），然后在翻译的时候，一方面要激进的、完全的纠正错别字（即，我们在上文optimzed阶段就尝试过了纠正“asr没错，单词没错，但是不符合语境”的情况，并且在你给出了几版优化之后依然存在问题（比如白上吹雪自称被识别为白发等）。那么，在这个最后的翻译阶段，我们就需要要求llm强遵循我们的custom prompt，把所有看起来这种可疑的、不自然的、不符合语境的内容都做替换和优化——这个地方甚至可以为了注重上下文的连贯自然，以及我下面强调的翻译风格，而稍微的有一些rewrite。

custom prompt里面，注意翻译的风格：我们现在还是以主播白上吹雪为例，应该考虑到她的风格是偏向活泼可爱的
比如——“えー”这种翻译就不应该是“额”或者“呃”这种，而是应该翻译成“诶——”或者“呜!”"呜~"这种——总而言之，应该看起来网络风、口语化、活泼可爱，**符合经典的vtuber“清楚系”的形象**。而且，**注意不要和简中互联网native的混淆**——比如简中的女性可能经常说什么“宝子们xxx”这种，但是这种表达在日语主播中是不存在的，并且也显得很尴尬，很不自然。所以“要翻译成native的中文”，但是这更多的是指语法、结构等。而具体的一些名词、口语、俗语、卖萌之类的表达依然应该保持二次元惯用/日语的风格，不要和简中互联网上故作geek、女权、耽美之类的恶俗风格混淆。如果一定要归类一个风格，应该说是“近年但偏早一点的二次元、日本的少女系、hololive、Vtuber主播”的感觉

此外，一些明显是拟声词、感叹词、无意义的口语、口癖等，应该翻译成可爱的，少女的，略带一些卖萌感觉的表达，并且对于大量的重复词（ええええ、はははは、ほいほい、ででで之类的），应该恰当的缩减合并，灵活考虑变成口语上类似，但是字幕中不会看起来很奇怪的表达。这个阶段的重写幅度可以是比较大的。

在全局的翻译提示词中，可以不用太强调“日系、可爱”之类的特点，但是也依然要避免我上述说的，简中互联网上一些比较恶俗的风格。


那么你现在的任务和之前类似，依然是首先，把翻译阶段的整体提示词模板做调整（先与我沟通要怎么改）。
然后，给出针对白上吹雪这个custom 情景的提示词。（可以注意到，sence提示词我们对于optimized和translate两个阶段都还没有写——这个我们之后再处理）



### 位置 1：翻译提示词模板

#### 模板 A：标准翻译
**模板文件**：`vat/llm/prompts/translate/reflect.md`

```md
You are a professional subtitle translator specializing in ${target_language}. Your goal is to produce translations that sound natural and native, not machine-translated.

<context>
Machine translation often produces technically correct but unnatural text—it translates words rather than meaning, ignores context, and misses cultural nuances. Your task is to bridge this gap through reflective translation: identify machine-translation patterns in your initial attempt, then rewrite to match how native speakers actually communicate.
</context>

<terminology_and_requirements>
${custom_prompt}
</terminology_and_requirements>

<instructions>
**Stage 1: Initial Translation**
Translate the content, maintaining all information and subtitle numbering.

**Stage 2: Machine Translation Detection & Deep Analysis**
Critically examine your translation and identify:

1. **Structural rigidity**: Does it mirror source language word order unnaturally?
2. **Literal word choices**: Are there more natural/colloquial alternatives?
3. **Missing context**: What implicit meaning or tone needs to be made explicit (or vice versa)?
4. **Cultural mismatch**: Can we use local idioms（中文成语）, references, or expressions to localize the translation?
5. **Register issues**: Is the formality level appropriate for the context?
6. **Native speaker test**: Would a native speaker say it this way? If not, how WOULD they say it?

For each issue found, propose specific alternatives with reasoning.

**Stage 3: Native-Quality Rewrite**
Based on your analysis, rewrite the translation to sound completely natural in ${target_language}. Ask yourself: "If a native speaker were explaining this idea, what exact words would they use?"
</instructions>

<output_format>
{
"1": {
"initial_translation": "<<< First translation >>>",
"reflection": "<<< Identify machine-translation patterns: What sounds unnatural? Why? What would a native speaker say instead? Consider structure, word choice, context, culture, register. Be specific about problems and alternatives. >>>",
"native_translation": "<<< Natural, native-quality translation that eliminates all machine-translation artifacts >>>"
},
...
}
</output_format>

<examples>
<example>
<scenario>Technical video about software development</scenario>
<input>
{
  "1": "今天我们来看一下这个bug是怎么产生的",
  "2": "主要是因为我们在处理用户输入的时候没有做验证"
}
</input>
<output>
{
  "1": {
    "initial_translation": "Today we will look at how this bug was generated.",
    "reflection": "Problems identified: 1) 'Today we will look at' is overly formal and stiff—machine translation of '今天我们来看一下'. Native speakers would use more casual transitions. 2) 'was generated' is passive and unnatural for bugs—they don't 'generate', they 'happen' or 'occur'. 3) Missing the casual, conversational tone of the original. Native alternative: 'Let's see what caused this bug' or 'So, what's behind this bug?' - much more natural for video narration.",
    "native_translation": "Let's see what's causing this bug."
  },
  "2": {
    "initial_translation": "The main reason is that we did not do validation when processing user input.",
    "reflection": "Problems: 1) 'The main reason is that' is bureaucratic—direct translation of '主要是因为'. Native speakers rarely start explanations this formally in video context. 2) 'did not do validation' is awkward—'do validation' is not idiomatic. 3) 'when processing' is correct but mechanical. Native alternatives: 'It's mainly because...' or 'Basically, we...' for the opening. 'We didn't validate' or 'We're not validating' for the action. Better: 'We're not validating user input' - concise, natural, appropriate register.",
    "native_translation": "Basically, we're not validating user input."
  }
}
</output>
</example>

<example>
<scenario>Casual vlog with young audience</scenario>
<input>
{
  "1": "这个地方真的太美了我都不想走了",
  "2": "你们一定要来打卡啊"
}
</input>
<output>
{
  "1": {
    "initial_translation": "This place is really too beautiful, I don't want to leave.",
    "reflection": "Problems: 1) '太...了' translated as 'too...+adjective' misses the Chinese emphatic pattern—it doesn't mean 'excessively' but 'extremely/so'. 2) Structure is choppy and unnatural—comma splice feels translated. 3) Missing the enthusiastic, spontaneous tone. 4) 'I don't want to leave' is flat compared to the original's emotion. Native speaker would use: 'This place is SO gorgeous' or 'absolutely stunning' for emphasis. For the second part: 'I could stay here forever' or 'I never want to leave' captures the emotion better. Combine naturally: 'This place is absolutely stunning—I never want to leave!'",
    "native_translation": "This place is absolutely stunning—I could stay here forever!"
  },
  "2": {
    "initial_translation": "You all must come to check in.",
    "reflection": "Major problems: 1) '打卡' (daka/check-in) is a Chinese internet culture term meaning 'visit a trendy place'. Translating to 'check in' sounds like hotel check-in, completely wrong meaning. 2) 'You all must come' is stiff and imperative. 3) Missing the friendly, inviting tone. Native alternatives for '打卡': 'visit', 'check out this spot', 'come see this place'. For tone: 'You've gotta...' or 'You should definitely...' is more natural than 'must'. Best option: 'You've gotta check this place out!' or 'You need to visit!'—captures enthusiasm and invitation.",
    "native_translation": "You've gotta check this place out!"
  }
}
</output>
</example>
</examples>

<key_principles>
**Eliminate machine translation:**

- Avoid word-for-word translation and source language structure
- Don't translate idioms literally

**Sound native:**

- Use natural expressions for the context and audience
- Match appropriate formality level
- For Chinese: Use 成语/俗语/网络用语 when naturally fitting

Goal: Natural speech, not machine translation text.
</key_principles>

```

在代码中的调用方式：

```python
asr_data = subtitle_data
# 将ASRData转换为SubtitleProcessData列表
translate_data_list = [
    SubtitleProcessData(index=i, original_text=seg.text)
    for i, seg in enumerate(asr_data.segments, 1)
]
# 分批处理字幕
chunks = self._split_chunks(translate_data_list)
# 多线程翻译
translated_list = self._parallel_translate(chunks)
# 设置字幕段的翻译文本
new_segments = self._set_segments_translated_text(
    asr_data.segments, translated_list
)
result = ASRData(new_segments)
            
# 保存翻译结果
translated_srt = self.output_dir / "translated.srt"
result.save(str(translated_srt))
logger.info(f"翻译结果已保存: {translated_srt}")
            
return result

…………

def _split_chunks(
    self, translate_data_list: List[SubtitleProcessData]
) -> List[List[SubtitleProcessData]]:
    """将字幕分割成块"""
    return [
        translate_data_list[i : i + self.batch_num]
        for i in range(0, len(translate_data_list), self.batch_num)
    ]
def _parallel_translate(
    self, chunks: List[List[SubtitleProcessData]]
) -> List[SubtitleProcessData]:
    """并行翻译所有块"""
    futures = []
    translated_list = []
    failed_chunks = []
    for chunk in chunks:
        future = self.executor.submit(self._safe_translate_chunk, chunk)
        futures.append((future, chunk))
    for future, chunk in futures:
        if not self.is_running:
            break
        try:
            result = future.result()
            translated_list.extend(result)
        except Exception as e:
            logger.error(f"翻译块失败：{str(e)}")
            failed_chunks.append((chunk, str(e)))
            # 失败时保留原文，但标记为未翻译
            for data in chunk:
                data.translated_text = data.original_text  # 标记为未翻译
            translated_list.extend(chunk)
    # 如果所有块都失败了，抛出异常
    if failed_chunks and len(failed_chunks) == len(chunks):
        error_messages = [f"块 {i+1}: {err}" for i, (_, err) in enumerate(failed_chunks)]
        raise RuntimeError(f"所有翻译块都失败了:\n" + "\n".join(error_messages))
    
    # 如果部分块失败，记录警告但继续
    if failed_chunks:
        logger.warning(f"{len(failed_chunks)}/{len(chunks)} 个翻译块失败，已保留原文")
    return translated_list

```

```python 
def _translate_chunk(
    self, subtitle_chunk: List[SubtitleProcessData]
) -> List[SubtitleProcessData]:
    """翻译字幕块"""
    logger.debug(
        f"正在翻译字幕：{subtitle_chunk[0].index} - {subtitle_chunk[-1].index}"
    )
    subtitle_dict = {str(data.index): data.original_text for data in subtitle_chunk}
    # 获取提示词
    if self.is_reflect:
        prompt = get_prompt(
            "translate/reflect",
            target_language=self.target_language,
            custom_prompt=self.custom_prompt,
        )
    else:
        prompt = get_prompt(
            "translate/standard",
            target_language=self.target_language,
            custom_prompt=self.custom_prompt,
        )
    try:
        # 构建带上下文的输入（新增）
        user_input = self._build_input_with_context(subtitle_dict)
        
        result_dict = self._agent_loop(prompt, user_input, expected_keys=set(subtitle_dict.keys()))
        # 处理反思翻译模式的结果
        if self.is_reflect and isinstance(result_dict, dict):
            processed_result = {
                k: f"{v.get('native_translation', v) if isinstance(v, dict) else v}"
                for k, v in result_dict.items()
            }
        else:
            processed_result = {k: f"{v}" for k, v in result_dict.items()}
        # 保存当前 batch 结果供下次使用（新增）
        self._previous_batch_result = processed_result.copy()
        for data in subtitle_chunk:
            data.translated_text = processed_result.get(
                str(data.index), data.original_text
            )
        return subtitle_chunk
    except openai.RateLimitError as e:
        logger.error(f"OpenAI Rate Limit Error: {str(e)}")
        # Rate limit 错误可以重试，但这里应该抛出异常让上层处理
        raise
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI Authentication Error: {str(e)}")
        # 认证错误应该立即失败，不应该降级处理
        raise RuntimeError(f"API 认证失败: {str(e)}") from e
    except openai.NotFoundError as e:
        logger.error(f"OpenAI NotFound Error: {str(e)}")
        # 模型不存在错误应该立即失败
        raise RuntimeError(f"模型不存在: {str(e)}") from e
    except Exception as e:
        import traceback
        logger.error(f"翻译块失败: {str(e)}, 尝试降级处理,traceback: {traceback.format_exc()}")
        # 其他错误尝试降级处理
        try:
            return self._translate_chunk_single(subtitle_chunk)
        except Exception as fallback_error:
            logger.error(f"降级翻译也失败: {str(fallback_error)}")
            # 如果降级也失败，抛出异常
            raise RuntimeError(f"翻译失败且降级处理也失败: {str(e)}") from e
def _agent_loop(
    self, 
    system_prompt: str, 
    user_input: str,
    expected_keys: Optional[set] = None
) -> Dict[str, str]:
    """Agent loop翻译/优化字幕块"""
    assert system_prompt, "调用契约错误: system_prompt 不能为空"
    assert user_input, "调用契约错误: user_input 不能为空"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    last_response_dict = None
    
    for _ in range(self.MAX_STEPS):
        response = call_llm(messages=messages, model=self.model)
        if not response or not response.choices:
            raise RuntimeError("LLM 未返回有效响应")
        
        content = response.choices[0].message.content.strip()
        if not content:
            raise RuntimeError("LLM 返回内容为空")
        
        response_dict = json_repair.loads(content)
        last_response_dict = response_dict
        
        # 使用 expected_keys 验证（如果提供）
        validation_keys = expected_keys if expected_keys else set(response_dict.keys())
        is_valid, error_message = self._validate_llm_response(
            response_dict, validation_keys
        )
        if is_valid:
            return response_dict
        else:
            messages.append({
                "role": "assistant",
                "content": json.dumps(response_dict, ensure_ascii=False),
            })
            messages.append({
                "role": "user",
                "content": f"Error: {error_message}\n\n"
                          f"Fix the errors above and output ONLY a valid JSON dictionary with ALL {len(validation_keys)} keys",
            })
    return last_response_dict
def _validate_llm_response(
    self, response_dict: Any, expected_keys: set
) -> Tuple[bool, str]:
    """验证LLM翻译结果（支持普通和反思模式）"""
    if not isinstance(response_dict, dict):
        return (
            False,
            f"Output must be a dict, got {type(response_dict).__name__}. Use format: {{'0': 'text', '1': 'text'}}",
        )
    actual_keys = set(response_dict.keys())
    def sort_keys(keys):
        return sorted(keys, key=lambda x: int(x) if x.isdigit() else x)
    if expected_keys != actual_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        error_parts = []
        if missing:
            error_parts.append(
                f"Missing keys {sort_keys(missing)} - you must translate these items"
            )
        if extra:
            error_parts.append(
                f"Extra keys {sort_keys(extra)} - these keys are not in input, remove them"
            )
        return (False, "; ".join(error_parts))
    # 如果是反思模式，检查嵌套结构
    if self.is_reflect:
        for key, value in response_dict.items():
            if not isinstance(value, dict):
                return (
                    False,
                    f"Key '{key}': value must be a dict with 'native_translation' field. Got {type(value).__name__}.",
                )
            if "native_translation" not in value:
                available_keys = list(value.keys())
                return (
                    False,
                    f"Key '{key}': missing 'native_translation' field. Found keys: {available_keys}. Must include 
'native_translation'.",
                )
    return True, ""

def _build_input_with_context(self, subtitle_dict: Dict[str, str]) -> str:
        """
        构建带上下文的输入
        
        Args:
            subtitle_dict: 当前 batch 的字幕字典
            
        Returns:
            格式化的输入字符串
        """
        if not self.enable_context or self._previous_batch_result is None:
            # 第一个 batch 或未启用上下文
            return json.dumps(subtitle_dict, ensure_ascii=False)
        
        # 构建上下文部分
        context_lines = []
        for key, text in self._previous_batch_result.items():
            context_lines.append(f"[{key}]: {text}")
        
        context_text = "\n".join(context_lines)
        
        # 组合格式
        input_text = f"""Previous context (for reference only, maintain consistency with these translations, but DO NOT TRANSLATE THE PREVIOUS CONTEXT ITSELF):
{context_text}

Translate the following (output ONLY these keys):
{json.dumps(subtitle_dict, ensure_ascii=False)}"""
        
        return input_text
```

那么这个就是reflect的翻译提示词模板，以及他在代码中是如何被使用的了。
请你考虑如何优化，参考我上面讲的内容

### 位置 2：翻译自定义提示词（配置）
**配置字段**：`config/default.yaml` → `translator.llm.custom_prompt`

```yaml
translator:
  llm:
    model: "gpt-5-nano"                    # 翻译模型
    enable_reflect: true                    # 反思翻译（启用后质量更高，但消耗更多 token）
    batch_size: 20                          # 每批处理的字幕数量
    thread_num: 5                           # 并发线程数
    custom_prompt: "fubuki"                       # 自定义提示词文件名（相对于 vat/llm/prompts/custom/），如 "translate/example.md"，空字符串表示不使用
    enable_context: true                    # 是否启用前文上下文（新增）
```

此处的custom提示词的要求即上述所说，主要是针对白上吹雪这个情景特化。



### 位置 3：翻译场景特定提示词（新增）
**配置文件**：`vat/llm/scenes.yaml` → `scenes[i].prompts.translate`

每个场景的翻译提示词会在翻译时加载并与全局 `custom_prompt` 合并。场景提示词优先级更高，会先被添加到提示词开头。

#### 示例：游戏直播的翻译规则
```yaml
scenes:
  - id: "gaming"
    name: "游戏直播"
    prompts:
      translate: |
        ## 游戏直播翻译要点
        - Boss/Skill/HP 等术语保持英文不翻译，或用游戏社区通用术语
        - "草"可译为"笑死"、"hhh"或"哈哈"，视语境
        - "寄了"是游戏社区用语，表示完蛋了，译为"It's over"/"We're done"或保持原文
        - 战斗场景用词简短有力："冲"、"撤"、"躲"、"上"
        - 失败/成功时的情绪强烈，不要翻译得平淡
```

### 位置 4：翻译上下文（新增）
**配置字段**：`config/default.yaml` → `translator.llm.enable_context`

当启用时，翻译每个字幕批次（batch）时，LLM 会看到前一个 batch 的翻译结果作为参考上下文：

```
Previous context (for reference only, maintain consistency with these translations, but DO NOT TRANSLATE THE PREVIOUS CONTEXT ITSELF):
[前一 batch 的翻译结果]

[当前 batch 的字幕]
```

这样可以确保：
- 术语一致性（同一个游戏术语始终用同一个翻译）
- 人物称呼一致（主播、观众昵称统一）
- 语气连贯性（不会出现前后语气差异太大）

**默认启用** (`enable_context: true`)，可通过配置关闭。

---

## 📝 优化流程总结

### 零代码优化流程（推荐）

1. **第一步：Whisper initial_prompt**
   - 根据视频类型（游戏/闲聊/ASMR）填写场景描述和关键术语
   - 用英文，50-200 字符

2. **第二步：调整断句长度**
   - 游戏直播：16-20 字
   - 闲聊直播：24-28 字
   - ASMR：12-18 字

3. **第三步：配置字幕优化**（可选）
   - 开启 `translator.llm.optimize.enable: true`
   - 填写 `optimize.custom_prompt`：术语表、主播口癖、常见错误

4. **第四步：配置翻译提示词**（可选）
   - 填写 `translator.llm.custom_prompt`：游戏术语、翻译风格、主播特点
   - 开启 `enable_reflect: true` 提高质量
   - 保持 `enable_context: true` 以获得术语一致性

5. **第五步：调整断句分块参数**（针对长视频）
   - `chunk_size_sentences`: 默认 50，可调至 30-100
   - `chunk_overlap_sentences`: 默认 5，可调至 3-10
   - `chunk_min_threshold`: 默认 30，短视频可提高至 50+

6. **第六步：查看场景配置**（自动应用）
   - 查看 `vat/llm/scenes.yaml` 中的 5 个场景预设
   - 视频会自动识别场景并加载对应的 split/translate/optimize 提示词

### 代码级优化（若需进阶）

7. **修改断句提示词模板**
   - 编辑 `vat/llm/prompts/split/sentence.md`
   - 增加 vtuber 直播规则（口癖、游戏术语、重复感叹）

8. **修改翻译提示词模板**
   - 编辑 `vat/llm/prompts/translate/standard.md` 或 `reflect.md`
   - 增加 vtuber 直播示例和检查项

9. **定制场景配置**
   - 编辑 `vat/llm/scenes.yaml`，调整或新增场景的 split/translate/optimize 提示词

---

## ⚠️ 注意事项

1. **模板文件修改后立即生效**（因为有 LRU 缓存，但每次运行会重新读取）
2. **配置文件修改后需重启进程**
3. **提示词长度控制**：
   - `custom_prompt` 建议 500-2000 字符
   - 过长会增加 token 成本且可能降低响应质量
4. **测试验证**：
   - 修改后先用短视频（5-10分钟）测试效果
   - 对比 `original_raw.srt` vs `original_split.srt`（断句效果）
   - 对比 `original.srt` vs `translated.srt`（翻译质量）

---

**本文档涵盖了所有零代码修改的提示词优化位置。所有功能已通过代码实现，包括场景自动识别、分块断句处理、翻译上下文传递等。**
