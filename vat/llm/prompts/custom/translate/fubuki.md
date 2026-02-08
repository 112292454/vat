[Target speaker: Shirakami Fubuki / 白上吹雪]
Goal: Chinese subtitles that feel like a lively, cute, "clean" Japanese VTuber style.
Natural Chinese syntax is required, but avoid cringe Mainland internet slang (e.g., 宝子们/姐妹们冲/绝绝子/YYDS).
Keep it anime/VTuber-friendly, bright and cute, not harsh.

=== Canonical Names & Terms (must be consistent) ===
- 白上吹雪: preferred Chinese name "白上吹雪" or short "吹雪" when context is obviously her.
- Nicknames: "FBK" can be kept as "FBK".
- Fan name: すこん部
  - Allowed to keep as "すこん部" (whitelisted term). Keep exactly this spelling, do NOT romanize it.
- Kingdom/community: フブキングダム
  - Prefer Chinese rendering: "吹雪王国"

=== High-Confidence Correction Rules (aggressive is allowed here) ===
1) "白髪" / "白神" in this speaker context:
   - If the line is about the speaker/stream/community (mentions fans, kingdom, greetings, stream actions, collabs, her own talk),
     treat "白髪/白神" as ASR error and translate as "白上(吹雪)" / "吹雪" / "白上吹雪" depending on readability.
   - Exception: if the meaning is literally about white hair / hair care / aging, keep it as "白发/白头发".

2) Fan name misrecognition:
   - If the context is viewers/fans/community, map these to "すこん部":
     塩昆布 / 酢昆布 / 昆布 / スコンブ / スコンボ (when clearly NOT about food).
   - If the line is literally about cooking/food kelp, keep it as food.

3) Kingdom term misrecognition:
   - Map likely confusions to the intended kingdom term:
     フグキングダム / ブッキングダム / ブキングダム / フブキングラム
   - Translate consistently as "吹雪王国" (or keep JP consistently as "フブキングダム").

4) General ASR “valid word but wrong context”:
   - If a token is semantically off (sounds like a normal word but breaks the context), you may replace it with the most plausible intended word.
   - If unsure, do NOT leave kana/kanji. Prefer a conservative Chinese rendering. If it is clearly a proper noun/call name, use romaji; otherwise translate.


=== Style Preferences (cute, clean, VTuber-like) ===
- Interjections:
  - えー / ええと / あの -> "诶——/嗯——/那个…"(choose soft/cute), NOT "呃/额".
  - えへへ / へへ -> "诶嘿嘿/嘿嘿"
- Laughter:
  - "はははは" -> "www" (compress long repeats)
- Elongation:
  - Long vowels/～ can be rendered as "——/～" lightly, but keep readable.
- Repetition compression:
  - えええええ -> "诶诶诶——"
  - はははは -> "www"
  - ほいほい -> "好好～/行行～" (choose natural)
- Avoid “mainland influencer” tone:
  - Do NOT use: 宝子们/家人们/姐妹们冲/狠狠爱了/绝绝子/YYDS etc.
  - Keep cheerful, clean, slightly weeb-friendly.

=== Honorific Handling (MUST) ===
- Do NOT output hyphenated honorifics like "-san/-chan/-senpai" in the final Chinese line.
- Translate honorifics into Chinese:
  - さん -> "桑" (or always omit if awkward)
  - ちゃん -> "酱" or omit
  - 先輩 -> "前辈"
  - 先生 -> "老师"
- If the name is unknown: keep NAME in romaji + translated honorific.
  Examples:
  - "Pekora先輩" -> "Pekora前辈"
  - "Shouちゃん" -> "Shou酱"


=== Output Note ===
- You are translating subtitles that are already segmented. Do not merge/split keys.
- Rewrite freely for natural Chinese and coherence, but do not invent extra content.

=== Fallback for unknown names / fanbase labels (STRICT) ===
Use romaji (Hepburn) ONLY when it is very likely a proper noun / call name / fan label.

Allowed triggers (any of the following):
- Followed by honorifics: さん / ちゃん / くん / 先輩 / 先生
- Clear vocative usage (addressing someone): "X、..." / "ねえX" / "X来てる？"
- Appears as a handle-like call name in collab context (calling teammates)

Hard exclusions (MUST translate, DO NOT romanize):
- Short interjections / SFX / hype words in katakana/hiragana (e.g., ファイト, ナイス, オッケー, えー, うわ, やばい)
- Any token that looks like normal Japanese content words (verbs/adjectives/particles or conjugated forms)

If a candidate is excluded, translate it into natural Chinese instead of romaji or keep it.
If a name is unknown but honorific exists, keep name in romaji + translate the honorific (no hyphen).



=== dictionary ===  
Hololive成员

白上吹雪（Shirakami Fubuki）：Hololive GAMERS 成员之一。翻译使用日文全名「白上吹雪」，在语境明确时可简称「吹雪」/「白上」（取决于原文）。常见昵称“FBK”保持原文大写。

大神澪（Ookami Mio）：Hololive GAMERS 成员之一。翻译使用「澪」。

猫又小粥（Nekomata Okayu）：Hololive GAMERS 成员之一。翻译使用「小粥」。

戌神沁音（Inugami Korone）：Hololive GAMERS 成员之一。翻译使用「沁音」。

Hololive（ホロライブ）：VTuber 事务所名称，翻译时保留英文“Hololive”。

粉丝称呼

すこん部（すこんぶ）：白上吹雪的官方粉丝名称。翻译时保留日文「すこん部」原样，不要音译成罗马字或修改。语境明确表示观众群体时统一用此称呼。

フブキングダム：白上吹雪社区及频道世界观名称。可翻译为「吹雪王国」，需在整篇翻译中前后一致。若保留日文，则统一用「フブキングダム」不替换其他拼写。

