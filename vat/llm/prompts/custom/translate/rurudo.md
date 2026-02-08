[Target speaker: rurudo / るるど (+ original character るるどらいおん)]
Goal: Simplified Chinese subtitles that feel natural and subtitle-friendly.
Allow rewrite for fluency and ASR repair, but do NOT invent new facts.
Avoid cringe Mainland internet influencer slang (宝子们/家人们/绝绝子/YYDS etc.).
Keep a light "JP/VTuber-friendly" tone when appropriate, but don't over-act cute in serious/explanatory lines.

=== Canonical Names & Terms (must be consistent) ===
- るるど: prefer "rurudo" in Chinese subtitles (romaji for readability). Keep consistent once chosen.
- るるどらいおん: prefer "rurudoLion" (romaji handle-style) OR "rurudo小狮子" (if the line is clearly persona-talk).
  Pick ONE style and keep consistent within the video.
- Mode phrases (translate, not romaji):
  - おっきいモード -> "大只模式"/"大号模式" (pick one)
  - ちっちゃいモード -> "小只模式"/"小号模式" (pick one)

=== Translate-first Policy (to prevent over-romanization) ===
- Default: translate meaning into natural Chinese.
- Use romaji ONLY when it is very likely a proper noun / call name / fan label.
- NEVER romanize normal content words with clear meaning (verbs/adjectives/particles, or common katakana phrases).

Hard exclusions (MUST translate, DO NOT romaji):
- ファイト -> "加油"
- ナイス -> "漂亮"/"不错"
- オッケー/OK -> "OK"/"行"
- ヤバい -> "离谱"/"糟了"/"太强了"(by context)
- ガチ -> "真"/"认真的"
- かわいい -> "可爱"
- うまい -> "好厉害"/"打得好"/"真香"(by context)
- えー/ええと/あの -> "嗯——/那个…"(soft), not "呃/额"
- Short SFX / interjections (single-word katakana/hiragana): translate to Chinese SFX/interjection, NOT romaji, NOT kana.

=== High-Confidence Correction Rules (aggressive allowed) ===
1) るるど / るるどらいおん variants:
   - If a token looks like a name/handle but is garbled (e.g., るるどライオン / ルルドらいおん),
     normalize to the chosen canonical form (rurudo / rurudoLion etc.).
2) Mode words:
   - If the line is clearly about switching size/mode, map to the mode translations above.

=== Honorific Handling (MUST) ===
- Do NOT output hyphenated honorifics like "-san/-chan/-senpai" in the final Chinese line.
- Translate honorifics into Chinese:
  - さん -> omit (or rarely "桑" if the vibe needs it)
  - ちゃん -> "酱" or omit
  - 先輩 -> "前辈"
  - 先生 -> "老师"
- If the name is unknown: keep NAME in romaji + translated honorific (no hyphen).
  Example: "X先輩" -> "X前辈"

=== Fallback for unknown NAMES / fan labels (STRICT) ===
Use romaji (Hepburn) ONLY when it is very likely a proper noun / call name / fan label.

Allowed triggers (any of the following):
- Followed by honorifics: さん / ちゃん / くん / 先輩 / 先生
- Clear vocative usage (addressing someone): "X、..." / "ねえX" / "X来てる？"
- Handle-like call name in collab context (calling teammates)

If not triggered, translate it as a normal word/phrase.
If a term is uncertain but seems like a practical meaning (food/tool/action), translate by meaning rather than romaji.

=== Output Note ===
- Subtitles are already segmented. Do not merge/split keys.
- Rewrite freely for natural Chinese and coherence, but do not invent extra content.
- Do NOT leave Japanese kana/kanji in the final Chinese line; if you must keep a proper noun, use romaji.
