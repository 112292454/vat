Reference (optimize stage). Aim for high precision and minimal edits.
Do NOT translate. Do NOT rewrite sentence structure. Prefer the smallest possible change.

[GLOBAL BASELINE: Japanese live transcript]

This is spontaneous live speech. Keep colloquial tone; do NOT formalize or rewrite into written style.

ASR common issues (use only for decision making, not for inventing text): homophones/near-homophones, mixed scripts (漢字/ひらがな/カタカナ), katakana loanwords (ー/ッ/ャュョ) confusion.

Confidence rule:
Only correct when (a) the corrected form is in this reference OR is consistently used across this chunk,
AND (b) the current token clearly conflicts with chunk context. If uncertain, keep the original unchanged.

Light polishing allowed but must stay minimal:
remove clear fillers, compress excessive laughter/repetition, tiny kana/particle fixes when obvious,
minimal punctuation (、。) only if it improves readability without changing structure.
Never paraphrase, never reorder clauses, never add new information.

Output safety:
Return JSON ONLY with exactly the same keys. No extra text.

[SPEAKER / canonical forms]

白上フブキ (Shirakami Fubuki)

Common short: フブキ / FBK

[FAN NAMES (high value)]

Canonical fan name: すこん部（すこんぶ） / Friends
Common ASR confusions when addressing viewers/fans:
塩昆布 / 酢昆布 / 昆布 / すこんぶ / スコンブ / スコンボ / すここんぶ
Rule:

If clearly addressing viewers/fans (みんな/ありがとう/今日も/集まって/助かる etc.), prefer “すこん部”.

If the chunk is literally about food/konbu cooking, keep the original token.

[NAME CONFUSION GUARDRAILS]
Possible ASR confusions for 白上フブキ:

白髪 / 白神 / 吹雪 / ふぶき (ambiguous)
Rule:

High-priority correction (when confidence is high): 白髪 -> 白上

If “白髪” appears in a context clearly referring to the person/character (e.g., near any of: フブキ / FBK / すこん部 / フブキングダム / 国民 / 配信 / チャンネル / 衣装 / 初期衣装 / グッズ / コラボ / hololive / ホロライブ),
then treat it as ASR misrecognition and correct to “白上”.

Also correct patterns like “白髪の◯◯(衣装/配信/フブキングダム/すこん部/etc.)” -> “白上の◯◯”.

Exception: If surrounding context is literally about hair color/white hair (e.g., “白髪が増えた”, “白髪染め”, hair care, aging), keep “白髪”.

“フブキングダム” normalization

Normalize likely confusions back to “フブキングダム” when referring to the community/setting:

フグキングダム / ブッキングダム / フブキンダム / フブキングラム

If “フブキングダム” is used as a proper noun, keep it consistent (do not alternate spellings).

Name adjacency heuristics (to decide the above):

If a line contains any of: 白上 / フブキ / FBK / ふーちゃん / フブちゃん / すこん部 / フブキングダム,
assume the topic is this VTuber unless strong evidence otherwise.

In such lines, prefer correcting ambiguous homophones toward the canonical forms above.

[RELATED PROPER NOUNS]

黒上フブキ (keep if mentioned; do not normalize away)

インテリジェンスユニコーン副部長 (treat as fixed proper noun if it appears)

[HOLOLIVE GAMERS member names (for co-stream mentions)]

白上フブキ / 大神ミオ / 猫又おかゆ / 戌神ころね / etc.
Rule: Only correct to these names when chunk context indicates member-name mentions.

[HIGH-PRECISION CORRECTIONS observed in this content]

Prefer minimal substitutions (1–2 characters) rather than rewriting the sentence.

Keep script style close to original when possible.

Term: フブキングダム (canonical)
Common ASR confusions: フグキングラム / ブッキングダム / ブキングダム
Rule:

If context refers to the community/kingdom/citizens/channel-related wording, correct to “フブキングダム”.

Use minimal substitution (e.g., フグ→フブ) rather than rewriting surrounding text.