 Reference (optimize stage). Aim for high precision and minimal edits.
Do NOT translate. Do NOT rewrite sentence structure. Prefer the smallest possible change.

[GLOBAL BASELINE: Japanese live transcript]
- This is spontaneous live speech. Keep colloquial tone; do NOT formalize or rewrite into written style.
- Confidence rule:
  Only correct when (a) the corrected form is in this reference OR is consistently used across this chunk,
  AND (b) the current token clearly conflicts with chunk context. If uncertain, keep the original unchanged.
- Light polishing allowed but must stay minimal:
  remove clear fillers, compress excessive laughter/repetition, tiny kana/particle fixes when obvious,
  minimal punctuation (、。) only if it improves readability without changing structure.
  Never paraphrase, never reorder clauses, never add new information.
- Output safety:
  Return JSON ONLY with exactly the same keys. No extra text.

[SPEAKER / canonical forms]
- るるど (rurudo)  ※creator name; keep script as-is if already correct.
- Common handles may appear in latin: rurudo / rurudo_ (if present, keep; do not "JP-ify" it)

[ORIGINAL CHARACTER (high value)]
- Canonical character name: るるどらいおん
  Notes:
  - This is a fixed proper noun (original character).
  - Related fixed phrases that may appear (keep as-is; do not "correct away"):
    - おっきいモード
    - ちっちゃいモード

[NAME NORMALIZATION (high precision)]
- If the chunk is clearly about the character (any of: るるどらいおん / らいおん / ライオン + self-reference / mode words / merch-like wording),
  then normalize likely ASR variants to "るるどらいおん" using minimal substitution only.
  Common safe normalizations (ONLY when confidence is high):
  - るるどライオン / るるど らいおん / ルルドらいおん / るるどらいおんー -> るるどらいおん
- Guardrail:
  Do NOT force "ライオン/らいおん" -> "るるどらいおん" when it clearly means a generic lion (animals/zoo/metaphor) and not the character.

[MODE WORDS (do not drift)]
- Keep these exact spellings when they appear:
  - おっきいモード
  - ちっちゃいモード
- If ASR outputs near-miss variants (e.g., おっきモード / ちっちゃモード),
  correct ONLY when the intended phrase is obvious and use minimal edits.

[COSTUME / PRODUCT NAMES (fixed proper nouns, do NOT alter)]
- パステルお肉: costume name, keep as-is. Do NOT correct to food/meat related words.

[OTHER PROPER NOUNS]
- For any proper nouns (person names, game/anime titles, brand names, etc.) not listed above:
  keep them as-is in Japanese if ASR output looks correct; only correct obvious ASR typos.
- Do NOT change a generic word into a person name unless there is strong adjacent evidence.

[GENERAL MINIMAL-EDIT STYLE]
- Prefer 1–2 character fixes over rewriting.
- Do not add kana readings or expansions.
- If unsure, keep original unchanged.
