# 🔍 LLM Translation Quality Evaluation — Claude vs GPT-4o
### KO→EN Drama Dialogue · Stranger (비밀의 숲) S01E01 · Human + Automated Evaluation

A structured evaluation of Claude and GPT-4o on Korean-to-English drama dialogue translation, testing how scene context and domain glossary affect translation quality. Built by a Localization Project Manager with 6+ years of KO-EN experience, combining localization domain expertise with Python-based NLP evaluation.

---

## Background & Motivation

LLMs are increasingly used in localization workflows, but their performance on culturally nuanced, context-dependent dialogue is poorly understood — especially for Korean, where subject omission, honorifics, and idiomatic expressions create unique translation challenges.

This project evaluates two leading LLMs under three conditions of increasing contextual support, using both human expert evaluation and automated metrics (BLEU, TER). A key research question: **do automated metrics agree with human judgment for short drama dialogue?**

---
## Demo
![Dashboard Demo](https://github.com/InYoungee/llm-translation-quality-evaluation/blob/main/images/llm_eval.gif)
---

## Dataset

- **Source:** Stranger (비밀의 숲) Season 1, Episode 1 — a Korean legal thriller drama
- **74 KO dialogue strings** manually selected from subtitle files for linguistic variety
- **Content types:** legal/procedural dialogue, emotional scenes, idiomatic expressions, honorifics, sarcasm, subject-omitted sentences
- **Human reference translations:** EN subtitle file (professionally translated)
- **Raw subtitle data not included** in this repository due to copyright

---

## Methodology

### Experimental Design — 3 Conditions

Each of the 74 strings was sent to both models under three conditions:

| Condition | Prompt Contents |
|---|---|
| **A — Baseline** | Korean dialogue only |
| **B — Context** | Dialogue + scene description + tone notes |
| **C — Context + Glossary** | Dialogue + scene description + domain glossary |

Context was provided for 12 strings identified as particularly context-dependent (political conspiracy scenes, emotional grief scenes, figurative language, honorific-heavy dialogue).

### Domain Glossary

A 45-entry glossary was manually curated covering:
- Character names and titles (검사, 차장, 경위)
- Legal terminology (긴급체포, 공무집행방해, 진범, 추후보강)
- Korean idioms (독야청청, 이판사판, 맛있게 먹겠습니다, 찔러도 피 한 방울 안 나올)
- Show-specific terms (서부지검, 블랙박스, 법복)

### Human Evaluation Rubric — 5 Dimensions, 1-5 Scale

| Dimension | What It Measures |
|---|---|
| **Accuracy** | Meaning faithfully conveyed without omissions or distortions |
| **Fluency** | Natural English, no awkward phrasing |
| **Cultural Appropriateness** | Idioms, register, honorifics handled correctly |
| **Terminology** | Domain-specific terms translated correctly and consistently |
| **Naturalness** | Reads as real spoken drama dialogue |

**Max score per string: 25** (5 dimensions × 5)

Evaluation conducted by the author — a Korean native speaker and professional KO-EN localization PM with 6+ years of game localization experience.

### Automated Metrics

- **BLEU** (BiLingual Evaluation Understudy) — n-gram overlap with reference translation, 0-100 higher is better
- **TER** (Translation Edit Rate) — estimated editing effort relative to reference length, lower is better
- Computed using `sacrebleu` at corpus level across all 74 strings

---

## Key Findings

### 1. Claude and GPT-4o Perform Comparably

| Model | Avg Score (/25) | BLEU | TER |
|---|---|---|---|
| Claude | **20.65** | 11.22 | 90.30 |
| GPT-4o | 20.41 | **11.59** | **85.97** |

The gap is small — Claude edges GPT-4o on human evaluation (20.65 vs 20.41) while GPT-4o scores better on automated TER. Per string: Claude wins 21, GPT-4o wins 21, 32 ties.

### 2. Context Meaningfully Improves Quality — GPT-4o Benefits More

| Condition | Avg Score (/25) | vs Baseline |
|---|---|---|
| A: No context | 19.94 | — |
| B: + Context | 20.72 | +0.78 |
| C: + Context + Glossary | 20.93 | +0.99 |

GPT-4o shows a larger jump from A→B (+1.39) than Claude (+0.17), suggesting Claude already incorporates contextual inference by default, while GPT-4o relies more on explicit context.

For the 12 context-dependent strings specifically, the effect is dramatic: **17.83 → 21.46 → 21.00** (A→B→C), confirming that scene context is most impactful for culturally ambiguous dialogue.

### 3. Accuracy Is the Weakest Dimension

| Dimension | Avg Score (/5) |
|---|---|
| Terminology | **4.52** ← strongest |
| Fluency | 4.42 |
| Naturalness | 4.27 |
| Cultural Appropriateness | 3.89 |
| Accuracy | **3.48** ← weakest |

Both models handle terminology and fluency well but struggle most with **accuracy** — faithfully conveying meaning in culturally nuanced, idiom-heavy, or subject-omitted sentences. Accuracy improves most from context (+0.35 from A→C).

### 4. Glossary Can Hurt as Well as Help

Condition C (with glossary) scores only marginally better than Condition B (+0.21). In some cases, providing glossary notes caused models to translate too rigidly, sticking to explanations rather than producing natural output. This is a documented LLM behavior called **prompt overloading** — too much instruction can reduce flexibility.

### 5. Human Evaluation and Automated Metrics Disagree

| | Human scores | BLEU | TER |
|---|---|---|---|
| Better model | Claude | GPT-4o (BLEU) | GPT-4o |
| Best condition | C | B | B |
| Glossary effect | Slight positive | Negative | Negative |

BLEU and TER both rate Condition C as the **worst** condition, while human evaluation rates it the **best**. This disagreement highlights a well-documented limitation of automated metrics for creative dialogue translation — surface-level word overlap fails to capture meaning, cultural nuance, and appropriateness. This finding supports the case for **human-in-the-loop evaluation** in localization workflows.

### 6. Hardest Strings

| ID | KO String | Avg Score /25 | Why It's Hard                                                                                                                                                                                          |
|---|---|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 45 | 아 저거 선배가 말하는데 저거 | 7.0 | Subject omission, conversational fragment, honorific inference                                                                                                                                         |
| 18 | 그놈은 우리 법복까지 다 벗겨낼지도 몰라 | 9.3 | Figurative use of 법복 (legal robes = professional standing)                                                                                                                                             |
| 43 | 원수 꼭 갚겠습니다 | 11.0 | Literally means "repay an enemy/grudge" but actual meaning here is "I will return the favor / repay your kindness" — a playful expression of gratitude that LLMs consistently mistranslate as a threat |
| 23 | 영 영감님 | 11.7 | 영감님 is a honorific used in Korean legal culture specifically to address prosecutors — meaning "Prosecutor Young" here — not recognized by either model without explicit context                        |
| 8 | 살인사건 용의자라고요 이름, 주소 빨리 | 12.0 | Terse legal command, subject implied                                                                                                                                                                   |

---

## Qualitative Takeaways

Through the evaluation process, five key failure patterns were identified:

1. **Subject omission** — Korean frequently drops subjects; without explicit context, both models sometimes inferred the wrong speaker or addressee
2. **Glossary overloading** — overly detailed glossary notes caused rigid, unnatural translations that followed the explanation rather than the spirit
3. **Informal address forms** — terms like 어머니 used as respectful address for an older woman (not literally "mother") were often mishandled
4. **Figurative language** — idiomatic expressions (법복, 독야청청, 맛있게 먹겠습니다 used sarcastically) challenged both models without explicit context
5. **Gender ambiguity** — Korean pronouns don't mark gender; both models occasionally defaulted to wrong gendered pronouns in English output

---

## Future Work

- **RAG-based translation** — store the full script, glossary, and style guide as a knowledge base; automatically retrieve relevant context per dialogue line using Retrieval-Augmented Generation
- **Structured prompt engineering study** — systematically test targeted prompt interventions for each failure pattern (explicit subject, gender markers, few-shot examples)
- **COMET evaluation** — add neural quality metric trained on human judgments; compare with BLEU/TER/human agreement
- **Glossary extraction via NLP** — use KoNLPy to automatically extract terminology candidates from full episode scripts, rather than manual curation
- **Inter-rater reliability** — add a second Korean-fluent evaluator to calculate agreement scores

---

## Project Structure

```
├── llm_eval_dashboard.py      # Streamlit evaluation dashboard
├── condition_A.py             # API script — baseline (dialogue only)
├── condition_B.py             # API script — dialogue + context
├── condition_C.py             # API script — dialogue + context + glossary
├── compute_bleu_ter.py        # BLEU and TER computation
├── evaluation_rubric_1to5.xlsx # 5-dimension 1-5 scoring rubric
├── stranger_glossary.xlsx      # 45-entry domain glossary
├── scoring_sheet_1to5_result.xlsx # Human evaluation scores (444 rows)
├── bleu_ter_summary.xlsx       # Corpus-level BLEU/TER results
└── README.md
```

---

## How to Run the Dashboard

```bash
pip install streamlit plotly pandas openpyxl
streamlit run llm_eval_dashboard.py
```

Requires `scoring_sheet_1to5_result.xlsx` and `bleu_ter_summary.xlsx` in the same directory.

---

## Tech Stack

- **Python** — pandas, sacrebleu, requests
- **APIs** — Anthropic Claude API (`claude-3-5-sonnet`), OpenAI GPT-4o API
- **Dashboard** — Streamlit, Plotly
- **Automated metrics** — sacrebleu (BLEU, TER)
- **Data** — Stranger S01E01 KO/EN subtitle files (not included, copyright)

---

## About

Built by a Localization Project Manager with 6+ years of KO-EN game localization experience, as part of an ongoing exploration of AI/ML evaluation methodology in localization workflows.

This project is part of a broader portfolio applying Python and ML to localization problems:
- [ML-Powered MT Post-Edit Effort Predictor](https://github.com/InYoungee/ml-powered-mtpe-effort-predictor-game-localization)
- [Strategic Translation Capacity Planning — Prophet Forecasting](https://github.com/InYoungee/ml-powered-project-forecasting-resource-palnning)
- [Localization Project Dashboard](https://github.com/InYoungee/2025-localization-dashboard)

→ [LinkedIn](https://www.linkedin.com/in/inyoungee/) | [Portfolio](https://inyoungee.github.io/portfolio/)
