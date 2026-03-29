# Evaluation

How do you know if your LLM application is actually working? Evaluation is the answer — and it's one of the most important (and hardest) parts of AI engineering.

---

## Notes in This Section

| File | Topic |
|------|-------|
| [llm-evaluation.md](./llm-evaluation.md) | Eval categories, ranking models, cross-entropy, perplexity |
| [factual-consistency.md](./factual-consistency.md) | Factual consistency, benchmarks (IFEval, INFOBench) |

---

## The Evaluation Problem

Evaluating LLMs is hard because:
- Outputs are **open-ended** — there's often no single correct answer
- Quality is **subjective** — tone, helpfulness, and safety are hard to quantify
- **Distribution shift** — a model that scores well on benchmarks may fail in production

The solution: **use multiple evaluation methods together**, not just one.

---

## Quick Reference: Evaluation Methods

| Method | Best For | Limitation |
|--------|----------|------------|
| Exact / Functional | Clear right/wrong answers | Too rigid for open-ended tasks |
| Semantic similarity | Meaning-based comparison | Needs good embeddings |
| Syntactic similarity (BLEU/ROUGE) | Summarization, translation | Poor for paraphrasing |
| AI-as-judge | Subjective quality, no reference | Prompt-sensitive, biased |
| Human eval | Ground truth, calibration | Expensive, slow |
| Cross-entropy / Perplexity | Model fluency, training signal | Not interpretable for end tasks |
