# LLM Evaluation — Key Concepts

> Source: AI Engineering (Chip Huyen) + personal notes

---

## 1. Evaluation Categories

### Exact / Functional Eval

Deterministic, rule-based checks. The answer is right or wrong — no ambiguity.

- **String match**: Does `output == expected_string`?
- **Regex check**: Does the response contain a valid date format?
- **Schema validation**: Is the JSON output well-formed?

**Example**: Ask "What is 2+2?" → assert output contains `"4"`

**When to use**: SQL generation, structured output, factual Q&A with a known answer, tool call validation.

---

### Approximate Eval (Semantic & Syntactic Similarity)

Used when there's no single correct answer but a **reference exists**.

#### Semantic Similarity (embedding-based)
- Compute **cosine similarity** between output embedding and reference embedding
- High score = similar meaning, even with different wording
- **Example**: `"The cat sat on the mat"` vs `"A cat was sitting on the mat"` → high similarity (~0.95)
- Tools: OpenAI embeddings, sentence-transformers, BGE models

#### Syntactic Similarity (ngram / fuzzy matching)
- **BLEU** — precision-based, measures how many ngrams in the output appear in the reference
- **ROUGE** — recall-based, commonly used for summarization
- **Fuzzy match** — character-level similarity
- Less robust to paraphrasing than semantic similarity

**When to use**: Summarization, translation, tasks where reference outputs exist.

---

### AI-as-Judge (LLM Eval)

Use a stronger LLM (e.g., GPT-4, Claude) to score another model's output.

- Subjective — result depends heavily on **judge prompt framing**
- Can be tuned: give rubric, persona, scoring criteria
- **Example**: "Rate this customer support response 1–5 for helpfulness and tone"

**Prompt tips for AI-as-judge**:
- Be specific: define what "good" means (criteria)
- Use a rubric with examples for each score
- Consider using **multiple judges** and averaging
- Watch for **position bias** (judges tend to prefer the first answer shown)

> ⚠️ Should be supplemented with exact eval or human eval — not used alone.

---

### Human Eval

Ground truth but expensive and slow.

- Used for **calibration** — to check if your automated evals agree with humans
- High-stakes tasks: safety, medical, legal
- Can use **pairwise comparison** (A vs B) rather than absolute scoring — more reliable signal

---

## 2. Ranking Models

### Independent Scoring
Evaluate each model on all test cases → rank by average score.

- Simple, parallelizable
- **Example**: Model A scores 0.82, Model B scores 0.76 on BLEU → A wins

### Comparative / Pairwise Ranking
Show two outputs side-by-side → pick winner.

- More reliable signal for subtle differences
- **Example**: ELO-style tournaments — used in [Chatbot Arena (LMSYS)](https://chat.lmsys.org/)
- ELO score updates based on win/loss: winner gains points, loser loses points

---

## 3. Loss Metrics: Cross-Entropy & Perplexity

### Cross-Entropy Loss

Measures how well the model's predicted probability distribution matches the true distribution.

**Formula**:
```
H = -Σ y_true × log(y_pred)
```

- Lower = better
- **Example**:
  - True token = `"cat"`, model assigns `P("cat") = 0.8` → loss = `-log(0.8) ≈ 0.22`
  - Model assigns `P("cat") = 0.1` → loss = `-log(0.1) ≈ 2.3` (much worse)
- Used during **training** as the primary objective for language models

---

### Perplexity (PPL)

Exponentiated average cross-entropy. Measures how "surprised" the model is by the text.

**Formula**:
```
PPL = exp(H)
```
where `H` = average cross-entropy per token.

- Lower = better (more confident predictions)
- **Intuition**: PPL = 10 means the model is as confused as if choosing uniformly among 10 options at each step

| Cross-Entropy | Perplexity |
|---------------|------------|
| 0 | 1 (perfect) |
| 1 | ~2.7 |
| 2 | ~7.4 |
| 3 | ~20 |

> ⚠️ Not directly comparable across models with different tokenizers.

---

## Summary

| Method | When to Use | Example |
|--------|-------------|---------|
| Exact eval | Clear right/wrong answer | SQL output, factual Q&A |
| Semantic similarity | Open-ended, meaning matters | RAG answer quality |
| Syntactic similarity | Summarization, translation | BLEU on MT benchmarks |
| AI-as-judge | No reference, subjective quality | Tone, helpfulness, safety |
| Cross-entropy | Training signal, token-level fit | LM training loop |
| Perplexity | Benchmarking LM fluency | Model comparison on held-out text |
