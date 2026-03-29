# Factual Consistency & Benchmarks

> Source: AI Engineering (Chip Huyen) + personal notes

---

## Factual Consistency (FC)

**Factual consistency** measures whether a model's output is consistent with known facts — either from a provided context or from general world knowledge.

### Two Settings

#### 1. Local Factual Consistency (Context-Grounded)
The model's output is checked against **explicitly provided facts** in the context/prompt.

- **Example**: Context says "The sky is purple" → model says "The sky is blue" → **factually inconsistent** (relative to the given context)
- Relevant for **RAG systems** — did the model faithfully use the retrieved documents?
- Also called **faithfulness** in RAG evaluation

**How to evaluate**:
- Extract claims from the output
- Check each claim against the source context
- Tools: NLI models (Natural Language Inference), AI-as-judge

#### 2. Open Knowledge Factual Consistency
The model's output is checked against **world knowledge** — facts that should be universally true.

- **Example**: Model says "Paris is the capital of Germany" → factually inconsistent with open knowledge
- Harder to evaluate at scale — requires a knowledge base or fact-checking model

---

## Instruction Following Benchmarks

### IFEval (Google)
**Instruction Following Evaluation** — measures how well a model follows **syntactic/structural instructions**.

Focus areas:
- **Word constraints** — "use exactly 500 words", "don't use the word X"
- **Frequency** — "mention the topic at least 3 times"
- **Keywords** — "include the phrase Y"
- **Language** — "respond in French"
- **Length** — "keep the response under 100 words"
- **Detectable content** — "include a JSON block", "use bullet points"

> Best for measuring compliance with explicit formatting/structural rules, not semantic quality.

---

### INFOBench
Focuses on **content constraints** — whether the model's output contains the right information, not just the right format.

- More semantic than IFEval
- Tests whether the model includes required facts, avoids forbidden content, and addresses the question fully

---

## Summary

| Concept | What It Measures | Tools / Methods |
|---------|-----------------|-----------------|
| Local FC | Output vs. provided context | NLI models, AI-as-judge |
| Open FC | Output vs. world knowledge | Fact-checking models, KB lookup |
| IFEval | Structural / syntactic instruction following | Rule-based checks |
| INFOBench | Content / semantic instruction following | LLM-based eval |

---

## Key Takeaway

> Factual consistency and instruction following are **different dimensions** of model quality. A model can follow instructions perfectly but hallucinate facts, or be factually accurate but ignore formatting rules. Evaluate both.
