# AI-as-Judge Evaluation

> Source: AI Engineering (Chip Huyen) — Chapter 3 + personal notes

---

## What Is AI-as-Judge?

An AI judge is **not just a model — it's a system that includes both a model and a prompt**.
Altering the model, the prompt, or the model's sampling variables will change the judge's output.

**How it works (Figure 3-8)**:
```
(Question, Answer)
    ↓
Prompt template: "Given the following question and answer,
evaluate how good the answer is. Use score from 1 to 5.
Q: {question}
A: {answer}
Score:"
    ↓
GPT-4
    ↓
Score
```

Use AI-as-judge when:
- There is no reference answer (open-ended tasks)
- You need to evaluate subjective qualities: tone, helpfulness, safety, clarity
- Human eval is too slow or expensive

> ⚠️ Should always be supplemented with exact eval or human eval — not used alone.

---

## What Models Can Act as Judges?

The judge can be **stronger, weaker, or the same** as the model being evaluated. Each has tradeoffs.

**Stronger judge** — makes sense intuitively (like an exam grader knowing more than the test taker)
- Makes better judgments
- Can guide weaker models through distillation / feedback

**Weaker judge** — still useful
- Key insight: it's easier to evaluate than to generate
- Just like anyone can have an opinion on whether a song is good, even if they can't write one
- Weaker models can judge outputs of stronger models

**Zheng et al. (2023)** found:
- Stronger models are better correlated with human preference
- → People opt for the strongest judge they can afford
- But this was limited to **general-purpose judges**

**Emerging direction: Small, Specialized Judges**
- A small model fine-tuned for a specific evaluation task can outperform a large general-purpose judge
- More efficient, cheaper, and potentially more consistent

---

## AI Judge Biases

Understanding biases helps you interpret scores correctly and mitigate them.

| Bias | What It Is | Example | Mitigation |
|------|-----------|---------|------------|
| **Self-bias** | Model favors its own outputs over other models' | GPT-4 gives itself +10% win rate; Claude-v1 gives itself +25% win rate (Zheng 2023) | Use a different model as judge; use multiple judges |
| **Position bias** | Favors the first answer in pairwise comparison or list | AI judge picks Answer A over Answer B just because it appeared first | Repeat test with different orderings; average scores |
| **Recency bias** | (Humans, not AI) favor the last answer seen | Human annotators tend to pick the second option | Be aware when using human eval for pairwise comparison |
| **Verbosity bias** | Favors longer answers regardless of quality | A long but mediocre answer beats a short but precise one | Explicitly instruct judge to penalize unnecessary length |

**Why self-bias happens**: The same mechanism that helps a model compute the most likely response to generate will also score that response highly — it's baked into how the model works.

---

## Global Factual Consistency

Factual consistency (FC) of a model's output can be verified in two settings:
1. **Local FC** — against explicitly provided context (see `factual-consistency.md`)
2. **Global FC** — against open world knowledge

**Global FC**: Output is evaluated against commonly accepted world knowledge.
- Example: "The sky is blue" is a universally accepted fact — if the model says otherwise, it's globally inconsistent
- Important for: general chatbots, fact-checking tools, market research applications

### Challenge: No Context Provided
If no reliable context is given, the verification pipeline becomes:
```
Search for reliable context
    → Derive relevant facts
        → Validate model's statement against those facts
```
The hardest step is determining whether the sourced facts are themselves correct.

### Verification Approaches

#### Self-Verification
- Ask the model the **same question multiple times**
- If answers disagree across runs → the model is probably **hallucinating**
- Works because hallucinations are often inconsistent

#### Knowledge-Augmented Verification
- **Google SAFE** (Search Augmented Factuality Evaluator)
- Uses a **search engine** to retrieve evidence and verify each claim in the model's response
- More reliable than self-verification for factual claims

#### Textual Entailment
- Verifying whether a statement is consistent with a given context can be framed as a **Natural Language Inference (NLI)** problem
- Given (context, claim) → does the context **entail**, **contradict**, or is it **neutral** toward the claim?
- NLI models (e.g., trained on MNLI) can be used as lightweight factual consistency checkers

---

## Instruction Following & Evaluation Pitfalls

### The Core Warning
> How well a model performs depends on the quality of its instructions, which makes it hard to evaluate AI models. When a model performs poorly, it can be because the **model is bad** OR because the **instruction is bad**.

This ambiguity is fundamental — always question both before drawing conclusions.

### Benchmark Limitations
- A model that performs well on standard benchmarks might not perform well **on your instructions**
- Benchmarks use their own curated instructions — those may not match your use case

**Tip**: Curate your own benchmark using your actual instructions and criteria.
- If you need YAML output → include YAML instructions in your benchmark
- If you want the model to avoid saying "As a language model" → evaluate on that specific instruction

### INFOBench — Content-Based Instruction Verification
- Breaks each instruction into **yes/no verification questions**
- Example: "Make a questionnaire to help hotel guests write hotel reviews"
  - Is the generated text a questionnaire?
  - Is the questionnaire designed for hotel guests?
  - Is the questionnaire helpful for hotel guests to write reviews?
- More semantic/content-focused than IFEval (which focuses on structural/syntactic constraints)

---

## Latency Considerations in Evaluation

- Autoregressive models generate **token by token** → more tokens = higher latency
- Total latency is a function of prompt length + output length + sampling variables

**Ways to control latency**:
- Instruct the model to be concise
- Set a stopping condition for generation
- Use quantization or other inference optimizations (Chapter 9)

**Tip**: When evaluating models on latency, distinguish between:
- **Must-have** latency requirements (hard SLA)
- **Nice-to-have** latency improvements (optimization targets)

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| AI-as-judge | It's model + prompt — both matter |
| Judge strength | Stronger = better judgment; weaker = still useful (evaluation < generation) |
| Self-bias | Models favor their own outputs — use a different judge |
| Position bias | AI favors first; humans favor last (recency bias) |
| Verbosity bias | Longer ≠ better — instruct judge to account for this |
| Global FC | Verify against world knowledge via search or NLI |
| Self-verification | Ask same question multiple times — disagreement = hallucination |
| Eval pitfall | Bad performance = bad model OR bad instruction — test both |

---

## Key Takeaways

- An AI judge is only as good as its prompt — invest time in the judging criteria
- Always be aware of self-bias: never use a model to judge its own outputs without controls
- For factual claims, augment with search (Google SAFE) rather than trusting the model alone
- Build your own benchmark with your own instructions — public benchmarks won't tell you how your model behaves in your system
