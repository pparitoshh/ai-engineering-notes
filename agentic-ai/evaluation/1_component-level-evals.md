# Evaluating Agentic Systems — Component-Level Evaluation

> Source: DeepLearning.AI · Andrew Ng + personal notes

---

## Development Process Summary

The agentic system development cycle is **iterative** — Build and Analyze feed each other continuously.

```mermaid
flowchart LR
    B["🔨 BUILD\n─────────────\n• Build end-to-end system\n• Improve individual component"]
    A["🔍 ANALYZE\n─────────────\n• Examine outputs & traces\n• Build evals; compute metrics\n• Error analysis\n• Component-level evals"]

    B <--> A
```

---

## Research Agent Pipeline

```mermaid
flowchart LR
    A([User Query]) --> B[LLM\nSearch Web]

    subgraph tools1[Tools]
        T1[web search]
    end
    tools1 --> B

    B --> C[Web Search]:::evalTarget

    subgraph tools2[Tools]
        T2[web fetch]
        T3[PDF to text]
    end
    tools2 --> D

    C --> D[LLM\nFetch 5 best sources]
    D --> E[Web Fetch]
    E --> F[LLM\nWrite Essay Draft]
    F --> G([📄 Final Essay])

    classDef evalTarget fill:#ffe0e0,stroke:#e53935,stroke-width:2px,color:#c62828
```

> **Red box = component being evaluated in isolation**

---

## Why Component-Level Evaluation?

- **Overall (end-to-end) eval is important** and should always be in place
- Component-level eval **complements** the overall eval — it doesn't replace it
- End-to-end eval alone is expensive and time-consuming, and hard to pinpoint *which* component failed
- Component eval gives **clearer signal** for specific errors — avoids noise from the rest of the system
- More efficient: focused team can iterate on smaller, targeted problems faster

**Key habit:** Look at **traces** across different agentic components to find where errors occur

Use error analysis output to decide **where to focus your effort**

---

## Improving Non-LLM Components

| Component | Tune Hyperparameters | Replace |
|-----------|---------------------|---------|
| Web Search | Number of results, date range | Try different search engine |
| RAG | Similarity threshold, chunk size | Try different RAG provider |
| ML Models | Detection threshold | Swap model entirely |

---

## Improving LLM Components

| Strategy | How |
|----------|-----|
| Improve prompts | Add explicit instructions, few-shot examples |
| Try a new model | Run evals across multiple LLMs, pick best |
| Split the step | Decompose task into smaller subtasks |
| Fine-tune | Train on internal data for the specific task |

---

## Example: Evaluating the Web Search Tool

```mermaid
flowchart TD
    A[Pick component to evaluate\ne.g. Web Search] --> B[Create gold standard\nresource list for test queries]
    B --> C[Run web search\non those queries]
    C --> D[Calculate F1-score\nresults vs gold standard]
    D --> E{Good enough?}
    E -- No --> F[Tune hyperparameters\nsearch engine · top-k · date range]
    F --> C
    E -- Yes --> G[Ship / move to next component]
```

### Steps
1. **Gold standard** — curate a list of ideal web resources for a set of test queries
2. **Metric** — measure how many results match gold standard using **F1-score**
3. **Tune** — vary hyperparameters (search engine, top-k results, date range) and track the metric
4. **Repeat** per component

---

## Key Takeaways

- Complement your overall eval with **component-level evals**
- For each non-LLM component (web search, RAG retriever, ML model): build a mini eval with gold standard data
- For LLM components: use prompting → model swap → decomposition → fine-tuning, in that order
- Always track metrics as you vary hyperparameters — don't guess, measure
