# Autonomous Agent Planning

> Source: DeepLearning.AI · Andrew Ng + personal notes

---

## What is Autonomous Planning?

In agentic systems, the LLM doesn't just answer — it **generates a step-by-step plan** and executes it autonomously using tools. The agent decides:
- What steps are needed
- Which tool to call at each step
- What args to pass, based on previous step outputs

---

## Planning with Code Execution

LLM generates executable code as its "plan" to answer a query.

```mermaid
flowchart LR
    SP["System Prompt\n─────────────\nWrite code to solve\nthe user's query.\nReturn answer as Python\ndelimited with\n<execute_python> tags."]
    Q(["User Query\nWhat were the amounts\nof the last 5 transactions?"])
    LLM["LLM"]
    CODE["<execute_python>\nimport pandas as pd\ndf = pd.read_csv('transactions.csv')\ndf['date'] = pd.to_datetime(df['date'])\ndf_sorted = df.sort_values(by='date', ascending=False)\nlast_5 = df_sorted.head(5)\nprint(last_5['price'].to_list())\n</execute_python>"]

    SP --> LLM
    Q --> LLM
    LLM --> CODE
```

---

## Formatting Plan as JSON

Instead of free text, the LLM returns a **structured JSON plan** — each step has a tool name + args, making it machine-parseable and executable.

**System prompt instructs:**
> "Create a step-by-step plan in JSON format. Each step should have: step number, description, tool name, and args."

```mermaid
flowchart LR
    Q(["User Query\nDo you have any round\nsunglasses in stock\nunder $100?"])
    LLM["LLM"]
    JSON["{\n  'plan': [\n    { step: 1, tool: get_item_descriptions, args: {query: 'round sunglasses'} },\n    { step: 2, tool: check_inventory, args: {items: 'results from step 1'} },\n    ...\n  ]\n}"]

    Q --> LLM
    LLM --> JSON
```

**Why JSON?**
- Machine-readable → executor can loop over steps automatically
- Each step references previous step's output
- Separates planning from execution

---

## Example: Customer Service Agent

Full autonomous planning + execution flow with multiple tools.

**Available tools:** `get_item_descriptions`, `check_inventory`, `process_item_return`, `get_item_price`, `check_past_transactions`, `process_item_sale`

```mermaid
flowchart TD
    Q(["Do you have round sunglasses in stock under $100?"])
    LLM0["LLM\nPlanner"]
    PLAN["Plan:\n1. get_item_descriptions → find round sunglasses\n2. check_inventory → see if in stock\n3. get_item_price → check if under $100"]

    Q --> LLM0 --> PLAN

    PLAN --> S1

    subgraph Execution
        S1["Step 1 text"] --> LLM1["LLM"] --> T1["get_item_\ndescriptions"]
        T1 --> S2["Step 1 output\n+ Step 2 text"] --> LLM2["LLM"] --> T2["check_\ninventory"]
        T2 --> S3["Step 2 output\n+ Step 3 text"] --> LLM3["LLM"] --> T3["get_item_\nprice"]
    end
```

**Key insight:** Each step feeds its output into the next LLM call — the agent chains tool results together autonomously.

---

## Key Takeaways

- Autonomous planning = LLM generates a plan (code or JSON), then executes it step by step
- Plan can be **code** (flexible, powerful) or **JSON** (structured, easier to parse + control)
- Each execution step passes previous output as context to the next LLM call
- The system prompt defines available tools + output format of the plan
- Agent picks the right tool per step based on the plan it generated
