# Multi-Agent Planning

> Source: DeepLearning.AI · Andrew Ng

---

## What is Multi-Agent Planning?

Instead of one LLM doing everything, a **manager LLM** creates a plan and delegates each step to **specialist sub-agents**. Each agent has its own role, tasks, and tools.

---

## Planning with Multiple Agents

```mermaid
flowchart TD
    Q(["Create a summer marketing\ncampaign for sunglasses"])
    MGR["LLM\nMarketing Manager"]
    SP["System Prompt\n─────────────\nYou are a marketing manager\nwith the following team of agents.\n{description of agents}\nReturn a step-by-step plan."]

    subgraph Agents
        R["researcher"]
        GD["graphic designer"]
        W["writer"]
    end

    SP --> MGR
    Q --> MGR

    MGR --> PLAN["Plan:\n1. researcher → research sunglasses trends\n2. graphic designer → create ad images\n3. writer → create report\n4. Review report"]

    PLAN --> S1["Step 1 text"] --> R
    R --> S2["Step 1 output\n+ Step 2 text"] --> GD
    GD --> S3["Step 2 output\n+ Step 3 text"] --> W
```

**Key:** Each step passes previous output as context to the next agent.

---

## Linear Execution: Marketing Team Example

```mermaid
flowchart LR
    Q(["Create a summer marketing\ncampaign for sunglasses"])
    Q --> researcher

    researcher --> RES["Here are current sunglasses\ntrends and competitor offerings..."]
    RES --> graphic_designer

    graphic_designer --> VIZ["Here are 5 data visualizations\nand 5 artwork options..."]
    VIZ --> writer

    writer --> OUT(["📄 Final Report"])
```

---

## Agent Roles & Tools

| Agent | Tasks | Tools |
|-------|-------|-------|
| **Researcher** | Analyze market trends, research competitors | Web search |
| **Graphic Designer** | Create data visualizations, create artwork | Image generation, image manipulation, code execution (charts) |
| **Writer** | Transform research into report text and marketing copy | None |

---

## Key Takeaways

- Manager LLM acts as **orchestrator** — plans and delegates, doesn't execute
- Each sub-agent is specialized with its own system prompt + tools
- Outputs chain linearly: each agent's output becomes the next agent's input
- This is a **linear plan** — steps are sequential, not parallel
- More complex plans can have branching or parallel execution
