### Changes to Context and Implementation MDs
- Research: https://notebooklm.google.com/notebook/e05368f2-4ffc-4d7b-995c-f9285ce7770f?authuser=2 
- Research Doc Overview: /Users/sameul.nasir/Downloads/Memory_Overview.md
- Context/Implementation Changes: /Users/sameul.nasir/Downloads/knowledge-layer-changes.md

### RAG & Retrieval

→ [https://lnkd.in/gc6iqNzJ](https://lnkd.in/gc6iqNzJ) — de facto standard for RAG pipelines
→ [https://lnkd.in/gJ8gKkWt](https://lnkd.in/gJ8gKkWt) — best data ingestion and indexing layer
→ [https://lnkd.in/gZ9rssT4](https://lnkd.in/gZ9rssT4) — most mature enterprise RAG framework
→ [https://lnkd.in/g59pdntX](https://lnkd.in/g59pdntX) — graph-based RAG, outperforms naive chunking on complex queries

### Fine-Tuning

→ [https://lnkd.in/gGthsfTa](https://lnkd.in/gGthsfTa) — 2-5x faster fine-tuning, 80% less memory
→ [https://lnkd.in/g5hwqBZN](https://lnkd.in/g5hwqBZN) — unified framework for 100+ models with web UI
→ [https://lnkd.in/gzyc_SQ8](https://lnkd.in/gzyc_SQ8) — official HuggingFace library for SFT, RLHF, DPO
### Vector Databases

→ [https://lnkd.in/g7CzWvVS](https://lnkd.in/g7CzWvVS) — Rust-based, best metadata filtering
→ [https://lnkd.in/gnKp9w23](https://lnkd.in/gnKp9w23) — simplest DX for prototyping
→ [https://lnkd.in/gfQeQrrx](https://lnkd.in/gfQeQrrx) — foundational similarity search, many vector DBs use it internally

### Articles
- https://www.analyticsvidhya.com/blog/2026/04/memory-systems-in-ai-agents/ (Very fresh, covers Core, Episodic, Semantic, Procedural, and Knowledge Vault layers.)
### Four-Tier Agentic Memory Architecture

A consistent four-tier standard has emerged across the leading agentic memory frameworks:

Knowledge Base Repo: https://gist.github.com/rohitg00/2067ab416f7bbe447c1977edaaa681e2

| Tier | What It Stores | Persistence | Analogue |
|------|---------------|-------------|----------|
| **Working Memory** | Current conversation, scratchpad, recent tool results | Session-scoped | Cognitive working memory |
| **Semantic Memory** | Facts, preferences, learned rules, user profile | Long-term | Semantic memory (concepts & facts) |
| **Entity/Relational Knowledge** | Named entities, relationships, knowledge graphs | Long-term | Relational/associative memory |
| **Archival Knowledge** | Full conversation logs, historical episodes, retrievable corpus | Long-term | Long-term episodic + reference memory |

**Sources:** [[1]](https://arxiv.org/abs/2309.02427) CoALA (Sumers et al., 2023) — [[2]](https://arxiv.org/abs/2310.08560) MemGPT (Packer et al., 2023) — [[3]](https://arxiv.org/abs/2504.19413) Mem0 (Chhikara et al., 2025) — [[4]](https://arxiv.org/abs/2501.00150) Zep/Graphiti (Trinh et al., 2025) — [[5]](https://arxiv.org/abs/2404.13501) Zhang et al. Survey (2024)

### Seven Framework Approaches

**MemGPT / Letta** (Apache-2.0, 22k stars) — The OS-inspired approach. Three tiers: core memory blocks (always in system prompt, ~2K tokens), archival memory (vector store for long-term facts), and recall memory (conversation log). The agent controls its own memory through tool calls (`core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `archival_memory_search`). Maximum flexibility, but requires the agent to be competent at memory management.

**Mem0** (Apache-2.0, 15k+ stars) — Memory-as-a-service with a 4-layer hierarchy: conversation (session-scoped), session (multi-turn), user (cross-session preferences), and organizational (shared across users). Write pipeline is 3-stage: (1) LLM extraction of memory-worthy facts, (2) conflict resolution against existing memories (ADD/UPDATE/DELETE/NOOP), (3) dual storage to vector DB + knowledge graph. Retrieval runs both backends in parallel and fuses results. Memory management is externalized — happens automatically without agent decisions.

**Zep / Graphiti** (Apache-2.0, temporal knowledge graph) — Built around bi-temporal fact tracking: every fact has `valid_at` (when it became true) and `invalid_at` (when it stopped being true) timestamps. Four graph elements: entities (nodes), facts (edges with temporal bounds), episodes (raw interaction chunks), and custom types. Hybrid retrieval combines semantic similarity, BM25 keyword matching, and graph traversal. Designed for scenarios where facts change over time — "user's address was X, now it's Y."

**LangGraph / LangMem** (LangChain ecosystem) — Two complementary scopes: checkpointers (conversation state, thread-scoped, auto-persisted) and stores (cross-thread shared memory, namespace-scoped). LangMem adds a cognitive taxonomy: semantic memory (facts), episodic memory (past experiences), and procedural memory (learned instructions in system prompts). Supports both "hot path" (synchronous) and "background" (async post-conversation) memory formation.

**CrewAI** (Unified Memory API) — Composite memory with a scoring formula: `score = semantic_similarity × 0.5 + recency × 0.3 + importance × 0.2`. Uses LanceDB as default vector store. Consolidation triggers at 0.85 cosine similarity — when a new memory is too similar to an existing one, they merge. Short-term (task conversation), long-term (cross-task), and entity (structured knowledge) memory types.

**OpenAI ChatGPT Memory** — Server-managed, opaque. Extracts memories as natural language facts ("User prefers Python over JavaScript"). Injected into system prompt. User can view and delete but cannot directly write. The "fully managed" end of the control spectrum.

**Anthropic Claude Memory** (claude-3-5-sonnet+) — Client-side, file-based memory tool. CRUD operations on a memory file that persists across conversations. Transparent — memories are visible and editable. Claude Code extends this with CLAUDE.md (project instructions) and auto-memory (per-user/per-project persistent notes). Pragmatic middle ground: file-based, inspectable, no infrastructure required.

**Google ADK** (Agent Development Kit) — Prefix-based key-value state with four scopes: no prefix (agent-private, conversation-scoped), `user:` (cross-session per-user), `app:` (shared across all users), and `temp:` (ephemeral, single-turn). Simple but effective for stateful workflows.

### The Control Spectrum

The fundamental architectural decision is where memory intelligence lives:

```
Agent-Managed ←—————————————————————————————→ Service-Managed
  MemGPT    CrewAI    LangMem    Zep    Mem0    ADK    ChatGPT
  (full     (score &  (hybrid    (temp  (auto   (KV    (opaque,
  control)  merge)    tools)     graph) extract) scope) system)
```

- **Agent-managed** (MemGPT): Maximum flexibility, agent reasons about what to store. Risk: agent may be bad at memory management, leading to bloat or missed important facts.
- **Framework-managed** (CrewAI): Framework handles scoring and consolidation, agent works within the structure. Risk: scoring weights may not fit all use cases.
- **Hybrid** (LangMem, Claude): Agent has memory tools but system also performs background processing. Balances flexibility with reliability.
- **Service-managed** (Zep, Mem0): External service extracts and manages memories. Consistent behavior, no memory management burden. Risk: may extract the wrong things or miss nuance.
- **Fully managed** (Google ADK, ChatGPT): Minimal agent involvement — system decides what to store and how. Simplest to operate, least flexible.

> **Recommendation:** Start service-managed for simplicity, move toward hybrid as you understand your memory access patterns. Pure agent-managed is only justified when memory reasoning is a core capability (e.g., a personal assistant handling complex preference conflicts).

### Cross-Framework Comparison

| Dimension | MemGPT/Letta | Mem0 | Zep/Graphiti | LangGraph | CrewAI |
|-----------|-------------|------|-------------|-----------|--------|
| Memory ownership | Agent self-edits | External service | External service | Hybrid | Framework-managed |
| Storage | Tiered (core/archival/recall) | Vector + Graph | Temporal KG | Checkpointer + Store | LanceDB |
| Temporal awareness | No | Partial | Full (bi-temporal) | No | No |
| Consolidation | Agent-controlled | Automatic (3-stage) | Temporal invalidation | LangMem background | 0.85 threshold |
| Multi-user | Via agents | User/org hierarchy | Namespace-scoped | Namespace-scoped | Per-crew |
| Latency | Variable | 0.20s median | <0.2s | Store-dependent | Variable |


