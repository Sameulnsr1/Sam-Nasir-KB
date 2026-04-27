# Building Agentic Memory Systems for AI Agents

A comprehensive research brief covering architectures, storage strategies, retrieval mechanisms, memory types, consolidation, and production patterns for building memory into AI agent systems.

**Date**: 2026-04-13
**Method**: Multi-source deep research across academic papers, open-source frameworks, production systems, and industry documentation.

---

## Executive Summary

Agentic memory — the ability for AI agents to persist, retrieve, and reason over information across interactions — has emerged as a critical differentiator between stateless chatbots and genuinely capable AI systems. This brief synthesizes findings from 70+ sources spanning academic research, production frameworks, and industry implementations to provide an actionable guide for building memory into agent systems.

The field has converged on a **tiered architecture** with working memory (active context), semantic memory (facts and knowledge), entity/relational memory (structured relationships), and archival memory (long-term compressed storage). The key architectural decision is not which tier to implement, but **who controls memory operations**: the agent itself (MemGPT/Letta), an external service (Mem0, Zep), or a hybrid of both (LangGraph + LangMem). Storage has shifted from pure vector databases toward **hybrid vector + graph backends**, with temporal awareness emerging as a non-negotiable requirement for production systems. Retrieval mechanisms have evolved far beyond simple cosine similarity — production systems now combine semantic search, BM25 keyword matching, graph traversal, temporal decay scoring, and importance weighting into composite retrieval pipelines. The most important lesson from production deployments: **retrieval quality matters more than storage volume**, and consolidation (deduplication, conflict resolution, temporal invalidation) is what separates toy demos from durable systems.

---

## Background

### Why Memory Matters

Large language models operate within fixed context windows. Without external memory, every conversation starts from zero. Memory transforms agents from reactive responders into systems that learn, adapt, and maintain coherent long-term relationships with users and tasks. The analogy to human cognition is instructive: humans don't store every sensory input — they extract meaning, consolidate during sleep, forget irrelevant details, and retrieve contextually. The best agentic memory systems mirror this selectivity.

### The Current Landscape

The field is roughly 2 years old as a distinct discipline. Key milestones:

- **2023-10**: MemGPT paper introduces OS-inspired virtual context management with self-editing memory [1]
- **2024**: Mem0, Zep, LangMem emerge as memory-as-a-service layers
- **2024-07**: Graphiti introduces bi-temporal knowledge graphs for agent memory [2]
- **2025**: Memory-as-a-Tool paper formalizes file-based memory operations [3]
- **2025-06**: CraniMem introduces neurocognitive-inspired memory with goal-conditioned gating [4]
- **2026**: Production deployments at scale (OpenAI ChatGPT Memory, Anthropic Claude Memory, Google ADK)

---

## 1. Memory Architecture Patterns

### 1.1 The Four-Tier Standard Architecture

A consensus architecture has emerged across frameworks, though implementations vary:

| Tier                         | Purpose                                          | Latency Target | Typical Backend                        |
| ---------------------------- | ------------------------------------------------ | -------------- | -------------------------------------- |
| **Working Memory**           | Active context, current conversation, scratchpad | <10ms          | In-context (system prompt), KV store   |
| **Semantic Memory**          | Facts, knowledge, learned preferences            | 10-50ms        | Vector DB, document store              |
| **Entity/Relational Memory** | People, concepts, relationships between entities | 10-100ms       | Knowledge graph, graph DB              |
| **Archival Memory**          | Historical interactions, compressed summaries    | 50-200ms       | Vector DB, blob storage, relational DB |

This maps loosely to cognitive science: working memory ≈ short-term memory, semantic ≈ declarative knowledge, entity/relational ≈ associative networks, archival ≈ long-term episodic storage.

### 1.2 Seven Framework Approaches

**MemGPT / Letta** (Apache-2.0, 22k GitHub stars)
The OS-inspired approach. Memory is organized into three tiers: core memory blocks (always in the system prompt, ~2K tokens), archival memory (vector store for long-term facts), and recall memory (conversation message log). The critical innovation: the **agent controls its own memory** through tool calls — `core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `archival_memory_search`. The agent decides what to remember, what to forget, and when to retrieve. This gives maximum flexibility but requires the agent to be competent at memory management — a non-trivial requirement.

**Mem0** (Apache-2.0, 15k+ stars)
Memory-as-a-service with a 4-layer hierarchy: conversation (session-scoped), session (multi-turn), user (cross-session preferences), and organizational (shared across users). The write pipeline is a 3-stage process: (1) LLM extraction of memory-worthy facts from conversation, (2) conflict resolution against existing memories (ADD/UPDATE/DELETE/NOOP decisions), (3) dual storage to vector DB + knowledge graph. Retrieval runs both backends in parallel and fuses results. The key differentiator: memory management is externalized from the agent — it happens automatically without the agent needing to decide what to store.

**Zep / Graphiti** (Apache-2.0, temporal knowledge graph)
Built around **bi-temporal fact tracking**: every fact has `valid_at` (when it became true in the world) and `invalid_at` (when it stopped being true) timestamps. Four graph elements: entities (nodes), facts (edges with temporal bounds), episodes (raw interaction chunks), and custom types. Hybrid retrieval combines semantic similarity, BM25 keyword matching, and graph traversal. Claims sub-200ms latency at production scale. Designed for scenarios where facts change over time — "user's address was X, now it's Y" — which pure vector stores handle poorly.

**LangGraph / LangMem** (LangChain ecosystem)
Two complementary memory scopes: **checkpointers** (conversation state, thread-scoped, automatic persistence after each graph node) and **stores** (cross-thread shared memory, namespace-scoped). LangMem adds a human-inspired taxonomy: semantic memory (facts via `create_manage_memory_tool`), episodic memory (past experiences via `create_search_memory_tool`), and procedural memory (learned instructions written into system prompts). Supports both "hot path" (synchronous, in the conversation flow) and "background" (asynchronous, post-conversation processing) memory formation patterns.

**CrewAI** (Unified Memory API)
Composite memory with a scoring formula: `score = semantic_similarity × 0.5 + recency × 0.3 + importance × 0.2`. Uses LanceDB as the default embedded vector store. Consolidation triggers at 0.85 cosine similarity threshold — when a new memory is too similar to an existing one, they merge rather than duplicate. Short-term (task conversation), long-term (cross-task), and entity (structured knowledge about people/concepts) memory types.

**OpenAI ChatGPT Memory**
Server-managed, opaque to the user. Extracts memories as natural language facts ("User prefers Python over JavaScript"). Injected into system prompt. User can view and delete but cannot directly write memories. Represents the "fully managed" end of the control spectrum — the system decides everything.

**Anthropic Claude Memory** (claude-3-5-sonnet and later)
Client-side, file-based memory tool (`memory_20250818`). CRUD operations on a memory file that persists across conversations. The agent reads/writes memories as explicit tool calls. Transparent to the user — memories are visible and editable. Claude Code extends this with CLAUDE.md (project instructions) and auto-memory (per-user/per-project persistent notes). Represents a pragmatic middle ground: file-based, inspectable, no infrastructure required.

**Google ADK** (Agent Development Kit)
Prefix-based key-value state system with four scopes: no prefix (agent-private, conversation-scoped), `user:` (cross-session per-user), `app:` (shared across all users of an application), and `temp:` (ephemeral, single-turn). Simple but effective for stateful agent workflows. Session-scoped by default with explicit opt-in for persistence.

### 1.3 The Control Spectrum

The fundamental architectural decision is where memory intelligence lives:

```
Agent-Managed ←————————————————————→ Service-Managed
  MemGPT/Letta    LangMem    Mem0    ChatGPT Memory
  (agent decides   (agent    (service  (fully opaque,
   everything)     has tools  decides   system decides)
                   but system  with
                   also acts)  agent input)
```

**Agent-managed** (MemGPT): Maximum flexibility, agent can reason about what to store. Risk: the agent may be bad at memory management, leading to bloat or missed important facts.

**Service-managed** (Mem0, ChatGPT): Consistent behavior, no memory management burden on the agent. Risk: the service may extract the wrong things or miss context-dependent nuance.

**Hybrid** (LangMem, Claude): Agent has memory tools but system also performs background processing. Balances flexibility with reliability.

**Recommendation**: Start service-managed for simplicity, move toward hybrid as you understand your memory access patterns. Pure agent-managed is only justified when memory reasoning is a core capability of your agent (e.g., a personal assistant that must handle complex preference conflicts).

### 1.4 Cross-Framework Comparison

| Dimension | MemGPT/Letta | Mem0 | Zep/Graphiti | LangGraph | CrewAI |
|-----------|-------------|------|-------------|-----------|--------|
| Memory ownership | Agent self-edits | External service | External service | Hybrid | Framework-managed |
| Storage | Tiered (core/archival/recall) | Vector + Graph | Temporal KG | Checkpointer + Store | LanceDB |
| Temporal awareness | No | Partial | Full (bi-temporal) | No | No |
| Consolidation | Agent-controlled | Automatic (3-stage) | Temporal invalidation | LangMem background | 0.85 threshold |
| Multi-user | Via agents | User/org hierarchy | Namespace-scoped | Namespace-scoped | Per-crew |
| Latency | Variable | 0.20s median | <0.2s | Store-dependent | Variable |

---

## 2. Memory Types and Taxonomy

### 2.1 Cognitive Science Mapping

The most useful taxonomy maps directly to cognitive science categories, adapted for AI agents:

**Episodic Memory** — Records of specific past interactions and experiences. "On March 5th, the user asked me to refactor the auth module and preferred the decorator pattern." Raw or summarized conversation logs. Used for few-shot learning from past successes/failures.

**Semantic Memory** — General knowledge extracted from episodes. "The user prefers Python, works at Company X, is building a data pipeline." Facts without specific temporal/episodic context. The backbone of personalization.

**Procedural Memory** — Learned behaviors and workflows. "When the user says 'deploy', run the CI pipeline first, then update staging, then notify the team." Instructions, rules, and processes that modify agent behavior. Often stored as system prompt additions or rule files.

**Working Memory** — Currently active information. The conversation so far, plus any retrieved memories relevant to the current turn. Bounded by context window. Managed through attention and retrieval.

### 2.2 Implementation Patterns by Type

| Memory Type | Write Pattern | Storage | Retrieval | Consolidation |
|-------------|--------------|---------|-----------|---------------|
| Episodic | Automatic (every interaction) | Message log, vector DB | Temporal + semantic | Summarization, decay |
| Semantic | Extracted by LLM or explicit save | Vector DB, KG, files | Semantic similarity | Dedup, conflict resolution |
| Procedural | User-defined or learned | System prompt, rule files | Pattern matching | Version control |
| Working | Automatic (current context) | In-context window | Attention mechanism | Eviction by relevance |

### 2.3 Memory Importance Scoring

Not all memories are equal. Production systems assign importance scores to prioritize storage and retrieval:

**Generative Agents approach** [5]: LLM rates each memory 1-10 on importance. "Eating breakfast" = 1, "Getting a promotion" = 8. Combined with recency and relevance for retrieval scoring.

**CrewAI approach**: Composite formula — `score = semantic × 0.5 + recency × 0.3 + importance × 0.2`. Weights are configurable but these defaults work well as starting points.

**CraniMem approach** [4]: Goal-conditioned gating with a neural "utility tagger" that scores memories based on relevance to the agent's current goal. Only memories exceeding a utility threshold enter the episodic buffer. Inspired by hippocampal gating in neuroscience.

**Practical recommendation**: Start with a simple 3-factor score (relevance × recency × importance). Tune weights based on your use case. For task-oriented agents, weight relevance highest. For personal assistants, weight importance highest. For temporal reasoning, weight recency highest.

---

## 3. Storage Backends

### 3.1 Vector Databases

The default choice for semantic memory. Store embedding vectors and retrieve by cosine similarity.

| Database | Type | QPS (1M vectors) | p95 Latency | Pricing Model | Best For |
|----------|------|------------------|-------------|---------------|----------|
| **Pinecone** | Managed | 200-1000 | 10-50ms | Per-pod | Production SaaS, zero-ops |
| **Weaviate** | Hybrid | 500-2000 | 5-20ms | Self-hosted or cloud | Multi-modal, hybrid search |
| **Qdrant** | Self-hosted/Cloud | 1000-5000 | 3-15ms | Self-hosted or cloud | High throughput, filtering |
| **Chroma** | Embedded | 100-500 | 1-5ms | Free (OSS) | Prototyping, local dev |
| **Milvus** | Distributed | 2000-10000 | 5-20ms | Self-hosted or Zilliz | Large scale, GPU acceleration |
| **pgvector** | Extension | 200-800 | 5-30ms | PostgreSQL pricing | Existing Postgres infra |
| **LanceDB** | Embedded | 500-2000 | 1-10ms | Free (OSS) | Embedded, serverless |
| **FAISS** | Library | 5000-50000 | <1ms | Free (OSS) | Research, in-process |

**pgvector deserves special attention**: Benchmarks show 28x lower p95 latency than Pinecone s1 for comparable workloads, with 75-79% cost reduction. If you already run PostgreSQL, pgvector eliminates an entire infrastructure dependency [6].

**LanceDB is the emerging default for embedded use cases**: Used by CrewAI as its default backend. Zero-copy columnar format, no server process, works in-process. Ideal for single-agent or local deployments.

### 3.2 Knowledge Graphs

Essential for entity/relational memory and temporal fact tracking.

| Database | Type | Claim | Query Language | Best For |
|----------|------|-------|---------------|----------|
| **Neo4j** | Managed/Self-hosted | Industry standard | Cypher | Complex traversals, ecosystem |
| **FalkorDB** | In-memory | 496x faster than Neo4j (their claim) | Cypher-compatible | Low-latency, real-time |
| **Kuzu** | Embedded | Fastest embedded | Cypher-compatible | Single-machine, no server |
| **Neptune** | AWS Managed | Fully managed | Gremlin, SPARQL | AWS-native deployments |
| **Apache AGE** | PostgreSQL extension | Use existing Postgres | openCypher | Postgres shops |

**FalkorDB and Kuzu are the most interesting for agent memory**: FalkorDB for its raw speed (critical for sub-200ms retrieval targets), Kuzu for embedded deployments that don't want a separate server process.

### 3.3 Hybrid Storage (The Production Standard)

No single backend handles all memory types well. Production systems use multiple backends:

**Mem0's architecture** (representative of the hybrid approach):
```
Write Path:
  Conversation → LLM Extraction → Conflict Resolution
                                        ↓
                              ┌─────────┴─────────┐
                              ↓                     ↓
                        Vector Store            Graph Store
                        (semantic facts)     (entity relationships)

Read Path:
  Query → ┌─────────────────┐
          ↓                  ↓
    Vector Search      Graph Traversal
          ↓                  ↓
          └────→ Fusion ←────┘
                   ↓
            Ranked Results
```

**Recommended backend stack by agent profile**:

| Agent Type | Vector | Graph | KV/State | Why |
|------------|--------|-------|----------|-----|
| Personal assistant | pgvector or Qdrant | Neo4j or FalkorDB | Redis | Needs all memory types, temporal facts about user |
| Task automation | LanceDB or Chroma | None | SQLite | Simple memory, no relationships needed |
| Research agent | Pinecone or Weaviate | Neo4j | None | Large corpus, citation graphs |
| Customer support | pgvector | FalkorDB | Redis | Fast retrieval, user entity tracking |
| Multi-agent system | Qdrant or Milvus | Neo4j | Redis | Shared memory, namespace isolation |

### 3.4 Emerging Standard: The Four-Layer Stack

```
┌─────────────────────────────┐
│     Working Memory          │  ← KV store (Redis) or in-context
├─────────────────────────────┤
│     Semantic Memory         │  ← Vector DB
├─────────────────────────────┤
│  Entity / Relational Memory │  ← Knowledge Graph
├─────────────────────────────┤
│     Archival Memory         │  ← Vector DB + Blob Storage
└─────────────────────────────┘
```

Each layer has its own backend, optimized for its access pattern. Cross-layer retrieval is orchestrated by a retrieval router that decides which layers to query based on the query type.

---

## 4. Retrieval Mechanisms

### 4.1 Embedding and Similarity Search

The foundation of semantic retrieval. Convert text to dense vectors, retrieve by approximate nearest neighbor (ANN) search.

**Embedding model comparison**:

| Model | Dimensions | MTEB Score | Latency | Cost |
|-------|-----------|------------|---------|------|
| OpenAI text-embedding-3-large | 3072 | 64.6 | ~50ms | $0.13/M tokens |
| OpenAI text-embedding-3-small | 1536 | 62.3 | ~30ms | $0.02/M tokens |
| Cohere embed-v3 | 1024 | 64.5 | ~40ms | $0.10/M tokens |
| SPECTER (S2) | 768 | — | ~20ms | Free (local) |
| MiniLM-L6-v2 | 384 | 56.3 | ~5ms | Free (local) |
| BGE-large-en-v1.5 | 1024 | 63.5 | ~15ms | Free (local) |

**ANN index types** (in order of common use):
- **HNSW** (Hierarchical Navigable Small World): Best recall/speed tradeoff, memory-intensive. Default in Qdrant, Weaviate, pgvector.
- **IVF** (Inverted File Index): Lower memory, good for very large datasets. Default in FAISS.
- **ScaNN** (Google): Optimized for x86, good throughput.
- **DiskANN**: Billion-scale, disk-based. Used when data exceeds RAM.

**Practical note**: For agent memory at typical scale (<10M memories), HNSW with default parameters is almost always sufficient. Don't over-optimize indexing — retrieval quality depends far more on embedding quality and chunking strategy than index choice.

### 4.2 Temporal and Importance Scoring

Pure semantic similarity is insufficient for agent memory. Two additional signals matter enormously:

**Temporal decay** (Generative Agents [5]):
```
recency_score = 0.995 ^ hours_since_last_access
```
Exponential decay that halves importance roughly every 6 days. Memories accessed recently get a boost — this models the human "tip of the tongue" effect.

**Combined scoring formula** (adapted from Generative Agents):
```
final_score = α_recency × recency + α_importance × importance + α_relevance × relevance
```
Where `relevance` is cosine similarity from embedding search, `importance` is an LLM-assigned 1-10 score, and `recency` is the decay function. The α weights control the personality of retrieval.

**Weight tuning guidance**:
- Task agents: α_relevance = 0.6, α_recency = 0.3, α_importance = 0.1 (prioritize what's relevant to the current task)
- Personal assistants: α_relevance = 0.4, α_recency = 0.2, α_importance = 0.4 (prioritize what matters to the user)
- Temporal reasoning: α_relevance = 0.3, α_recency = 0.5, α_importance = 0.2 (prioritize what's current)

### 4.3 Graph-Enhanced Retrieval

Graph traversal unlocks retrieval patterns impossible with vector search alone:

**Graphiti's hybrid search** (representative):
1. Semantic search over entity/fact embeddings (vector similarity)
2. BM25 keyword search over entity names and fact text
3. Graph traversal from seed entities (follow relationships)
4. Fusion of all three result sets with learned weights

**Graph traversal strategies**:
- **BFS** (Breadth-First Search): Good for "what's related to X?" — broad context gathering
- **DFS** (Depth-First Search): Good for "trace the chain from X to Y" — following a specific reasoning path
- **Random Walk with Restart**: Good for discovering indirect connections. Start at seed node, walk randomly with probability p of restarting at seed. Nodes visited most often are most relevant.
- **Community detection** (Louvain/Leiden): Group densely connected nodes into communities. Used by Microsoft GraphRAG for hierarchical summarization [7].

### 4.4 Hybrid Retrieval and Fusion

Production systems combine multiple retrieval methods. The standard fusion approach:

**Reciprocal Rank Fusion (RRF)**:
```
RRF_score(d) = Σ 1/(k + rank_i(d))
```
Where `k` is typically 60, and `rank_i(d)` is the rank of document `d` in retrieval method `i`. RRF is simple, robust, and doesn't require score normalization across methods.

**Alpha blending** (simpler alternative):
```
score = α × vector_score + (1 - α) × keyword_score
```
Where α = 0.75 is a common default (favor semantic over keyword).

### 4.5 Self-Directed Retrieval

The most sophisticated pattern: the agent decides *when and how* to retrieve, not just *what*.

**MemGPT's approach**: The agent has explicit memory tools (`archival_memory_search`, `conversation_search`) and must decide to call them. It can also chain searches — search archival memory, use results to form a more specific query, search again.

**Self-RAG** [8]: The agent generates special "reflection tokens" to decide:
1. Whether retrieval is needed at all (`[Retrieve]` / `[No Retrieve]`)
2. Whether retrieved passages are relevant (`[IsRel]` / `[IsIrrel]`)
3. Whether the generation is supported by the passage (`[IsSup]` / `[IsNotSup]`)

**CRAG** (Corrective RAG): Confidence-gated retrieval. If the agent's confidence in retrieved results is low, fall back to web search. Three confidence levels: Correct (use directly), Ambiguous (refine query), Incorrect (trigger web search).

**FLARE** (Forward-Looking Active Retrieval): Token-level confidence monitoring. During generation, if the model's probability for upcoming tokens drops below a threshold, pause, retrieve additional context, and continue generating.

**Practical recommendation**: Start with always-retrieve (query memory on every turn). Add conditional retrieval (skip when the query is clearly conversational) as a latency optimization. Full self-directed retrieval is only needed for complex multi-step reasoning tasks.

### 4.6 Retrieval Latency Budget

| Stage | Target | Method |
|-------|--------|--------|
| Vector search | 10-50ms | ANN (HNSW), pre-filtered |
| Re-ranking | 50-200ms | Cross-encoder (BGE Reranker, Cohere Rerank) |
| Graph traversal | 10-100ms | 2-3 hop BFS, indexed |
| LLM rewriting | 200-500ms | Query expansion, HyDE |
| **Total pipeline** | **<300ms** | Parallel where possible |

The 300ms total target is critical for conversational agents. Users perceive delays >500ms as sluggish. Cache aggressively: top-k results for common queries, pinned memories for frequently accessed facts.

---

## 5. Memory Lifecycle and Consolidation

### 5.1 The Write Path

Memory formation follows a pipeline from raw interaction to durable storage:

```
Conversation → Extraction → Classification → Conflict Resolution → Storage
                   ↓              ↓                   ↓                ↓
             LLM identifies   Episodic vs      ADD/UPDATE/      Vector + Graph
             memory-worthy    Semantic vs      DELETE/NOOP      + Index update
             content          Procedural       decision
```

**Mem0's 3-stage pipeline** (best documented):
1. **Extraction**: LLM identifies memory-worthy facts from conversation. Not everything is worth storing — the LLM filters.
2. **Conflict resolution**: Each extracted fact is compared against existing memories. Decision: ADD (new fact), UPDATE (modify existing), DELETE (contradicted by new info), NOOP (already stored).
3. **Storage**: Facts are written to both vector store (for semantic retrieval) and knowledge graph (for relational queries).

**Key insight**: The conflict resolution stage is what separates production systems from demos. Without it, memory bloats with duplicates and contradictions.

### 5.2 Consolidation Strategies

**Similarity-based deduplication** (CrewAI): When a new memory's cosine similarity to an existing memory exceeds 0.85, merge them rather than creating a duplicate. The merged memory retains the most recent timestamp and combines content.

**Temporal invalidation** (Graphiti/Zep): Don't delete old facts — mark them with `invalid_at` timestamps. "User lives in NYC" (valid_at: 2024-01, invalid_at: 2025-03) → "User lives in SF" (valid_at: 2025-03, invalid_at: null). This preserves history while ensuring current queries get current facts.

**Reflection-based consolidation** (Generative Agents [5]): When cumulative importance of recent memories exceeds a threshold (150 in the original paper), trigger a "reflection" that synthesizes multiple memories into higher-level insights. These reflections become memories themselves, creating a hierarchy: raw observations → reflections → meta-reflections.

**LangMem background processing**: After a conversation ends, an asynchronous process reviews the conversation, extracts memories, and merges them with existing knowledge. This avoids adding latency to the conversation flow.

**Summarization and compression**: Old episodic memories (conversation logs) are periodically summarized by an LLM and the summaries replace the raw logs. This compresses storage while preserving key information. Typical compression ratio: 10-50x.

### 5.3 When to Store vs. When to Forget

Not everything should be stored. Production heuristics:

| Signal | Store? | Rationale |
|--------|--------|-----------|
| User explicitly says "remember this" | Always | Direct user intent |
| User states a preference or fact about themselves | Yes | High-value personalization |
| User corrects the agent | Yes (as procedural) | Prevents repeat mistakes |
| Routine task execution | Summarize only | Details rarely needed later |
| Casual conversation / small talk | No | Low signal, high noise |
| Repeated access to same information | Pin / cache | Optimize retrieval path |
| Contradicts existing memory | Update existing | Maintain consistency |
| Memory hasn't been accessed in 90+ days | Candidate for archival | Recency decay signal |

### 5.4 Memory Growth Management

Without active management, memory stores grow unboundedly. Production strategies:

- **Importance-gated storage**: Only store memories above an importance threshold. Generative Agents uses LLM scoring (1-10); memories scoring ≤3 are not stored.
- **Rolling consolidation**: Periodically merge similar memories. CrewAI's 0.85 threshold, run as a background job.
- **Tiered archival**: Move cold memories (low access frequency, high age) to cheaper storage. Keep hot memories in fast-access stores.
- **Hard caps with eviction**: Set maximum memory count per scope (e.g., 1000 per user). When full, evict lowest-scoring memories.
- **Forgetting as a feature**: CraniMem's [4] bounded episodic buffer explicitly evicts memories when the buffer is full, replacing the least-useful memory with a more relevant one. This is inspired by the cognitive principle that **forgetting is not failure — it's curation**.

---

## 6. Production Patterns and Lessons

### 6.1 API Design Patterns

Five patterns have emerged for how applications interact with memory systems:

| Pattern | Example | Pros | Cons |
|---------|---------|------|------|
| **REST API** | Mem0 (`POST /memories`, `GET /search`) | Standard, any client | Network hop latency |
| **Tool Calls** | MemGPT/Letta (agent calls memory tools) | Agent-controlled, flexible | Agent must be good at memory management |
| **System Prompt Injection** | ChatGPT Memory, Claude Memory | Transparent, always available | Limited by context window |
| **Conversation Middleware** | Zep (intercepts messages, enriches context) | Zero agent changes | Opaque, hard to debug |
| **File System** | Claude Code CLAUDE.md, Gallego 2026 [3] | Inspectable, version-controlled | Manual management, no built-in retrieval |

**Recommendation**: REST API for multi-agent or multi-service architectures. Tool calls for single-agent systems where memory reasoning matters. System prompt injection for the simplest possible integration. File system for developer-facing tools (IDEs, CLI agents) where human inspectability is paramount.

### 6.2 Multi-User Memory

Production systems must isolate memory per user while allowing shared knowledge:

- **user_id scoping** (universal): Every memory is tagged with a user_id. Queries are always scoped. Non-negotiable for production.
- **Organizational memory** (Mem0): Four-level hierarchy — conversation → session → user → organization. Org-level memories are shared across all users (e.g., "our return policy is 30 days").
- **Namespace federation** (LangMem): Memories live in namespaces. Agents can read from multiple namespaces (their own + shared) but write only to their own.
- **Permission models**: Who can read vs. write to shared memory. Graphiti uses graph-level ACLs. Mem0 uses API-level scoping.

### 6.3 Failure Modes and Prevention

Seven failure modes that consistently appear in production deployments:

| # | Failure Mode | Symptoms | Prevention |
|---|-------------|----------|------------|
| 1 | **Memory Poisoning** | Wrong facts stored, agent gives incorrect answers | Validate extracted facts against source, allow user correction, maintain provenance |
| 2 | **Retrieval Failures** | Relevant memories exist but aren't found | Hybrid retrieval (vector + keyword + graph), test recall regularly, monitor retrieval precision |
| 3 | **Memory Bloat** | Storage grows unboundedly, retrieval slows | Consolidation pipeline, importance gating, hard caps with eviction |
| 4 | **Hallucinated Memories** | Agent "remembers" things that never happened | Only store from verified sources (actual conversations, not model generation), provenance tracking |
| 5 | **Stale Memory** | Outdated facts presented as current | Temporal invalidation (Graphiti pattern), explicit expiry, periodic review |
| 6 | **Over-Reliance** | Agent trusts memory over current context | Always allow current conversation to override stored memory, freshness scoring |
| 7 | **Privacy Leaks** | User A's memories leak to User B | user_id scoping at storage and retrieval layers, namespace isolation, access auditing |

### 6.4 Scaling Characteristics

| System | Median Latency | Token Savings | Scale |
|--------|---------------|---------------|-------|
| Mem0 | 0.20s | ~90% (vs. full context) | Millions of memories |
| Zep/Graphiti | <0.2s | ~85% | Production scale |
| Letta | Variable (agent-controlled) | 50-90% (depends on core memory size) | Thousands of agents |

**Four-stage architecture progression** (Zhou 2026):
1. **Prototype**: Single vector DB (Chroma/LanceDB), all-in-one
2. **MVP**: Vector DB + simple consolidation, user_id scoping
3. **Production**: Hybrid vector + graph, temporal awareness, multi-user
4. **Scale**: Distributed backends, caching layers, background consolidation, monitoring

### 6.5 Evaluation and Benchmarks

**LOCOMO benchmark** [9]: The gold standard for conversational memory evaluation. Tests long-term memory across multi-session conversations. Mem0 scores 66.9% vs. OpenAI 52.9%, establishing that dedicated memory systems outperform prompt-based approaches.

**Three-regime evaluation** (Zhang 2025):
1. **Parametric-only**: Test what the model knows from training
2. **Offline retrieval**: Fixed memory store, test retrieval quality
3. **Online retrieval**: Dynamic memory with writes and reads during evaluation

**E-MARS+**: Evaluates memory-augmented retrieval systems across multiple dimensions (accuracy, freshness, consistency, coverage).

**Practical evaluation checklist**:
- **Recall at k**: Can the system find relevant memories in the top-k results?
- **Freshness**: Does the system return current facts when they've been updated?
- **Consistency**: Are contradictory memories resolved correctly?
- **Latency**: Does end-to-end retrieval meet the <300ms target?
- **Noise tolerance**: Does performance degrade gracefully as memory grows?

### 6.6 Engineering Best Practices

Seven principles distilled from production deployments:

1. **Retrieval > Storage**: Investing in retrieval quality (better embeddings, hybrid search, re-ranking) yields more impact than storing more memories. A system that retrieves the right 5 memories from 1000 outperforms one that retrieves 50 from 10000.

2. **External-memory-first**: Don't try to fit everything in the context window. Use retrieval to bring in what's needed. The context window is working memory, not long-term storage.

3. **Consolidation is non-negotiable**: Without deduplication, conflict resolution, and temporal invalidation, memory quality degrades linearly with time. Build consolidation into the pipeline from day one.

4. **Temporal awareness matters**: Facts change. "User's favorite language is JavaScript" might be true in 2024 and false in 2025. Systems without temporal awareness serve stale facts indefinitely. Implement at minimum a `last_updated` timestamp; ideally, bi-temporal tracking (Graphiti pattern).

5. **Interpretability enables trust**: Users and developers must be able to inspect, edit, and delete memories. Opaque memory systems erode trust. Claude's file-based approach and Mem0's REST API both score well here. ChatGPT's opaque system scores poorly.

6. **Separate state from context**: Agent state (what step am I on, what have I tried) should use a different mechanism than semantic memory (what do I know about the user). State is ephemeral and structured; memory is durable and unstructured. Mixing them leads to confusion.

7. **Privacy is architecture, not policy**: user_id scoping, namespace isolation, and access controls must be built into the storage and retrieval layers. Bolting on privacy after the fact is fragile and error-prone.

---

## 7. Building Your Own: A Decision Framework

### 7.1 Start Here

Answer these questions to narrow your architecture:

1. **How many users?** Single user → embedded (LanceDB, file-based). Multi-user → server-based (Qdrant, pgvector, Mem0).

2. **Do facts change over time?** Yes → temporal awareness required (Graphiti pattern, bi-temporal tracking). No → simpler vector-only approach works.

3. **Does the agent need to reason about memory?** Yes → agent-managed with memory tools (MemGPT pattern). No → service-managed with automatic extraction (Mem0 pattern).

4. **What's your latency budget?** <100ms → embedded DB or in-memory graph. <300ms → any managed vector DB + graph. >300ms → can afford LLM-in-the-loop retrieval.

5. **Do you need relational reasoning?** "What do I know about Person X's relationship with Project Y?" → knowledge graph required. "What did the user tell me about their preferences?" → vector-only is sufficient.

### 7.2 Minimal Viable Memory System

For a first implementation, start with this stack:

```
┌────────────────────────────────────┐
│ 1. Conversation Logger             │ ← Store raw conversations
│    (SQLite or PostgreSQL)          │
├────────────────────────────────────┤
│ 2. Memory Extractor                │ ← LLM extracts facts per turn
│    (Prompted LLM, async)           │
├────────────────────────────────────┤
│ 3. Vector Store                    │ ← Store embeddings of facts
│    (pgvector or LanceDB)           │
├────────────────────────────────────┤
│ 4. Retrieval + Injection           │ ← Query → top-k → system prompt
│    (Cosine similarity + recency)   │
├────────────────────────────────────┤
│ 5. Consolidation Job               │ ← Nightly: dedup, merge, expire
│    (Cron + LLM)                    │
└────────────────────────────────────┘
```

This covers 80% of use cases. Add a knowledge graph (step 3b) when you need relational reasoning. Add temporal tracking (modify step 2) when facts change. Add importance scoring (modify step 4) when memory volume is high.

### 7.3 Reference Architectures

**Personal Assistant Agent**:
- Vector: pgvector (leverage existing Postgres)
- Graph: Neo4j (user entity model, preference relationships)
- State: Redis (session state, conversation cache)
- Write: Async extraction + conflict resolution (Mem0 pattern)
- Read: Hybrid vector + graph, recency-weighted
- Consolidation: Nightly — dedup, temporal invalidation, summarize old episodes

**Research / Knowledge Agent**:
- Vector: Qdrant or Weaviate (large corpus, multi-modal)
- Graph: Neo4j (citation graphs, concept relationships)
- Write: Explicit storage via tool calls (MemGPT pattern)
- Read: Self-directed retrieval, iterative search refinement
- Consolidation: On-demand — merge related findings, update summaries

**Customer Support Agent**:
- Vector: pgvector (customer facts, ticket history)
- Graph: FalkorDB (fast entity lookup, product relationships)
- State: Redis (conversation state, escalation tracking)
- Write: Automatic extraction (Mem0 pattern), validated against CRM
- Read: Entity-first (look up customer), then semantic (find relevant history)
- Consolidation: Real-time — update customer profile on every interaction

**Multi-Agent System**:
- Vector: Milvus or Qdrant (distributed, namespace-isolated)
- Graph: Neo4j (shared knowledge graph with ACLs)
- State: Redis (per-agent state, shared blackboard)
- Write: Each agent writes to own namespace, shared graph via coordinator
- Read: Own namespace + shared namespace, graph for cross-agent reasoning
- Consolidation: Coordinator agent merges cross-agent findings periodically

---

## 8. Open Questions

1. **Optimal memory extraction prompts**: What instructions produce the best memory extraction from conversations? No systematic study exists comparing extraction prompt strategies across domains.

2. **Memory capacity limits**: At what point does adding more memories degrade retrieval quality? Early evidence suggests diminishing returns above ~10K memories per user without aggressive consolidation, but rigorous benchmarks are lacking.

3. **Cross-session coherence**: How do you maintain a coherent user model across thousands of sessions? Current approaches (vector similarity + recency) may not capture complex preference evolution.

4. **Memory transfer and portability**: How should memories be exported, imported, or transferred between agent systems? No standard format exists. Claude's file-based approach is the most portable; Mem0's REST API allows export but in a proprietary schema.

5. **Evaluation of memory quality over time**: LOCOMO tests point-in-time retrieval, but how do you evaluate whether a memory system is improving or degrading over weeks and months of use?

6. **Learned retrieval routing**: ACGM [10] shows promise with neural predictors that learn which memories to retrieve. Can this generalize beyond the paper's narrow benchmark?

7. **Biological plausibility and cognitive alignment**: CraniMem [4] draws from neuroscience. Do cognitively-aligned architectures actually perform better, or is the analogy misleading?

8. **Privacy-preserving memory**: Can techniques like differential privacy, federated learning, or homomorphic encryption be applied to agent memory without destroying retrieval quality?

---

## Sources

### Academic Papers

[1] Sumers, T.R., Yao, S., Narasimhan, K., and Griffiths, T.L. "Cognitive Architectures for Language Agents." arXiv:2309.02427, 2023. CoALA framework mapping agent memory to cognitive science.
https://arxiv.org/abs/2309.02427

[2] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S.G., Stoica, I., and Gonzalez, J.E. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023. OS-inspired virtual context management with 3-tier memory and agent-controlled paging.
https://arxiv.org/abs/2310.08560

[3] Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., and Bernstein, M.S. "Generative Agents: Interactive Simulacra of Human Behavior." arXiv:2304.03442, 2023. Exponential decay, 1-10 importance scoring, reflection-based consolidation.
https://arxiv.org/abs/2304.03442

[4] Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." arXiv:2310.11511, 2023. Reflection tokens for metacognitive retrieval decisions.
https://arxiv.org/abs/2310.11511

[5] Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. "Reflexion: Language Agents with Verbal Reinforcement Learning." arXiv:2303.11366, 2023. Verbal self-reflection as episodic memory without weight updates.
https://arxiv.org/abs/2303.11366

[6] Yan, S., Gu, J., Zhu, Y., and Ling, Z. "Corrective Retrieval Augmented Generation (CRAG)." arXiv:2401.15884, 2024. Confidence-gated retrieval with web search fallback.
https://arxiv.org/abs/2401.15884

[7] Edge, D., Trinh, H., Cheng, N., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130, 2024. Microsoft GraphRAG — LLM-generated knowledge graphs with Louvain/Leiden community detection.
https://arxiv.org/abs/2404.16130

[8] Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and Chalef, D. "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv:2501.13956, 2025. Bi-temporal knowledge graphs with valid_at/invalid_at fact tracking.
https://arxiv.org/abs/2501.13956

[9] Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings." arXiv:2402.03216, 2024. Dense + sparse + ColBERT in a single model.
https://arxiv.org/abs/2402.03216

[10] Gao, Y., et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv:2312.10997, 2024. Comprehensive RAG survey covering retrieval, generation, and augmentation.
https://arxiv.org/abs/2312.10997

[11] Zhang, Z., et al. "A Survey on the Memory Mechanism of Large Language Model Based Agents." arXiv:2404.13501, 2024. Three-regime evaluation framework for agent memory.
https://arxiv.org/abs/2404.13501

[12] Hou, Y., Tamoto, H., and Miyashita, H. "My Agent Understands Me Better: Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents." arXiv:2404.00573, 2024. Human-like memory recall and consolidation patterns.
https://arxiv.org/abs/2404.00573

[13] Gallego, V. "Distilling Feedback into Memory-as-a-Tool." arXiv:2601.05960, 2026. File-based memory operations with feedback distillation, 42-scenario benchmark.
https://arxiv.org/abs/2601.05960

[14] Gaikwad, M. "Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents." arXiv:2603.15658, 2026. Cost-sensitive routing across memory stores.
https://arxiv.org/abs/2603.15658

[15] Mody, P., Panchal, M., Kar, R., Bhowmick, K., and Karani, R. "CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems." arXiv:2603.15642, 2026. Neural utility tagger, bounded episodic buffer, long-term knowledge graph.
https://arxiv.org/abs/2603.15642

[16] Zhou, C., Chai, H., Chen, W., et al. "Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering." arXiv:2604.08224, 2026. Comprehensive taxonomy of agent externalization mechanisms.
https://arxiv.org/abs/2604.08224

[17] Nowaczyk, S. "Architectures for Building Agentic AI." arXiv:2512.09458, 2025. Architectural patterns for production agent systems.
https://arxiv.org/abs/2512.09458

[18] Plaat, A., van Duijn, M., van Stein, N., Preuss, M., et al. "Agentic Large Language Models: A Survey." arXiv:2503.23037, 2025. Broad survey of agentic LLM architectures.
https://arxiv.org/abs/2503.23037

[19] Mem0 Research. "Mem0: Building Production-Ready AI Agent Memory." arXiv:2504.19413, 2025. Two-phase extraction pipeline, ADD/UPDATE/DELETE/NOOP operations, LOCOMO benchmark (66.9% accuracy).
https://arxiv.org/abs/2504.19413

[20] FadeMem. "FadeMem: Memory Strength-Based Forgetting for LLM Agents." arXiv:2601.18642, 2026. Memory tuples with strength, frequency, and temporal decay.
https://arxiv.org/abs/2601.18642

[21] MACLA. "Compressed Procedural Memory for LLM Agents." arXiv:2512.18950, 2025. Hierarchical procedure abstraction, 15:1 compression ratio from trajectories.
https://arxiv.org/abs/2512.18950

[22] Shi, W., et al. "FLARE: Active Retrieval Augmented Generation." arXiv:2309.07864, 2023. Token-level confidence monitoring for retrieval decisions.
https://arxiv.org/abs/2309.07864

[23] Wang, L., et al. "Survey on Memory-Augmented Neural Networks." arXiv:2402.08787, 2024. Cross-domain memory augmentation survey.
https://arxiv.org/abs/2402.08787

[24] Wang, P., et al. "A Comprehensive Survey of AI-Generated Content." arXiv:2402.17753, 2024. AIGC landscape including memory-augmented generation.
https://arxiv.org/abs/2402.17753

[25] arXiv:2504.13684v1. Memory architecture survey. 2025.
https://arxiv.org/html/2504.13684v1

[26] arXiv:2604.07863v1. Agent memory systems. 2026.
https://arxiv.org/html/2604.07863v1

### Source Code

[27] Park, J.S., et al. Generative Agents — Source code (retrieve.py, reflect.py). Stanford/Google, 2023.
https://github.com/joonspk-research/generative_agents

[28] Generative Agents — Retrieval module (exponential decay + importance + recency scoring).
https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py

[29] Generative Agents — Reflection module (threshold-triggered reflection generation).
https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/reflect.py

### Framework & Tool Documentation

[30] Letta (formerly MemGPT) — GitHub repository. OS-inspired 3-tier agent memory.
https://github.com/letta-ai/letta

[31] Letta Documentation. Agent memory guides, core/recall/archival memory management.
https://docs.letta.com

[32] Letta — Agent memory guide. Core memory blocks, recall search, archival storage.
https://docs.letta.com/guides/agents/memory

[33] Mem0 — GitHub repository. Memory layer for AI assistants, 20+ vector backends.
https://github.com/mem0ai/mem0

[34] Mem0 Documentation — Overview. Architecture, API, multi-user memory.
https://docs.mem0.ai/overview

[35] Mem0 — Memory operations (add). Two-phase extraction pipeline.
https://docs.mem0.ai/core-concepts/memory-operations/add

[36] Mem0 — Graph memory overview. Entity extraction, relation generation, Neo4j/Memgraph/Kuzu.
https://docs.mem0.ai/features/graph-memory

[37] Mem0 — Open-source graph memory features.
https://docs.mem0.ai/open-source/graph_memory/features

[38] Mem0 — Open-source graph memory overview.
https://docs.mem0.ai/open-source/graph_memory/overview

[39] Mem0 — Open-source graph memory.
https://docs.mem0.ai/open-source/graph-memory

[40] Mem0 — Vector database backends (20+ supported).
https://docs.mem0.ai/components/vectordbs/overview

[41] Mem0 — Metadata filtering for memory search.
https://docs.mem0.ai/open-source/features/metadata-filtering

[42] Mem0 — Controlling memory ingestion.
https://docs.mem0.ai/cookbooks/essentials/controlling-memory-ingestion

[43] Mem0 — Memory expiration (short and long-term TTL).
https://docs.mem0.ai/cookbooks/essentials/memory-expiration-short-and-long-term

[44] Zep — GitHub repository. Temporal knowledge graphs for agent memory.
https://github.com/getzep/zep

[45] Zep — Graphiti GitHub repository. Dynamic, temporally-aware knowledge graphs.
https://github.com/getzep/graphiti

[46] Zep Documentation. Concepts, architecture, fact invalidation, context blocks.
https://help.getzep.com/concepts

[47] Zep — Help center.
https://help.getzep.com/

[48] LangChain/LangGraph — Memory concepts. Checkpointers, stores, hot-path vs background patterns.
https://docs.langchain.com/oss/python/langgraph/memory

[49] LangMem — GitHub repository. Long-term memory for LangGraph agents.
https://github.com/langchain-ai/langmem

[50] LangMem Documentation.
https://langchain-ai.github.io/langmem/

[51] CrewAI Documentation — Memory. Unified Memory API, composite scoring, LanceDB default.
https://docs.crewai.com/concepts/memory

[52] Microsoft AutoGen — Memory. Pluggable Memory protocol, ChromaDB/Redis/Mem0 backends.
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html

[53] Microsoft GraphRAG — Documentation. LLM-generated knowledge graphs with community detection.
https://microsoft.github.io/graphrag/

[54] Google ADK — Sessions and state. Prefix-based KV state system for agent sessions.
https://adk.dev/sessions/state/

[55] AWS Bedrock — Agents memory. Managed agent memory service.
https://docs.aws.amazon.com/bedrock/latest/userguide/agents-memory.html

[56] Cognee Documentation. Cognitive CRUD: remember/recall/forget/improve.
https://docs.cognee.ai/

[57] Cognee — GitHub repository. Knowledge engine with cognitive metaphor.
https://github.com/topoteretes/cognee

[58] Anthropic — Claude Code memory. File-based CLAUDE.md memory, hierarchical loading.
https://code.claude.com/docs/en/memory

[59] Anthropic — Claude Code best practices. Context engineering, skills, subagents.
https://code.claude.com/docs/en/best-practices

[60] Anthropic — Prompt caching cookbook. Context window optimization patterns.
https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb

### Database Documentation

[61] Chroma — GitHub repository. Open-source embedding database.
https://github.com/chroma-core/chroma

[62] pgvector — GitHub repository. Open-source vector similarity search for PostgreSQL.
https://github.com/pgvector/pgvector

[63] Qdrant Documentation. Vector database with payload filtering.
https://qdrant.tech/documentation/overview/

[64] Weaviate Documentation. Vector database with modules.
https://docs.weaviate.io/weaviate

[65] Supabase — Vector columns. pgvector integration in Supabase.
https://supabase.com/docs/guides/ai/vector-columns

[66] Redis — AI documentation. Vector similarity search (VSS) module.
https://redis.io/docs/latest/develop/ai/

[67] Memgraph Documentation. In-memory graph database.
https://memgraph.com/docs/getting-started

[68] Pinecone — Vector database concepts.
https://www.pinecone.io/learn/vector-database/

### Embedding Models & Retrieval

[69] Galileo. "Mastering RAG: How to Select an Embedding Model." Embedding model comparison and selection guide.
https://www.galileo.ai/blog/mastering-rag-how-to-select-an-embedding-model

[70] Cohere — Embed API reference. embed-english-v3.0, 1024-dim embeddings.
https://docs.cohere.com/v2/reference/embed

[71] Cohere — Rerank overview. Cross-encoder reranking for retrieval.
https://docs.cohere.com/v2/docs/rerank-overview

[72] HuggingFace — BAAI/bge-m3. Multi-lingual, multi-functionality embedding model.
https://huggingface.co/BAAI/bge-m3

[73] HuggingFace — AllenAI/SPECTER2. Scientific paper embeddings.
https://huggingface.co/allenai/specter2

[74] Pinecone — Rerankers guide. Two-stage retrieval with reranking.
https://www.pinecone.io/learn/series/rag/rerankers/

[75] Pinecone — Hybrid search intro. Combining dense and sparse retrieval.
https://www.pinecone.io/learn/hybrid-search-intro/

[76] Weaviate — Hybrid search explained. BM25 + vector fusion.
https://weaviate.io/blog/hybrid-search-explained

[77] Anthropic — Contextual retrieval. Contextual embeddings and BM25 for RAG.
https://www.anthropic.com/news/contextual-retrieval

### Benchmarks & Comparisons

[78] Superlinked — Vector DB Comparison. Multi-dimensional benchmark across vector databases.
https://superlinked.com/vector-db-comparison

[79] VectorView — Vector database benchmarks. Independent performance comparisons.
https://benchmark.vectorview.ai/vectordbs.html

[80] ANN Benchmarks. Standard approximate nearest neighbor benchmarks.
https://ann-benchmarks.com/

[81] TigerData. "pgvector is now as fast as Pinecone at 75% less cost." HNSW benchmark results.
https://www.tigerdata.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost

### Blog Posts & Case Studies

[82] Letta Blog — "Agent Memory." Overview of memory architectures for stateful agents.
https://www.letta.com/blog/agent-memory

[83] Letta Blog — "RAG vs. Agent Memory." Distinguishing retrieval augmentation from persistent agent state.
https://www.letta.com/blog/rag-vs-agent-memory

[84] Letta Blog — "Stateful Agents." Building agents that maintain state across sessions.
https://www.letta.com/blog/stateful-agents

[85] Letta Blog — "Memory Blocks." Structured memory blocks for agent core memory.
https://www.letta.com/blog/memory-blocks

[86] Letta Blog — "Sleep-Time Compute." Background memory consolidation during idle periods.
https://www.letta.com/blog/sleep-time-compute

[87] Letta Blog — "Context Repositories." Git-based memory versioning with merge conflict resolution.
https://www.letta.com/blog/context-repositories

[88] Letta Blog — "Context Constitution." Principles for agent context engineering.
https://www.letta.com/blog/context-constitution

[89] Letta Blog — "Guide to Context Engineering." Practical patterns for managing agent context.
https://www.letta.com/blog/guide-to-context-engineering

[90] Letta Blog — "Continual Learning." Learning via context updates rather than weight changes.
https://www.letta.com/blog/continual-learning

[91] Letta Blog — "Skill Learning." Agents acquiring new capabilities through experience.
https://www.letta.com/blog/skill-learning

[92] Letta Case Study — Bilt Rewards. 1M+ personalized agents with tiered inference.
https://www.letta.com/case-studies/bilt

[93] Mem0 Blog — "Multi-Agent Memory Systems." Shared and isolated memory for multi-agent architectures.
https://www.mem0.ai/blog/multi-agent-memory-systems

[94] Mem0 Blog — "AI Memory for Voice Agents." Low-latency memory for real-time voice.
https://www.mem0.ai/blog/ai-memory-for-voice-agents

[95] Mem0 Blog — "CrewAI Memory Production Setup with Mem0." Fixing production memory failures in CrewAI.
https://www.mem0.ai/blog/crewai-memory-production-setup-with-mem0

[96] Mem0 Blog — "How Sunflower Scaled Personalized Recovery Support to 80,000 Users." 70-80% token reduction case study.
https://www.mem0.ai/blog/how-sunflower-scaled-personalized-recovery-support-to-80-000-users-with-mem0

[97] Mem0 Blog — "How OpenNote Scaled Personalized Visual Learning with Mem0." 40% token cost reduction for ed-tech.
https://www.mem0.ai/blog/how-opennote-scaled-personalized-visual-learning-with-mem0-while-reducing-token-costs-by-40

[98] Mem0 Blog — "Self-Host Mem0 with Docker." Deployment guide for self-hosted memory.
https://www.mem0.ai/blog/self-host-mem0-docker

[99] Mem0 Research. Papers and benchmarks from the Mem0 team.
https://www.mem0.ai/research

[100] LangChain Blog — "Memory for Agents." Hot-path vs background memory patterns.
https://blog.langchain.com/memory-for-agents

[101] LangChain Blog — "Agentic RAG with LangGraph." Agent-controlled retrieval patterns.
https://blog.langchain.com/agentic-rag-with-langgraph/

[102] LangChain Blog — "Semi-Structured Multi-Modal RAG." Handling diverse document types.
https://blog.langchain.com/semi-structured-multi-modal-rag/

[103] Microsoft Research Blog — "GraphRAG: Unlocking LLM Discovery on Narrative Private Data."
https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/

[104] FalkorDB Blog — "AI Agent Memory." Graph database patterns for agent memory.
https://falkordb.com/blog/ai-agent-memory/

[105] Weaviate Blog — "What is Agentic RAG." Agent-controlled retrieval augmented generation.
https://weaviate.io/blog/what-is-agentic-rag

[106] Lilian Weng. "LLM Powered Autonomous Agents." OpenAI, 2023. Foundational blog post on agent architectures with memory.
https://lilianweng.github.io/posts/2023-06-23-agent/

[107] Mem0 — GitHub Issue #3968. Community discussion on memory architecture.
https://github.com/mem0ai/mem0/issues/3968

[108] Mem0 — GitHub Issue #4573. Community discussion on graph memory features.
https://github.com/mem0ai/mem0/issues/4573

[109] Docs.mem0.ai — Main documentation portal.
https://docs.mem0.ai

---

## 9. Application: Knowledge Layer for Data Product Context

When agents query data products (not conversations, not personal assistants), the seven frameworks above serve as a **pattern library** — each contributes specific design patterns to specific tiers, but none provides the complete architecture. The key differentiator for data products is the **graduation chain**: patterns accumulate evidence through recurrence, graduate to enforced rules, and eventually harden into definitions. No existing framework implements this.

### 9.1 Design Patterns by Tier

| Tier | Design Pattern | Source Framework(s) | What It Provides |
|---|---|---|---|
| **Working Memory** | Attentional gating with overflow-aware context assembly — score candidate entries by utility before injection, page out low-priority entries when approaching token limits | CraniMem, MemGPT | Prevents context pollution (filters before injection, not after) and manages capacity explicitly rather than silently truncating. Addresses the finding that 97.8% of auto-stored memories are junk. |
| **Semantic Memory** | Vector similarity retrieval with deprecate-and-replace lifecycle — surface entries by meaning against query + table context; retire stale entries explicitly, never compress | Mem0 (vector retrieval path), Context Layer framework (lifecycle) | Catches semantically related entries that keyword search misses, while preventing summarization drift — the low-frequency signals that compression kills first (gift subscription churn, timezone edge cases) are exactly the ones that matter most. |
| **Entity/Relational** | Structured lookup by product ID with cadence-based staleness detection and composite retrieval ranking | Context Layer framework (lookup + staleness), CrewAI (weighted scoring) | Fast retrieval by product ID, automated freshness enforcement via `last_updated` + `review_cadence`. When an entity has many associated entries, weighted scoring (similarity + recency + importance) ranks retrieval toward fresh, high-priority items. |
| **Procedural** (cross-tier) | Recurrence tracking with threshold-based graduation to Rules and contradiction-checked writes | Context Layer framework (graduation chain), Du et al. (write governance) | Patterns accumulate evidence via recurrence count, graduate at threshold 3 or multi-product scope — no existing framework implements this. New entries validated against existing store content before commit; conflicts queued for human resolution, preventing memory poisoning. |

### 9.2 Memory Management Recommendation

The five-position control spectrum (Section 1.3) collapses to **two positions** for data product agents — a hybrid split where the boundary is reads vs. writes:

| Position | What It Governs | Why |
|---|---|---|
| **Framework-managed** | All reads/retrieval (context assembly engine selects what to load), recurrence counting in Procedural Store, automated staleness detection (cadence checks, migration cutover dates, trace contradictions) | The agent doesn't decide what context it gets — the framework decides. This is what prevents both context pollution and the 97.8% junk memory problem. |
| **Human-in-the-loop** | All writes (new Knowledge entries, Entity updates), contradiction resolution, Procedural graduation approval | Every write to the Knowledge layer goes through human review. Automated systems detect and flag — they don't resolve. This is the memory poisoning firewall. |

**The principle: automated detection, human resolution.** The spectrum is useful as a literature review tool (here's what exists), but the implementation recommendation is a specific split, not a position on a slider.

**What drops out and why:**

- **Agent-managed** (MemGPT) — the agent deciding what to remember is exactly how you get poisoned or junk-filled stores
- **Service-managed** (Zep/Mem0) — autonomous conflict resolution and bi-temporal tracking add complexity without proportional value when staleness is managed through cadence-based detection and human-reviewed writes
- **Fully managed** (ADK/ChatGPT) — no platform provider managing memory; this is team-owned infrastructure

### 9.3 What Must Be Built Custom

The frameworks provide design patterns but not the connective tissue. Four components require custom implementation:

1. **Context Assembly Engine** — the query-time orchestrator that selects which sub-store entries to load into Working Memory based on the query, table context, and active rules
2. **Procedural Store with recurrence tracking and graduation** — the counter, threshold check, and promotion workflow that moves patterns to Rules
3. **Contradiction checker** — validates new Knowledge writes against existing entries in the same store before commit
4. **Staleness monitor** — automated checks against review cadence (Entity), trace contradictions (Semantic), and reversal conditions (Episodic)

---

*Research conducted 2026-04-13. 109 sources consulted across academic papers, open-source documentation, production deployment reports, and industry analysis. Key claims verified across 2+ independent sources where possible.*
