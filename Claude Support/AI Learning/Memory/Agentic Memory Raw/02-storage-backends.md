# Storage Backends for Agentic Memory Systems

A comparative analysis of how different storage technologies serve the memory needs of AI agents and coding assistants. Research conducted April 2026, focusing on 2024-2026 production systems and frameworks.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Memory Type Taxonomy](#memory-type-taxonomy)
3. [Vector Databases](#1-vector-databases)
4. [Graph Databases](#2-graph-databases)
5. [Relational / SQL](#3-relational--sql-databases)
6. [Key-Value Stores](#4-key-value-stores)
7. [File-Based / Local](#5-file-based--local-storage)
8. [Hybrid Approaches](#6-hybrid-approaches)
9. [Comparison Matrix](#comparison-matrix)
10. [Decision Framework](#decision-framework)
11. [Emerging Research Directions](#emerging-research-directions)
12. [Sources](#sources)

---

## Executive Summary

No single storage backend adequately serves all memory needs of an agentic system. The industry has converged on a layered approach where different backends handle different memory types. The dominant pattern in production systems (Mem0, Zep/Graphiti, CrewAI, Letta) is a **hybrid architecture** combining vector stores for semantic recall, graph databases for entity relationships and temporal facts, and relational/key-value stores for structured state and session management.

The most significant architectural shift between 2024 and 2026 has been the move from vector-only memory (where everything is embedded and similarity-searched) toward **graph-augmented memory** that preserves relationships and temporal validity. Zep's Graphiti framework and Mem0's dual vector+graph retrieval are the clearest production examples of this shift.

Meanwhile, file-based approaches (Claude Code's CLAUDE.md, Cursor's rules files) have proven surprisingly effective for developer-facing agents where the "memory" is human-readable, version-controlled, and collaboratively maintained -- a pattern that scales through social processes rather than database engineering.

---

## Memory Type Taxonomy

Before evaluating backends, it helps to define what types of memory an agent needs. Drawing from cognitive science and the MemGPT/Letta framework (https://arxiv.org/abs/2310.08560), agent memory falls into these categories:

| Memory Type | Description | Example | Persistence | Primary Query Pattern |
|---|---|---|---|---|
| **Working / Short-term** | Current conversation context, scratchpad | Active task state, recent messages | Session-scoped | Direct access (key lookup) |
| **Episodic** | Records of past events and interactions | "Last Tuesday the user asked me to refactor auth" | Long-term | Temporal + similarity search |
| **Semantic** | Extracted facts and knowledge | "The user prefers pytest over unittest" | Long-term | Similarity search, keyword |
| **Procedural** | How-to knowledge, workflows, skills | "To deploy, run `make deploy-prod`" | Long-term | Pattern match, keyword |
| **Entity** | Knowledge about specific entities and their relationships | "Alice manages Bob; Bob works on Project X" | Long-term | Graph traversal, entity lookup |

The key insight is that each memory type has a natural affinity with certain storage backends. No single technology handles all five well.

---

## 1. Vector Databases

### How They Work

Vector databases store high-dimensional embedding representations of text (or other modalities) and retrieve them via approximate nearest neighbor (ANN) search. When an agent needs to recall something, the query is embedded and compared against stored vectors using cosine similarity, Euclidean distance, or inner product.

### Major Implementations

| Database | License | Language | Index Type | Max Dimensions | Deployment | Notable Feature |
|---|---|---|---|---|---|---|
| **Pinecone** | Proprietary | Rust | Proprietary | 20,000 | Managed cloud only | Serverless, zero-ops |
| **Weaviate** | BSD-3 | Go | HNSW variants + compression | 65,536 | Self-hosted + managed | Hybrid search (dense + sparse), multi-modal |
| **Qdrant** | Apache 2.0 | Rust | HNSW-PQ, HNSW-SQ, HNSW-BQ | 65,536 | Self-hosted + cloud | Payload filtering integrated into HNSW graph, faceted search |
| **Chroma** | Apache 2.0 | Python/Rust | HNSW | -- | In-process, persistent, client-server | Simplest local setup, multi-modal |
| **Milvus** | Apache 2.0 | Go/C++ | IVF, HNSW, DiskANN, GPU | Unlimited | Self-hosted + Zilliz Cloud | GPU-accelerated indexing, massive scale |
| **pgvector** | PostgreSQL | C | IVFFlat, HNSW | 2,000 (std) / 4,000 (half) | PostgreSQL extension | Lives alongside relational data, ACID |
| **FAISS** | MIT | C++/Python | IVF, HNSW, PQ, flat | -- | Library (in-process) | Facebook's reference implementation, no server |
| **LanceDB** | Apache 2.0 | Rust | IVF-PQ | -- | Embedded, serverless | Columnar storage, zero-copy |

Source: Superlinked VDB comparison (https://superlinked.com/vector-db-comparison)

### Memory Types Served

- **Semantic memory**: Primary strength. Embeddings capture meaning, enabling "find memories similar to this query" retrieval.
- **Episodic memory**: Partial. Can store episode embeddings with temporal metadata filters, but temporal ordering is bolted on, not native.
- **Entity memory**: Weak. No native relationship modeling. Entities are flat documents, not connected nodes.
- **Procedural memory**: Moderate. Procedures can be embedded and retrieved by similarity, but structured step-by-step workflows are awkward as vectors.

### Query Patterns and Performance

Vector search excels at one pattern: **"find the N most similar items to this query."** This maps well to semantic memory recall ("what do I know about the user's testing preferences?") but poorly to structured queries ("what changed between Tuesday and Thursday?").

**Retrieval performance** depends on the index type and configuration:
- **HNSW** (used by Qdrant, Weaviate, pgvector, Chroma): O(log N) query time, excellent recall (>95% at practical settings). Qdrant reports P50 latency of 36ms for graph traversal operations.
- **IVFFlat** (pgvector, Milvus): Faster builds but lower query performance than HNSW. Requires data before index creation.
- **Flat/brute-force**: Perfect recall, O(N) -- only practical for small collections (<100K vectors).

**Filtered search** is where implementations diverge significantly. Qdrant integrates payload filtering directly into the HNSW graph for single-pass traversal (no pre/post-filtering penalty). Pinecone maintains dual indexes (vector + metadata) supporting both pre-filtering and post-filtering strategies. pgvector uses standard PostgreSQL WHERE clauses, which can degrade ANN performance on highly selective filters.

**Independent benchmarks** (VectorView, nytimes-256-angular dataset) provide cross-database comparisons:

| Database | QPS | p95 Latency | 50K vectors/mo | 20M vectors/mo (standard) | 20M vectors/mo (high perf) |
|---|---|---|---|---|---|
| Milvus | 2,406 | 1ms | ~$65 | ~$309 | ~$2,291 |
| Weaviate | 791 | 2ms | ~$25 | ~$1,536 | -- |
| Qdrant | 326 | 4ms | ~$9 | ~$281 | ~$820 |
| Pinecone | 150 (single pod) | 1ms | ~$70 | ~$227 | ~$2,074 |
| pgvector | -- | 8ms | ~$835 (full PG instance) | -- | -- |

Source: VectorView benchmarks (https://benchmark.vectorview.ai/vectordbs.html)

Milvus leads in raw throughput (2,406 QPS) but at higher operational complexity (11 index types to tune). Qdrant is the most cost-effective at small scale (~$9/month for 50K vectors). Pinecone offers the best developer experience with the highest per-unit cost.

Source: Qdrant documentation (https://qdrant.tech/documentation/overview/), pgvector README (https://github.com/pgvector/pgvector), Pinecone documentation (https://www.pinecone.io/learn/vector-database/)

### Embedding Strategies for Agent Memory

What gets embedded is as important as where it gets stored. Agent memory systems vary in their embedding approaches:

**What gets embedded:**
- **Conversation summaries** rather than raw transcripts, reducing noise and improving retrieval precision
- **Extracted facts and preferences** as discrete memory units (e.g., "User prefers pytest over unittest")
- **Task outcomes and tool call results** for episodic memory retrieval
- **Entity descriptions** that capture both the entity and its contextual relationships

**Embedding models and dimensions:**
- 384 dimensions: lightweight models (all-MiniLM-L6-v2) -- fast, lower quality
- 768 dimensions: mid-range (msmarco-distilbert, Cohere embeddings) -- good balance
- 1536 dimensions: OpenAI text-embedding-3-small -- high quality, standard choice
- 3072 dimensions: OpenAI text-embedding-3-large -- highest quality, highest cost

CrewAI supports 10+ embedding providers (OpenAI, Cohere, Voyage, Jina, HuggingFace, Ollama, AWS Bedrock, IBM WatsonX, Google AI, Azure) with a unified interface. Its composite retrieval score blends `semantic_weight * similarity + recency_weight * decay + importance_weight * importance`, weighting semantic relevance (50%), temporal recency (30%), and assessed importance (20%) by default. (https://docs.crewai.com/concepts/memory)

For agent memory specifically, 768-1536 dimensions typically provide the best trade-off between retrieval quality and storage/latency cost. The embedding computation itself (50-200ms for API-based models) often dominates the retrieval latency, not the database query.

### Scalability

- **Pinecone**: Fully managed, scales transparently. Claims billions of vectors. Serverless architecture separates storage from compute, with a "freshness layer" that temporarily caches new vectors while the main index updates.
- **Qdrant**: Sharding (recommended 12+ shards) + replication (2+ for production). RAM-first with disk offload for cost optimization. Supports both dense and sparse vectors for hybrid semantic-lexical retrieval.
- **Milvus**: Designed for billion-scale with GPU acceleration and distributed architecture. 11 index types provide maximum tuning flexibility.
- **pgvector**: Scales with PostgreSQL -- practical to ~10M vectors per table with HNSW, beyond that requires partitioning or a dedicated vector DB.
- **Chroma**: Written primarily in Rust (66.9%), supports in-memory, client-server, and cloud modes. Best for prototyping and small-to-medium collections. 4-function core API (create, add, query, retrieve) emphasizes simplicity.

### Production Examples in Agent Memory

- **Mem0** uses vector stores as its primary memory layer, supporting 20 backends in Python (Qdrant default, plus Pinecone, Chroma, pgvector, Milvus, Weaviate, FAISS, MongoDB, Redis, Elasticsearch, and more). Vector results are returned as the main memory list, with graph-discovered relationships appended in a separate `relations` array. (https://docs.mem0.ai/components/vectordbs/overview)
- **AutoGen** supports ChromaDB and Redis as vector memory backends via its `ChromaDBVectorMemory` and `RedisMemory` classes. (https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html)
- **CrewAI** uses LanceDB as its default backend with composite scoring that blends semantic similarity (50%), recency decay (30%), and importance (20%). (https://docs.crewai.com/concepts/memory)
- **Weaviate** positions itself as a retrieval tool in agentic RAG pipelines, where "retrieved results also act as a form of long-term memory, allowing agents to recall relevant information across steps." (https://weaviate.io/blog/what-is-agentic-rag)

### When to Choose Vector Databases

**Best for:**
- Semantic similarity retrieval ("find memories related to X")
- RAG-style memory augmentation
- Rapid prototyping of memory systems
- Applications where the primary query is "what do I know about this topic?"

**Avoid when:**
- You need to model relationships between entities
- Temporal reasoning is critical ("what changed and when?")
- You need structured, filterable fact storage
- You need ACID guarantees on memory updates

---

## 2. Graph Databases

### How They Work

Graph databases store data as nodes (entities) and edges (relationships), enabling traversal queries that follow connections between entities. For agent memory, this means storing facts as triples (Entity A --[relationship]--> Entity B) and retrieving context by traversing the graph from relevant starting points.

### Major Implementations

| Database | License | Language | Query Language | Latency (P50) | Key Feature for Memory |
|---|---|---|---|---|---|
| **Neo4j** | GPL / Commercial | Java | Cypher | ~469ms | Mature ecosystem, APOC plugin, full-text + vector search |
| **FalkorDB** | Source-available | C | Cypher-compatible | ~36ms | In-memory, 496x faster than Neo4j (claimed), multi-tenant |
| **Memgraph** | Source-available | C++ | Cypher | Sub-ms (in-memory) | In-memory, built-in algorithms (PageRank, BFS, DFS, community detection), MAGE library |
| **Kuzu** | MIT | C++ | Cypher | Sub-ms (embedded) | Embedded/in-process, zero network overhead |
| **Amazon Neptune** | Proprietary | -- | Gremlin, SPARQL, openCypher | Varies | Managed, integrates with OpenSearch for full-text |
| **Apache AGE** | Apache 2.0 | C | openCypher | PostgreSQL-bound | PostgreSQL extension -- graph queries alongside relational |

Source: FalkorDB benchmarks (https://falkordb.com/blog/ai-agent-memory/), Graphiti documentation (https://github.com/getzep/graphiti)

### Memory Types Served

- **Entity memory**: Primary strength. Entities as nodes, relationships as edges, with properties on both. Native support for multi-hop traversal ("Alice manages Bob, who works on Project X, which uses Python").
- **Episodic memory**: Strong with temporal graphs. Graphiti's bi-temporal tracking stores when facts became true and when they were superseded.
- **Semantic memory**: Moderate. Facts can be stored as relationship properties, but retrieval requires knowing the entity to start from (unlike vector similarity which is query-agnostic).
- **Procedural memory**: Weak as a standalone. Workflows are sequential, not naturally graph-shaped.

### The Temporal Knowledge Graph Pattern (Graphiti/Zep)

The most significant innovation in graph-based agent memory is Zep's **Graphiti** framework (https://github.com/getzep/graphiti), which introduces bi-temporal tracking to knowledge graphs:

**Architecture:**
- **Entities (nodes)**: People, products, policies, concepts -- with summaries that evolve over time
- **Facts/Relationships (edges)**: Triples with temporal validity windows (valid_from, valid_until)
- **Episodes (provenance)**: Raw ingested data that serves as ground truth. Every derived fact traces back to an episode.
- **Custom Types (ontology)**: Developer-defined entity and edge types via Pydantic models

**Temporal Metadata:** When contradictory information arrives, Graphiti does not delete old facts. Instead, it "invalidates" them by closing their validity window. This preserves complete history while maintaining current-state accuracy. This is critical for agent memory where the user's preferences, team structure, or project state changes over time.

**Supported backends:** Neo4j 5.26+ (primary), FalkorDB 1.1.2+, Kuzu 0.11.2+, Amazon Neptune (with OpenSearch for full-text).

**Performance:** Sub-second query latency for typical traversals. Incremental updates without batch recomputation (unlike Microsoft's GraphRAG which requires full re-summarization). Concurrency controlled via semaphore to prevent LLM rate limiting during entity extraction.

**LLM Integration:** Graphiti uses an LLM (OpenAI, Anthropic, or Gemini) for autonomous entity/relationship extraction from unstructured conversation data. This is both a strength (no manual schema) and a cost (LLM calls per ingestion).

Source: Graphiti GitHub README (https://github.com/getzep/graphiti)

### How Mem0 Uses Graph Memory

Mem0 runs graph retrieval in parallel with vector retrieval (https://docs.mem0.ai/features/graph-memory):

1. **Extraction**: An LLM identifies entities, relationships, and timestamps from conversation
2. **Dual Storage**: Embeddings go to the vector store; nodes and edges go to the graph backend
3. **Parallel Retrieval**: Vector similarity search narrows candidates; graph queries discover related entities via a `relations` array
4. **No automatic reranking**: Vector results and graph results are returned separately, not merged into a single ranked list

Mem0's graph layer also includes temporal tracking: entities carry a `lastSeen` attribute enabling cleanup policies such as `MATCH (n) WHERE n.lastSeen < date() - duration('P90D') DETACH DELETE n` for pruning dormant nodes. Confidence thresholds filter low-confidence edges, and custom prompts can restrict extraction scope (e.g., "only capture people, organisations, and project links").

Supported graph backends: Neo4j (Aura or self-hosted with APOC plugin), Memgraph (Docker-deployable with schema introspection), Amazon Neptune Analytics and Neptune DB, Kuzu (embedded, in-process using file paths), Apache AGE (PostgreSQL extension enabling Cypher queries alongside SQL).

### Scalability

- **Neo4j**: Proven at enterprise scale (billions of nodes). Clustering available in Enterprise edition. But higher latency than purpose-built alternatives.
- **FalkorDB**: Claims 10,000+ multi-tenant graphs with 6x better memory efficiency than Neo4j. In-memory architecture limits dataset size to available RAM.
- **Kuzu**: Embedded model means no network overhead but limited to single-machine scale.
- **Neptune**: Managed AWS service, scales with cloud infrastructure.

### Production Examples

- **Zep** uses Graphiti (temporal knowledge graph on Neo4j/FalkorDB) as its primary memory layer. Per-user graphs where all messages are ingested and facts are extracted with temporal validity. (https://help.getzep.com/concepts)
- **Mem0** offers graph memory as an opt-in layer alongside its vector-first approach, supporting Neo4j, Memgraph, Kuzu, Neptune, and Apache AGE. (https://docs.mem0.ai/features/graph-memory)
- **FalkorDB** is used in production by AdaptX (healthcare), XR.Voyage (media), and Virtuous AI (ethical AI) for agent memory graphs. (https://falkordb.com/blog/ai-agent-memory/)

### When to Choose Graph Databases

**Best for:**
- Entity relationship tracking ("who reports to whom?", "what projects use this technology?")
- Multi-hop reasoning ("find all people connected to this project through two degrees")
- Temporal fact management ("what was true last week vs. now?")
- Applications where relationships between entities are as important as the entities themselves

**Avoid when:**
- Your primary query pattern is semantic similarity
- You need simple key-value session state
- LLM extraction costs are a concern (graph ingestion requires LLM calls)
- Your memory is mostly unstructured text without clear entities

---

## 3. Relational / SQL Databases

### How They Work

Relational databases store structured data in tables with defined schemas, enforcing data integrity through ACID transactions, foreign keys, and constraints. For agent memory, this means storing facts, sessions, user profiles, and metadata in normalized tables with SQL query access.

### Role in Agent Memory

Relational databases rarely serve as the *primary* memory retrieval layer (that role goes to vector or graph stores), but they are ubiquitous as the **state management and metadata backbone**:

- **Session tracking**: Which conversations happened, when, with what user
- **User profiles**: Structured facts about users (name, preferences, plan tier)
- **Fact tables**: Extracted knowledge with structured fields (fact_text, source, confidence, created_at, invalidated_at)
- **Agent state**: Tool permissions, model configuration, memory block contents
- **Audit trails**: What the agent did, when, and why

### pgvector: The Hybrid Play

The most important development for relational databases in agent memory is **pgvector** (https://github.com/pgvector/pgvector), which adds vector similarity search to PostgreSQL:

**Capabilities:**
- Store embeddings alongside relational data in the same table (e.g., `user_facts` table with columns for `fact_text`, `embedding vector(1536)`, `created_at`, `user_id`, `confidence`)
- HNSW and IVFFlat indexes for approximate nearest neighbor search
- Standard SQL filtering combined with vector distance ordering
- Multiple distance functions: L2 (`<->`), cosine (`<=>`), inner product (`<#`), L1 (`<+>`), Hamming (`<~>`), Jaccard (`<%>`)
- Vector types: standard float32 (up to 2,000 dimensions), half-precision float16 (up to 4,000 dimensions), binary (up to 64,000 dimensions), sparse (up to 1,000 non-zero elements)
- Full ACID compliance, point-in-time recovery, JOINs with other tables

**Performance:**
- HNSW index: configurable `m` (connections, default 16) and `ef_construction` (candidate list, default 64)
- Query-time `ef_search` parameter controls speed-recall tradeoff
- Practical for collections up to ~10M vectors per table
- Build performance scales with `maintenance_work_mem` and parallel workers

**pgvector vs. Pinecone benchmarks** (Timescale/TigerData, 50M Cohere embeddings at 768 dimensions):
- Against Pinecone s1 index: PostgreSQL achieved **28x lower p95 latency** and **16x higher query throughput** at 99% recall
- Against Pinecone p2 index: PostgreSQL delivered **1.4x lower p95 latency** and **1.5x higher throughput** at 90% recall
- Monthly hosting: PostgreSQL ~$835 vs. Pinecone s1 ~$3,241 / p2 ~$3,889 -- a **75-79% cost reduction**
- Additional advantages: tunable accuracy-performance trade-offs (Pinecone offers only three fixed index types), superior backup capabilities, and streaming filtering via pgvectorscale's StreamingDiskANN that avoids accuracy degradation on filtered queries

Source: Timescale/TigerData benchmarks (https://www.tigerdata.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost)

**Limitations:**
- Not a dedicated vector engine -- at very large scale (100M+ vectors), purpose-built vector databases outperform
- Filtered ANN can be unreliable on highly selective queries without pgvectorscale extensions
- Requires PostgreSQL expertise for production tuning

Source: pgvector GitHub (https://github.com/pgvector/pgvector)

### Zep's PostgreSQL + Graph Architecture

Zep uses a graph-based architecture where nodes represent entities and edges represent facts/relationships, with temporal tracking where "the time the fact became invalid is stored on that fact's edge in the knowledge graph." While the exact database isn't disclosed in public docs, the architecture combines structured session management with graph-based fact storage and vector search for retrieval.

Source: Zep concepts documentation (https://help.getzep.com/concepts)

### Memory Types Served

- **Working memory**: Strong. Session state, conversation history, and agent configuration are naturally tabular.
- **Semantic memory**: Moderate with pgvector. Can store and retrieve facts by similarity, but lacks the pure performance of dedicated vector DBs at scale.
- **Episodic memory**: Strong for structured episodes. Timestamps, session IDs, and user IDs enable precise temporal queries that vector stores struggle with.
- **Entity memory**: Moderate. Entities can be stored in tables with foreign key relationships, but multi-hop traversal requires recursive CTEs or multiple JOINs -- far less ergonomic than graph databases.
- **Procedural memory**: Moderate. Workflows can be stored as structured records but lack native execution semantics.

### Production Examples

- **Letta** (formerly MemGPT) uses PostgreSQL as its primary state store for agent memory blocks, conversation history, and tool state. The "core memory" blocks that agents read/write are persisted as structured records.
- **pgvector** is supported by Mem0, Supabase, and numerous RAG frameworks as a "good enough" vector store that avoids the operational complexity of a separate database.
- **Supabase** wraps pgvector with a developer-friendly API, enabling vector search via RPC functions alongside standard Postgres queries. (https://supabase.com/docs/guides/ai/vector-columns)
- **AWS Bedrock Agents** stores session summaries with configurable retention (1-365 days), using memory identifiers to associate sessions with users. Storage backend is not publicly disclosed but likely DynamoDB or Aurora. (https://docs.aws.amazon.com/bedrock/latest/userguide/agents-memory.html)

### When to Choose Relational / SQL

**Best for:**
- Session state and conversation history management
- Structured fact storage with temporal metadata
- Applications that already run PostgreSQL (add pgvector for "good enough" vector search)
- Audit trails and compliance requirements (ACID guarantees)
- Small-to-medium memory collections where a separate vector DB is overkill

**Avoid when:**
- Semantic similarity is the primary retrieval pattern at scale (>10M vectors)
- Complex relationship traversal is needed
- Schema flexibility is important (relational schemas are rigid)

---

## 4. Key-Value Stores

### How They Work

Key-value stores provide ultra-fast read/write access to data indexed by simple keys. For agent memory, they serve as the **working memory and session cache layer** -- the fastest tier in the memory hierarchy.

### Major Implementations for Agent Memory

| Store | Latency | Vector Support | Persistence | Best For |
|---|---|---|---|---|
| **Redis** | Sub-millisecond | Yes (HNSW, FLAT) | Optional (RDB/AOF) | Session state, semantic cache, vector search |
| **DynamoDB** | Single-digit ms | No (native) | Durable | Session state, user profiles, serverless |
| **Memcached** | Sub-millisecond | No | None | Pure caching layer |
| **Valkey** | Sub-millisecond | Yes (Redis-compatible) | Optional | Redis alternative (Linux Foundation fork) |

### Redis: The Swiss Army Knife

Redis has evolved significantly for AI workloads (https://redis.io/docs/latest/develop/ai/):

**Vector Search (Redis VSS):**
- FLAT and HNSW index types for vector similarity search
- Vectors stored in hashes or JSON documents
- KNN search, range queries, and metadata filtering
- Supports hybrid retrieval (vector + full-text)
- Claims competitive performance at billion-vector scale

**Semantic Caching:**
- Store LLM responses keyed by semantic similarity of the input
- Reduces redundant API calls and latency
- LangCache integration for transparent caching

**Session Management:**
- Native TTL (time-to-live) on keys for automatic session expiry
- Pub/sub for real-time memory updates across distributed agents
- Atomic operations for concurrent agent access

**Ecosystem:** Integrations with LangGraph, LangChain, LlamaIndex, Amazon Bedrock, NVIDIA NIM, Microsoft Semantic Kernel.

### Memory Types Served

- **Working memory**: Primary strength. Sub-millisecond access to current session state, conversation context, and agent scratchpad.
- **Semantic memory**: Moderate (with Redis VSS). Can serve as a vector store, though typically used for caching rather than primary long-term storage.
- **Episodic memory**: Weak as standalone. No native temporal querying or validity windows.
- **Entity memory**: Weak. Flat key-value model, no relationship traversal.
- **Procedural memory**: Weak. Can store procedure text by key, but no execution semantics.

### Google ADK's Key-Value Approach

Google's Agent Development Kit uses a prefix-based key-value state system (https://adk.dev/sessions/state/):

| Prefix | Scope | Persistence |
|---|---|---|
| (none) | Session-specific | Per-session |
| `user:` | Cross-session, per-user | User-scoped |
| `app:` | Global across application | Application-scoped |
| `temp:` | Discarded after invocation | None |

Three backend options:
- **InMemorySessionService**: Development only, lost on restart
- **DatabaseSessionService**: Persistent, production-ready
- **VertexAiSessionService**: Google Cloud managed

Values must be serializable basic types (strings, numbers, booleans, lists, dicts).

### Production Examples

- **AutoGen** supports `RedisMemory` for distributed vector memory across agent instances. (https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html)
- **Mem0** supports Redis and Valkey as vector store backends. (https://docs.mem0.ai/components/vectordbs/overview)
- **LangChain/LangGraph** uses Redis for both semantic caching and conversation memory persistence.
- **AWS Bedrock Agents** likely uses DynamoDB for session state and memory persistence (sessions indexed by memory_id with configurable TTL up to 365 days).

### When to Choose Key-Value Stores

**Best for:**
- Session state and working memory (sub-millisecond access)
- Semantic caching of LLM responses
- Distributed agent coordination (pub/sub, atomic operations)
- Serverless architectures (DynamoDB)
- Hot-path memory access where latency matters most

**Avoid when:**
- You need complex queries (joins, aggregations, graph traversal)
- Long-term archival memory with rich metadata
- Relationship modeling between entities
- Strong consistency guarantees across memory operations (Redis is eventually consistent in cluster mode)

---

## 5. File-Based / Local Storage

### How They Work

File-based memory systems store knowledge as plain text files (typically Markdown) on the local filesystem. The agent reads these files at session start and writes to them during or after sessions. There is no database, no embedding pipeline, no server -- just files.

### Claude Code's Memory Architecture

Claude Code (https://code.claude.com/docs/en/memory) implements the most sophisticated file-based memory system in production:

**Two complementary systems:**

| | CLAUDE.md Files | Auto Memory |
|---|---|---|
| **Who writes** | Human | Claude |
| **Content** | Instructions and rules | Learnings and patterns |
| **Scope** | Project, user, or org | Per working tree |
| **Loaded into** | Every session (full) | Every session (first 200 lines / 25KB) |
| **Use for** | Coding standards, workflows, architecture | Build commands, debugging insights, preferences |

**File hierarchy (most specific wins):**

| Scope | Location | Shared With |
|---|---|---|
| Managed policy | `/Library/Application Support/ClaudeCode/CLAUDE.md` | All org users |
| Project | `./CLAUDE.md` or `./.claude/CLAUDE.md` | Team (via git) |
| User | `~/.claude/CLAUDE.md` | Just you (all projects) |
| Local | `./CLAUDE.local.md` | Just you (current project) |

**Auto memory directory structure:**
```
~/.claude/projects/<project>/memory/
  MEMORY.md          # Index, loaded every session (200-line cap)
  debugging.md       # Topic files loaded on demand
  api-conventions.md
  ...
```

**Key design decisions:**
- Files loaded at session start by walking up the directory tree from cwd
- `@path/to/import` syntax for referencing other files (max 5 hops deep)
- Path-scoped rules via `.claude/rules/` with glob pattern matching
- HTML comments stripped before injection (save context tokens)
- Auto memory survives compaction; nested CLAUDE.md files do not auto-reload after `/compact`
- Symlinks supported for sharing rules across projects

Source: Claude Code memory documentation (https://code.claude.com/docs/en/memory)

**Scaling mechanism:** Claude Code's memory scales through *social processes* -- teams commit CLAUDE.md to git, review it in PRs, and prune it like code. The `claudeMdExcludes` setting lets monorepo teams skip irrelevant CLAUDE.md files from other teams.

### Cursor's Approach

Cursor uses `.cursor/rules/` files with a similar philosophy -- plain text instructions loaded into context at session start. Rules can be scoped to specific file types via glob patterns. The system is less sophisticated than Claude Code's (no auto-memory, no import syntax, no directory-walk loading) but follows the same file-based paradigm.

### Windsurf/Cascade

Windsurf (formerly Codeium) has explored persistent memory through its Cascade system, though technical details are less publicly documented than Claude Code's approach.

### Memory Types Served

- **Procedural memory**: Primary strength. "Run `npm test` before committing", "Use 2-space indentation" -- these are naturally expressed as text instructions.
- **Semantic memory**: Moderate. Facts can be written as bullet points, but retrieval is by file-read (loading everything), not by query. No similarity search.
- **Entity memory**: Weak. Can describe entities in text, but no structured relationships or traversal.
- **Episodic memory**: Weak. Can log events with dates, but no temporal querying.
- **Working memory**: N/A. Each session starts fresh; files provide *persistent* context, not working state.

### Strengths of File-Based Approaches

1. **Human-readable and editable**: Anyone can read, edit, and review memory files. No special tools needed.
2. **Version-controlled**: Git provides full history, diff, blame, and collaboration primitives for free.
3. **Zero infrastructure**: No database to deploy, maintain, backup, or scale.
4. **Composable**: Import syntax, directory hierarchies, and scoping rules create flexible organization without a schema.
5. **Debuggable**: When the agent misbehaves, you can read exactly what it was told. No black-box embedding retrieval.
6. **Team-scalable**: Knowledge compounds across team members via shared repositories.

### Limitations

1. **No semantic search**: Memory is loaded in full (or not at all). No "find the most relevant memory for this query."
2. **Context window cost**: Every line of CLAUDE.md consumes tokens. Claude Code recommends <200 lines per file, and adherence degrades with length.
3. **Manual curation required**: Humans must prune, organize, and maintain memory files. Auto-memory helps but still needs periodic review.
4. **Not suitable for large knowledge bases**: A 10,000-fact knowledge base cannot be a CLAUDE.md file. File-based works for dozens to low hundreds of facts.
5. **No structured queries**: Cannot ask "what facts were added last week?" without reading everything.

### When to Choose File-Based

**Best for:**
- Developer tools and coding agents where memory is procedural ("how to build/test/deploy")
- Small teams where memory can be collaboratively maintained
- Systems where transparency and debuggability are paramount
- Bootstrapping memory before investing in database infrastructure
- Applications where memory fits in <200 lines per scope

**Avoid when:**
- Memory exceeds what fits in a context window
- Semantic retrieval is needed ("find relevant memories for this query")
- You need to store thousands of facts about entities and their relationships
- Multiple agents need concurrent read/write access to shared memory

---

## 6. Hybrid Approaches

The most capable production systems combine multiple backends, assigning each memory type to its natural storage layer.

### Mem0: Vector + Graph Parallel Retrieval

**Architecture** (https://docs.mem0.ai/features/graph-memory):
```
Conversation
    |
    v
Extraction LLM --> entities, relationships, timestamps
    |                    |
    v                    v
Vector Store         Graph Store
(20 backends)        (Neo4j, Memgraph, Kuzu, Neptune, AGE)
    |                    |
    v                    v
Similarity Search    Graph Traversal
    |                    |
    v                    v
Candidate Memories   Related Entities (relations array)
    |                    |
    +--------+-----------+
             |
             v
        Combined Results
```

**Key design choice:** Vector and graph results are *not* automatically merged or reranked. The API returns vector similarity results as the main list and graph-discovered relationships in a separate `relations` array. This gives the application control over how to combine them.

**Supported backends:**
- Vector: 20 databases (Qdrant default, Pinecone, Chroma, pgvector, Milvus, Weaviate, FAISS, MongoDB, Redis, Elasticsearch, and more)
- Graph: Neo4j, Memgraph, Amazon Neptune, Kuzu, Apache AGE

### Zep: Temporal Knowledge Graph + Vector Search

**Architecture** (https://help.getzep.com/concepts):
- Per-user knowledge graphs where nodes are entities and edges are temporal facts
- All messages from all threads for a user are ingested into that user's graph
- Fact invalidation (not deletion) preserves history
- "Context Block" assembly combines user summary + graph-relevant facts + temporal validity markers
- Vector search for finding relevant facts; graph traversal for following relationships

### CrewAI: Unified Memory with Composite Scoring

**Architecture** (https://docs.crewai.com/concepts/memory):
- Single `Memory` class replacing separate short/long/entity memory types
- **LanceDB** as default backend (local, embedded)
- LLM-driven categorization on save (scope, categories, importance)
- Composite retrieval score: `0.5 * semantic_similarity + 0.3 * recency_decay + 0.2 * importance`
- Exponential recency decay configurable via `recency_half_life_days`

**Key insight:** CrewAI avoids separate databases for different memory types by mathematically modeling temporal decay and importance within a single vector store. This trades architectural complexity for retrieval sophistication.

### Letta (formerly MemGPT): Tiered Virtual Context Management

**Architecture** (https://arxiv.org/abs/2310.08560):
Inspired by OS virtual memory with paging between tiers:

| Tier | Analogy | Contents | Access Pattern |
|---|---|---|---|
| **Core Memory** | RAM/registers | Active persona, human info, scratchpad | Always in context, agent reads/writes directly |
| **Recall Memory** | Recent file cache | Conversation history | Search by recency, keyword, embedding |
| **Archival Memory** | Disk/cold storage | Long-term facts, documents | Vector similarity search |

- Core memory blocks are structured text that the agent edits via tool calls (`core_memory_append`, `core_memory_replace`)
- PostgreSQL serves as the state store for all tiers
- Archival memory uses vector embeddings for retrieval
- The agent *self-manages* memory by deciding when to page information between tiers

### AutoGen: Pluggable Memory Protocol

**Architecture** (https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html):
- Abstract `Memory` protocol with `add`, `query`, `update_context`, `clear`, `close`
- Pluggable backends: ListMemory (in-memory), ChromaDBVectorMemory, RedisMemory, Mem0Memory
- RAG pattern via `SimpleDocumentIndexer` for chunking and indexing documents
- Memory injected into agent context via `update_context` before each execution step

### The Emerging Standard Architecture

Across these systems, a common layered pattern is emerging:

```
+------------------------------------------+
|         WORKING MEMORY                    |
|   Key-Value Store (Redis/DynamoDB)        |
|   Current session, scratchpad, tool state |
+------------------------------------------+
              |
+------------------------------------------+
|         SEMANTIC MEMORY                   |
|   Vector Database (Qdrant/Pinecone/       |
|   pgvector/Chroma)                        |
|   Embedded facts, similarity retrieval    |
+------------------------------------------+
              |
+------------------------------------------+
|         ENTITY / RELATIONAL MEMORY        |
|   Graph Database (Neo4j/FalkorDB) +       |
|   Relational (PostgreSQL)                 |
|   Entity relationships, temporal facts,   |
|   structured metadata, audit trails       |
+------------------------------------------+
              |
+------------------------------------------+
|         ARCHIVAL / COLD STORAGE           |
|   Object Store (S3) or Document DB        |
|   Full conversation logs, raw documents   |
+------------------------------------------+
```

---

## Comparison Matrix

### Backend vs. Memory Type Fitness

| Backend | Semantic | Episodic | Procedural | Entity | Working |
|---|---|---|---|---|---|
| **Vector DB** | +++++ | ++ | +++ | + | + |
| **Graph DB** | ++ | ++++ | + | +++++ | + |
| **Relational/SQL** | ++ (pgvector: +++) | ++++ | +++ | +++ | +++ |
| **Key-Value** | ++ (Redis VSS: +++) | + | + | + | +++++ |
| **File-Based** | ++ | + | +++++ | + | -- |

Scale: +++++ = excellent, + = minimal, -- = not applicable

### Operational Characteristics

| Backend | Query Latency | Write Latency | Scalability | Ops Complexity | Cost |
|---|---|---|---|---|---|
| **Vector DB (managed)** | 10-50ms | 10-50ms | High (billions) | Low | $$-$$$ |
| **Vector DB (self-hosted)** | 5-50ms | 5-50ms | High | Medium-High | $-$$ |
| **Graph DB (Neo4j)** | 50-500ms | 50-200ms | High | High | $$-$$$ |
| **Graph DB (FalkorDB)** | 5-50ms | 10-50ms | Medium-High | Medium | $-$$ |
| **PostgreSQL + pgvector** | 10-100ms | 5-20ms | Medium (10M vectors) | Low-Medium | $ |
| **Redis** | <1ms | <1ms | High | Medium | $$ |
| **DynamoDB** | 1-10ms | 1-10ms | Very High | Low | $ (at scale: $$$) |
| **File-Based** | N/A (bulk load) | N/A (file write) | Low (context window) | None | Free |

### Framework Backend Support

| Framework | Vector | Graph | Relational | KV | File |
|---|---|---|---|---|---|
| **Mem0** | 20 backends | Neo4j, Memgraph, Kuzu, Neptune, AGE | -- | Redis, Valkey | -- |
| **Zep/Graphiti** | Integrated | Neo4j, FalkorDB, Kuzu, Neptune | PostgreSQL (metadata) | -- | -- |
| **CrewAI** | LanceDB (default) | -- | -- | -- | -- |
| **Letta/MemGPT** | Archival (embedded) | -- | PostgreSQL (state) | -- | -- |
| **AutoGen** | Chroma, Redis, Mem0 | -- | -- | Redis | -- |
| **LangGraph** | Various via integrations | -- | PostgreSQL (checkpoints) | Redis | -- |
| **Claude Code** | -- | -- | -- | -- | CLAUDE.md + auto memory |
| **Google ADK** | -- | -- | Database service | In-memory, Vertex AI | -- |

---

## Decision Framework

### Choose Your Backend Based on Your Memory Profile

**Profile 1: Semantic Recall Agent**
- Primary need: "Find relevant past knowledge for this query"
- Examples: Customer support bot, research assistant
- **Recommended:** Vector DB (Qdrant or Pinecone) + PostgreSQL for session state
- **Why:** Semantic similarity is the dominant query pattern; relational DB handles the structured metadata

**Profile 2: Relationship-Aware Agent**
- Primary need: "Understand how entities relate and how facts change over time"
- Examples: Enterprise knowledge management, CRM copilot, project management agent
- **Recommended:** Graphiti (Neo4j or FalkorDB) + Vector DB for semantic search
- **Why:** Entity relationships and temporal facts are first-class; vector search fills the semantic gap

**Profile 3: Developer/Coding Agent**
- Primary need: "Remember project conventions, build commands, and user preferences"
- Examples: Claude Code, Cursor, coding assistants
- **Recommended:** File-based (CLAUDE.md pattern) + optional vector store for large codebases
- **Why:** Procedural memory is naturally expressed as text; version control provides collaboration and history for free

**Profile 4: High-Performance Session Agent**
- Primary need: "Ultra-fast access to current conversation state across distributed agents"
- Examples: Real-time chatbot fleet, multi-agent orchestration
- **Recommended:** Redis (working memory) + Vector DB (long-term) + PostgreSQL (audit)
- **Why:** Sub-millisecond KV access for hot path; vector DB for cold recall; SQL for durability

**Profile 5: Full-Stack Memory Agent**
- Primary need: "All memory types, enterprise-grade, production-ready"
- Examples: Personal AI assistant, enterprise copilot
- **Recommended:** Mem0 or Zep (handles the hybrid architecture) or custom: Redis (working) + Qdrant (semantic) + Neo4j/FalkorDB (entity/temporal) + PostgreSQL (state/audit)
- **Why:** No single backend does everything; the framework abstracts the multi-backend complexity

### Cost-Complexity Tradeoff

```
                        HIGH
                         |
                         |  Custom Hybrid
                         |  (Redis+Qdrant+Neo4j+PG)
                         |
         Capability      |     Mem0 / Zep
                         |
                         |  PG + pgvector
                         |
                         |  Single Vector DB
                         |
                         |  File-Based (CLAUDE.md)
                        LOW
                         +-------------------------->
                        LOW                       HIGH
                             Operational Complexity
```

The file-based approach offers the best capability-to-complexity ratio for small teams and developer tools. As memory needs grow beyond what fits in a context window, pgvector-in-PostgreSQL is the next natural step (one database, two capabilities). Dedicated vector DBs and graph DBs become justified at scale or when their specific query patterns are dominant.

---

## Emerging Research Directions

### Learned Graph Memory for Adaptive Retrieval

The ACGM system (Forouzandeh et al., 2026) represents a significant advance in graph-based agent memory. Rather than using fixed similarity thresholds for building memory graphs, ACGM employs "a neural predictor that estimates the probability that observation i is relevant to observation j based on task feedback." This policy-gradient approach discovers task-specific retrieval patterns that static thresholds cannot capture, with edges created only when predicted relevance exceeds 0.5.

Key findings:
- Learned graph sparsity (3.2 edges/node vs. 8.7 for dense approaches) enables O(log T) hierarchical retrieval through a two-tier system -- a **3.3x speedup** over flat retrieval
- The system discovered that **visual memory decays 4.3x faster than text** (lambda_v=0.47 vs. lambda_x=0.11), informed by cognitive science principles
- Per-modality decay rates improved precision by 8.7 points over uniform temporal weighting
- Achieved 82.7 nDCG@10 (+9.3 over GPT-4o) and 89.2% Precision@10 across WebShop, VisualWebArena, and Mind2Web benchmarks

Implication for storage backends: future memory systems may adaptively tune their graph structure and retrieval parameters based on task feedback, rather than relying on fixed configurations.

Source: Forouzandeh et al. (2026), "Task-Adaptive Retrieval over Agentic Multi-Modal Web Histories via Learned Graph Memory" (https://arxiv.org/html/2604.07863v1)

### Context-Aware Cognitive Augmentation

Research on cognitive augmentation for knowledge workers (Xiangrong et al., 2025) found that "rigid, one-size-fits-all AI models failed to support diverse cognitive needs." Users employ fundamentally different organizational strategies -- broad-first (conceptual frameworks before detail) vs. detail-first (comprehensive capture before synthesis) -- and memory systems that impose a single retrieval mode fail to support both.

The paper also found that personal notes "focused more on subjective impressions than on structured content, making later recall difficult," suggesting agent memory systems should capture and index subjective context (impressions, feelings about a decision) alongside structured facts.

Implication for storage backends: multi-modal retrieval interfaces (browsing, targeted search, temporal navigation, structured queries) should be supported rather than optimizing for a single query pattern. This reinforces the case for hybrid storage architectures.

Source: Xiangrong et al. (2025), "Intelligent Interaction Strategies for Context-Aware Cognitive Augmentation" (https://arxiv.org/html/2504.13684v1)

### Microsoft GraphRAG: Community Detection for Global Reasoning

Microsoft's GraphRAG system demonstrated that LLM-generated knowledge graphs with hierarchical community detection can answer questions that baseline RAG fails on entirely. The system applies bottom-up clustering on graph structure, creating "pre-summarization of semantic concepts and themes, which aids in holistic understanding of the dataset." Where baseline RAG failed to answer "What has Novorossiya done?" despite relevant documents existing (vector search didn't retrieve them), GraphRAG's graph structure enabled traversing entity relationships to synthesize comprehensive answers with source provenance.

Implication for storage backends: graph databases with community detection (Neo4j's GDS library, Memgraph's MAGE algorithms) enable a class of "global reasoning" queries that no amount of vector similarity search can answer.

Source: Microsoft Research (https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

### LangChain's Memory Taxonomy

LangChain's framework identifies three memory types mapped to human cognition:
- **Procedural memory**: Agent's core operational setup (LLM weights, code) -- currently difficult to update dynamically
- **Semantic memory**: Knowledge repository for facts, extracted from conversations and injected into prompts for personalization
- **Episodic memory**: Storage of past action sequences, implemented via few-shot example prompting to guide agents toward correct procedures

Two update patterns emerge: **"in the hot path"** (agent explicitly decides what to remember via tool calling, adds latency) and **"in the background"** (asynchronous updates during/after conversations, no latency penalty but delayed availability). Most production systems are converging on the background approach for writes, with synchronous reads.

Source: LangChain blog (https://blog.langchain.com/memory-for-agents)

---

## Sources

1. Mem0 Graph Memory Documentation - https://docs.mem0.ai/features/graph-memory
2. Mem0 Vector Database Overview - https://docs.mem0.ai/components/vectordbs/overview
3. Graphiti (Zep) GitHub Repository - https://github.com/getzep/graphiti
4. Zep Concepts Documentation - https://help.getzep.com/concepts
5. Qdrant Documentation Overview - https://qdrant.tech/documentation/overview/
6. pgvector GitHub Repository - https://github.com/pgvector/pgvector
7. FalkorDB AI Agent Memory - https://falkordb.com/blog/ai-agent-memory/
8. Redis AI Documentation - https://redis.io/docs/latest/develop/ai/
9. Claude Code Memory Documentation - https://code.claude.com/docs/en/memory
10. Claude Code Best Practices - https://code.claude.com/docs/en/best-practices
11. CrewAI Memory Documentation - https://docs.crewai.com/concepts/memory
12. AutoGen Memory Documentation - https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html
13. Google ADK Session State - https://adk.dev/sessions/state/
14. AWS Bedrock Agent Memory - https://docs.aws.amazon.com/bedrock/latest/userguide/agents-memory.html
15. Superlinked Vector DB Comparison - https://superlinked.com/vector-db-comparison
16. MemGPT Paper (arXiv:2310.08560) - https://arxiv.org/abs/2310.08560
17. Weaviate Agentic RAG - https://weaviate.io/blog/what-is-agentic-rag
18. LangChain Memory for Agents - https://blog.langchain.com/memory-for-agents
19. Supabase pgvector Integration - https://supabase.com/docs/guides/ai/vector-columns
20. Anthropic Prompt Caching - https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb
21. ANN Benchmarks - https://ann-benchmarks.com/
22. Zhang et al., "A Survey on the Memory Mechanism of Large Language Model Based Agents" (arXiv:2404.13501) - https://arxiv.org/abs/2404.13501
23. VectorView Vector Database Benchmarks - https://benchmark.vectorview.ai/vectordbs.html
24. Timescale/TigerData: pgvector vs Pinecone Performance & Cost - https://www.tigerdata.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost
25. Pinecone Vector Database Documentation - https://www.pinecone.io/learn/vector-database/
26. Forouzandeh et al. (2026), "Task-Adaptive Retrieval over Agentic Multi-Modal Web Histories via Learned Graph Memory" (arXiv:2604.07863) - https://arxiv.org/html/2604.07863v1
27. Xiangrong et al. (2025), "Intelligent Interaction Strategies for Context-Aware Cognitive Augmentation" (arXiv:2504.13684) - https://arxiv.org/html/2504.13684v1
28. Microsoft Research: GraphRAG Blog - https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
29. Memgraph Documentation - https://memgraph.com/docs/getting-started
30. Chroma GitHub Repository - https://github.com/chroma-core/chroma
31. Weaviate Documentation - https://docs.weaviate.io/weaviate
32. Mem0 Graph Memory Features - https://docs.mem0.ai/open-source/graph_memory/features
