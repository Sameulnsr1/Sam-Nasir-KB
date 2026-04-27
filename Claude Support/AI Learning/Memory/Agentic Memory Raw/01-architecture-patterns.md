# Agentic Memory System Architectures

**Research Chunk 01 -- Architecture Patterns, Frameworks, and Design Decisions**
*Compiled: 2026-04-13 (v2 -- expanded with academic papers, cognitive foundations, and emerging patterns)*

---

## Table of Contents

1. [Cognitive Foundations](#1-cognitive-foundations)
2. [Major Frameworks](#2-major-frameworks)
   - 2.1 MemGPT / Letta
   - 2.2 Mem0
   - 2.3 Zep / Graphiti
   - 2.4 LangGraph / LangMem
   - 2.5 CrewAI
   - 2.6 OpenAI ChatGPT Memory
   - 2.7 Anthropic Claude Code Memory
   - 2.8 Google Gemini Memory
3. [Architectural Patterns](#3-architectural-patterns)
4. [Neurocognitive Approaches](#4-neurocognitive-approaches)
5. [Design Decisions and Trade-offs](#5-design-decisions-and-trade-offs)
6. [Cross-Framework Comparison](#6-cross-framework-comparison)
7. [Cross-Cutting Themes](#7-cross-cutting-themes)
8. [Sources](#8-sources)

---

## 1. Cognitive Foundations: How Memory Maps to Agent Design

The design of memory systems for AI agents draws heavily from cognitive science. The **CoALA** (Cognitive Architectures for Language Agents) framework, proposed by Sumers et al. (2023), establishes the canonical mapping between human cognitive memory and agent architecture [1]. CoALA divides agent memory into **working memory** (active, readily available information for the current decision cycle), and **long-term memory** subdivided into three types:

- **Episodic memory**: records of specific experiences -- decision points, tool calls, failures, outcomes, and reflections from prior runs. These function as concrete precedents and raw material for abstraction.
- **Semantic memory**: abstractions that outlive single episodes -- domain facts, general heuristics, project conventions, and stable world knowledge. Unlike episodic memory, semantic knowledge is not organized around a specific time and place.
- **Procedural memory**: both implicit knowledge (LLM weights) and explicit knowledge (agent code, reusable methods). CoALA notes that updates to procedural memory are risky for both functionality and alignment.

The decision-making cycle in CoALA involves iterative phases: **proposal** (generating candidate actions), **evaluation** (assigning values), and **selection** (choosing the highest-value action). Memory interacts through three internal action types: retrieval (reading from long-term into working memory), reasoning (reading/writing within working memory via the LLM), and learning (writing from working memory into long-term stores). Working memory serves as the central hub connecting different components of a language agent [1].

This tripartite memory taxonomy has become the standard reference point for nearly all subsequent frameworks.

The **Externalization survey** by Zhou et al. (2026) extends CoALA with a fourth category: **personalized memory** -- stable information about particular users, teams, or environments (preferences, habits, recurring constraints). The authors argue this layer should not collapse into general self-improvement stores due to distinct retention and privacy requirements [3].

Zhou et al. further reframe memory externalization as a **representational transformation** (drawing on Norman's cognitive artifact theory): converting an internal recall problem into an external recognition-and-retrieval problem. The model no longer recovers relevant history from parameters but recognizes curated slices that the memory system surfaces. The success criterion becomes not "how much did we save?" but "did we make the current decision legible?" [3].

---

## 2. Major Frameworks and Their Architectures

### 2.1 MemGPT / Letta: Virtual Context Management

**MemGPT** (Packer et al., 2023) introduced the most influential architectural metaphor in the space: treating LLM memory like an operating system's virtual memory hierarchy [2]. The core insight is that an LLM's fixed context window is analogous to physical RAM, and a memory management system can page information in and out to create the illusion of unbounded memory.

#### Three-tier architecture

| Tier | Analogy | Characteristics | Persistence |
|------|---------|-----------------|-------------|
| **Main Context** (working memory) | RAM | Currently in the LLM's context window. Contains system prompt, memory blocks, recent messages. Size-limited by model context. | Ephemeral per request |
| **Core Memory Blocks** | Pinned pages | Structured text blocks always injected into the system prompt. Agent can edit them via tools. | Persistent, self-edited |
| **Archival Memory** | Disk (cold storage) | External vector store for unbounded long-term storage. Agent can insert and search. | Persistent, append-mostly |
| **Recall Memory** | Swap file / message log | Complete conversation history. Agent can search past messages. | Persistent, append-only |

#### Core Memory Blocks

Blocks are Letta's most distinctive abstraction. Each block is a labeled chunk of text (with a character limit) that is always present in the agent's system prompt. The agent reads them on every turn and can edit them via tool calls.

Default block types:
- **`human`** -- stores information about the user (name, preferences, context)
- **`persona`** -- stores the agent's self-description, personality, and role

Blocks are injected into the system prompt at compile time, so the agent sees them as part of its instructions. The key design decision: **the agent modifies its own system prompt** through structured tool calls.

#### Memory Tools (Self-Editing Mechanisms)

| Tool | Operation | Target |
|------|-----------|--------|
| `core_memory_append` | Append text to a memory block | Core memory blocks |
| `core_memory_replace` | Find-and-replace within a block | Core memory blocks |
| `archival_memory_insert` | Store a passage for later retrieval | Archival (vector) store |
| `archival_memory_search` | Semantic search over stored passages | Archival (vector) store |
| `conversation_search` | Search past message history | Recall (message) store |

The agent autonomously decides when to call these tools. The system uses interrupts to manage control flow between itself and the user, directly paralleling OS task management [2].

#### Control Flow

The system operates via nested loops: an **inner loop** where the LLM processes queries and makes memory operations through function calls, and an **outer loop** that manages task execution, context updates, and coordination between memory tiers.

#### Storage Architecture

Internally, Letta uses: **BlockManager** for core memory block persistence (database-backed), **PassageManager** for archival memory (vector store with embeddings), and **MessageManager** for recall/conversation history (relational database). Read-only protection is enforced with validation before persistence.

#### Multi-Agent Memory

Letta supports shared memory blocks across agents. Multiple agents can read from and write to the same block. Agent-to-agent coordination uses message passing (async, sync, and broadcast patterns) rather than direct memory manipulation, keeping memory ownership clear.

#### Key Design Decisions

1. **Agent-as-memory-manager**: The LLM itself decides what to remember, forget, and retrieve.
2. **Always-in-context blocks**: Core memory blocks are always visible, trading context space for guaranteed access.
3. **Character limits on blocks**: Forces selectivity about what stays in core memory vs. archival.
4. **Tool-call interface**: All memory operations go through the same function-calling interface as other tools.

*Sources: [2], [16]*

### 2.2 Mem0: Hybrid Vector + Graph Memory Layer

**Mem0** positions itself as a standalone "memory layer for AI" -- an external service that any application can use. Its architecture combines vector embeddings with a knowledge graph for richer memory representation.

#### Memory Types (Four-Layer Hierarchy)

| Layer | Scope | Lifetime | Use Case |
|-------|-------|----------|----------|
| **Conversation Memory** | Current turn | Seconds | Recent messages for immediate context |
| **Session Memory** | Current task/run | Minutes to hours | Multi-step workflows |
| **User Memory** | Per person/account | Weeks to indefinite | Preferences, facts, domain knowledge |
| **Organizational Memory** | Cross-agent/team | Indefinite | Shared FAQs, policies |

Scoping is controlled by identifiers: `user_id` (per-person), `run_id` (per-session), and `agent_id` (per-agent instance).

#### Processing Pipeline

When memories are added, Mem0 runs a three-stage pipeline:
1. **Information Extraction** -- LLM identifies key facts, decisions, preferences
2. **Conflict Resolution** -- Checks against existing memories for duplicates/contradictions
3. **Storage** -- Persists to vector store + optional graph store

Deduplication only runs when `infer=True` (the default). When inference is disabled, raw messages are stored verbatim.

#### Graph Memory (Hybrid Vector + Graph)

The graph memory system extracts three core components: **nodes** (entities -- people, places, organizations, facts), **edges** (directional relationships between entities with temporal context), and **timestamps** (metadata tracking when relationships were established or changed) [10].

An extraction LLM analyzes conversation payloads to identify these components before persisting them. During search, vector similarity identifies candidate memories while the graph layer runs in parallel, adding related entities in a `relations` array. The system preserves vector search ordering while enriching results with graph context [10].

**Supported graph backends**: Neo4j, Memgraph, Amazon Neptune, Kuzu, and Apache AGE.

Graph memory is particularly valuable when conversation history mixes multiple actors and objects that vectors alone blur together, or when compliance requirements demand auditable provenance.

#### Key Design Decisions

1. **External service model**: Memory is a service, not embedded in the agent.
2. **LLM-powered extraction**: An LLM extracts structured memories from raw conversations.
3. **Hybrid retrieval**: Vector similarity + graph traversal combine for richer recall.
4. **Conflict resolution as first-class**: The pipeline explicitly detects and resolves contradictions.

*Sources: [4], [5], [10]*

### 2.3 Zep / Graphiti: Temporal Knowledge Graphs

**Zep** frames itself as a "context engineering platform." **Graphiti**, its open-source core, is a temporal knowledge graph framework usable independently.

#### Core Architecture

Four primary graph elements:

| Element | Description |
|---------|-------------|
| **Entities (nodes)** | People, products, concepts with evolving summaries |
| **Facts/Relationships (edges)** | Triplets (Entity -> Relationship -> Entity) with temporal validity windows |
| **Episodes** | Raw ingested data serving as provenance for all derived facts |
| **Custom Types** | Developer-defined entity and edge types via Pydantic models |

**Bi-temporal awareness** is the key differentiator. Every fact carries two time dimensions: `valid_at` (when the fact became true) and `invalid_at` (when the fact was superseded). Old facts are never deleted -- they are invalidated. This enables historical queries even after information has changed [11].

**Incremental construction**: new data integrates immediately without full recomputation -- critical for real-time agent operations. **Contradiction handling**: conflicting information automatically invalidates outdated facts while preserving temporal context [11].

#### Hybrid Retrieval

Graphiti combines three retrieval methods: semantic search (embedding-based similarity), keyword search (BM25, no LLM dependency), and graph traversal (relationship-based navigation with reranking by graph distance). This multi-method approach achieves sub-second query latencies without relying on LLM-based summarization bottlenecks [11].

#### Zep Platform

Beyond Graphiti, Zep adds: per-user graph instances, conversation threads that feed into the user's graph, pre-formatted context blocks optimized for LLM injection, and `thread.get_user_context()` for high-level context assembly.

#### Key Design Decisions

1. **Temporal-first**: Bi-temporal fact tracking is the core primitive.
2. **Episode-centric provenance**: Every derived fact traces back to raw source data.
3. **Context assembly, not just retrieval**: Pre-formats context blocks for LLM consumption.
4. **Sub-200ms retrieval latency**: Designed for real-time agent use.

*Sources: [7], [8], [11]*

### 2.4 LangGraph / LangMem: Memory as Graph State

**LangGraph** models agents as state machines (graphs) where nodes are computation steps and edges are transitions. Memory is implemented through two complementary mechanisms [12].

#### Two Memory Scopes

| Scope | Mechanism | Lifetime |
|-------|-----------|----------|
| **Short-term** (thread-scoped) | Checkpointer | Per conversation thread |
| **Long-term** (cross-thread) | Store | Across all threads |

**Checkpointers** persist the entire graph state after each step, enabling thread resumption, time travel (replay or branch from any historical state), and fault tolerance. Available implementations: `MemorySaver` (in-memory), `SqliteSaver`, `PostgresSaver`.

**The Store** provides a namespace-based key-value system with hierarchical organization (namespaces like a filesystem), JSON document storage, semantic search, and content-based filtering. Thread-scoped memory isolates conversations; cross-thread memory uses custom namespaces for shared context [12].

#### Memory Taxonomy

LangGraph explicitly maps patterns to cognitive science: semantic memory (continuously updated profiles in the Store), episodic memory (few-shot examples stored in the Store), and procedural memory (system prompts refined via reflection/meta-prompting) [12].

#### LangMem: Extractive Memory Formation

**LangMem**, the companion library, adds automatic memory extraction rather than raw transcript storage. Two operational modes: **hot path** (active conversation memory management where agents decide what and when to store) and **background** (automatic extraction and consolidation running asynchronously) [13].

LangMem provides `create_manage_memory_tool` (for storing relevant details) and `create_search_memory_tool` (for retrieving semantically similar information). It supports both functional primitives for any storage system and native integration with LangGraph's storage layer [13].

#### Key Design Decisions

1. **Separation of short-term and long-term**: Different abstractions for different lifetimes.
2. **Namespace-based organization**: Hierarchical, filesystem-like memory organization.
3. **Pluggable backends**: Swappable from in-memory to PostgreSQL.
4. **No built-in extraction** in LangGraph itself (LangMem fills this gap).

*Sources: [12], [13]*

### 2.5 CrewAI: Unified Memory for Multi-Agent Teams

CrewAI has undergone a major architectural shift from separate memory types (short-term, long-term, entity, external) to a **single unified `Memory` class** with LLM-powered analysis [14].

#### LLM-Driven Organization

During storage, an LLM analyzes content and the existing scope tree, suggesting optimal placement. Memories organize into a filesystem-like tree (`/project/alpha`, `/agent/researcher`, etc.) [14].

#### Composite Scoring

Retrieval blends three signals:
- **Semantic similarity** (weight 0.5): `1 / (1 + distance)` from vector index
- **Recency** (weight 0.3): exponential decay `0.5^(age_days / half_life_days)`
- **Importance** (weight 0.2): LLM-assigned importance score (0-1)

Configurable per use case: sprint retrospectives benefit from high recency weight; architecture knowledge bases from high importance weight [14].

#### Adaptive Depth Retrieval

| Depth | Mechanism | Latency | LLM Calls |
|-------|-----------|---------|-----------|
| **Shallow** | Direct vector search with composite scoring | ~200ms | 0 |
| **Deep** | Multi-step RecallFlow with query analysis and scope selection | Higher | 1+ |

Queries under 200 characters bypass LLM analysis entirely [14].

#### Multi-Agent Sharing

When `memory=True` is enabled, all agents share the crew's unified memory instance. Agents can optionally receive scoped views -- a researcher maintains private findings via `memory.scope("/agent/researcher")` while writers access shared knowledge. After each task, the crew automatically extracts discrete facts from the task output; before each task, agents recall relevant context [14].

#### Consolidation and Degradation

Similarity threshold (default 0.85) triggers LLM-based decision to keep, update, delete, or insert new records. If LLM analysis fails, memory degrades gracefully: stores content with default scope `/`, empty categories, and mid-range importance (0.5) -- no data loss [14].

*Source: [14]*

### 2.6 OpenAI ChatGPT Memory

OpenAI's ChatGPT memory is the most widely deployed agentic memory system. The model autonomously decides what to remember during conversations, extracting discrete factual statements (not verbatim text). Users can also explicitly request storage. Memories persist across all conversations and are injected into the system prompt at each session start.

User controls include viewing, editing, and deleting individual memories; toggling memory globally; and "Temporary Chat" mode that disables memory entirely. Memories are per-user, not per-conversation.

The architectural details are not publicly documented beyond user-facing behavior. The system likely uses extraction (identifying salient facts), embedding-based retrieval (finding relevant memories), and injection (prepending to system context) as "memory tokens" that consume context window space.

As of early 2026, OpenAI has not exposed persistent memory through the developer API. The Responses API provides conversation state management via `previous_response_id` chaining (server-side state), but this is session-scoped, not persistent long-term memory.

*Source: OpenAI product announcements and help documentation (2024).*

### 2.7 Anthropic Claude Code: File-Based Memory

Claude Code implements a distinctive **file-based memory architecture** that is transparent, version-controllable, and human-editable. Rather than using databases or vector stores, memory lives entirely in markdown files on the filesystem [15].

#### Two Complementary Memory Systems

| | CLAUDE.md files | Auto memory |
|---|---|---|
| **Who writes it** | Human | Claude |
| **What it contains** | Instructions and rules | Learnings and patterns |
| **Scope** | Project, user, or org | Per working tree |
| **Loaded into** | Every session (in full) | Every session (first 200 lines or 25KB) |
| **Use for** | Coding standards, workflows, architecture | Build commands, debugging insights, preferences |

**CLAUDE.md** files can be scoped by location:
- Home folder (`~/.claude/CLAUDE.md`): applies to all sessions
- Project root (`./CLAUDE.md`): shared via version control with teams
- Local overrides (`./CLAUDE.local.md`): personal, gitignored
- Subdirectory files: loaded on demand when working in those directories
- Organization-wide: managed policy files deployed by IT
- Path-scoped rules (`.claude/rules/`): conditional loading via `paths` frontmatter in YAML, only applying when Claude works with matching files [15]

**Auto memory** is stored at `~/.claude/projects/<project>/memory/`. A `MEMORY.md` index links to topic-specific files. Claude decides what is worth remembering based on whether the information would be useful in a future conversation. Topic files load on demand, not at startup [15].

**Memory loading hierarchy**: files are discovered by walking up the directory tree from the working directory. All discovered files are concatenated (not overriding). `CLAUDE.local.md` appends after `CLAUDE.md` at each level. CLAUDE.md content survives compaction (re-read from disk).

#### Memory Tool API

At the API level, Anthropic also provides a **memory tool** (`type: memory_20250818`) that exposes file operations: `view`, `create`, `str_replace`, `insert`, `delete`, `rename`. Claude automatically follows a protocol: always view the memory directory first, check for earlier progress, work on the task while recording progress, and assume interruption could happen at any time.

#### Key Design Decisions

1. **File-based metaphor**: Memory as files and directories, not databases. Maximizes transparency, debuggability, and integration with developer workflows (git, text editors, code review).
2. **Context, not configuration**: CLAUDE.md is loaded as context, not enforced. Adherence depends on conciseness and specificity.
3. **Client-side ownership**: The developer owns the storage infrastructure. Anthropic stores nothing.
4. **No extraction pipeline**: Unlike Mem0 or Zep, there is no automatic fact extraction. Claude writes what it deems important.
5. **Composable with compaction**: Memory handles cross-session persistence; compaction handles within-session context management.

*Source: [15]*

### 2.8 Google Gemini Memory

Google Gemini introduced a memory feature in late 2024 as part of the Gemini 2.0 update. The system allows Gemini to remember facts and preferences across conversations. Users can ask Gemini to remember something, view saved memories, and delete them. Observable behavior suggests a pattern similar to ChatGPT's: extraction of salient facts, persistent storage, and relevance-based injection into new conversations. Architectural specifics have not been publicly documented in detail.

*Source: Google product announcements (December 2024). Limited technical detail publicly available.*

---

## 3. Architectural Patterns

### 3.1 Four Evolutionary Paradigms

The Externalization survey by Zhou et al. (2026) traces a clear evolutionary arc [3]:

**Paradigm 1: Monolithic Context** -- All relevant history remains directly in the prompt. Transparent but scales poorly; history disappears when sessions end.

**Paradigm 2: Context with Retrieval Storage** -- Near-term working state stays in context; longer-horizon traces are externally stored and retrieved on demand. This pattern underlies most practical memory systems in production copilots, assistants, and coding agents. Variants include GraphRAG (graph structure), ENGRAM (latent compression), and SYNAPSE (spreading activation) [3].

**Paradigm 3: Hierarchical Memory and Orchestration** -- Explicit lifecycle operations for extraction, consolidation, and forgetting. Two tendencies emerge:
- *Spatio-temporal decoupling*: systems like MemGPT and MemoryOS separate hot working state from cold storage, swapping across tiers as task demands change.
- *Semantic decoupling*: systems like MemoryBank and MIRIX organize memory by function (events, user profiles, world knowledge) rather than flat retrieval [3].

**Paradigm 4: Adaptive Memory Systems** -- Two expanding directions: dynamic modules (MemEvolve decomposes lifecycle into separate encode, store, retrieve, and manage modules that evolve during execution) and feedback-based optimization (MemRL learns better control policies through reinforcement learning or mixture-of-experts gating) [3].

**The critical insight**: the major transition is **from storage to control**. Memory ceases being a passive appendix and becomes part of harness control that determines what past context the model can effectively act upon [3].

### 3.2 Memory-as-a-Tool Pattern

The memory-as-a-tool pattern exposes memory operations as callable functions within the agent's tool set, letting the agent reason about what to store and retrieve. Gallego (2026) presents a particularly clean implementation [6]:

- Memory is exposed through **file-system operations**: `ls(path="/memories/")` to list available files, `read_file(path)` to retrieve, and `write_file`/`edit_file` to persist.
- The agent performs **two-step retrieval**: first listing available files, then selectively reading relevant ones based on meaningful filenames -- enabling active reasoning about what to retrieve rather than relying on opaque embeddings.
- **Distillation** transforms raw feedback into generalizable principles. Specific critiques convert to semantic rules (abstraction), and the model resolves conflicts between old and new knowledge (conflict resolution).
- Memory stores structured guidelines rather than raw logs -- maintaining high-signal "lessons learned" journals [6].

Experimental results across Claude Sonnet, GPT-5.1, and Gemini 3 Pro: memory-augmented approaches matched self-critique performance after just two feedback rounds while substantially reducing inference costs. On long-horizon mixed tasks (H=12), memory agents achieved 0.78 +/- 0.10 scores versus 0.52 +/- 0.25 baseline [6].

This pattern is also the approach used by Claude Code's file-based memory.

### 3.3 Hierarchical Memory (L1-L4)

The hierarchical pattern organizes memory into tiers with different access speeds, capacities, and retention policies:

| Tier | Name | Characteristics | Persistence | Access Pattern |
|------|------|----------------|-------------|----------------|
| L1 | Working Memory / Scratchpad | Active context, chain-of-thought traces, current task state | Session-scoped | Direct (in-context) |
| L2 | Short-term / Recall | Recent conversation history, recent tool outputs | Session or multi-session | Fast retrieval |
| L3 | Long-term / Semantic | User preferences, project conventions, learned facts | Persistent | Search-based |
| L4 | Archival | Raw transcripts, historical data, audit trails | Persistent | Batch/search |

MemGPT implements L1-L3-L4. CraniMem implements L1-L2-L3 (working context, episodic buffer, knowledge graph). The **store routing problem** across tiers is non-trivial: Gaikwad (2026) finds that oracle routing achieves 86.7% accuracy while using 62% fewer context tokens (299 vs. 787) than uniform retrieval across all stores. Irrelevant context actively hinders the model's ability to identify answer-bearing information [7a]. A simple fixed policy retrieving STM+Summary+LTM approaches oracle-level accuracy while maintaining moderate cost [7a].

### 3.4 Read-Write Memory (Explicit Store/Retrieve)

The agent has explicit tools for storing and retrieving memories, and must reason about when to use them:

- **MemGPT/Letta**: function calls for memory insertion and search
- **Mem0**: CRUD operations (add/search/update/delete)
- **LangMem**: `create_manage_memory_tool` and `create_search_memory_tool`
- **Claude Code**: file read/write operations on MEMORY.md and topic files

The advantage is transparency and controllability. The disadvantage is that the agent must learn when to write -- leading to either excessive storage (filling memory with noise) or insufficient storage (failing to capture important information).

### 3.5 Reflection-Based Memory

Reflection-based memory involves the agent reviewing and consolidating its own memories:

- **Reflexion** (Shinn et al., 2023): stores reflective summaries from failed attempts as reusable experience, enabling iterative learning [3].
- **CraniMem's consolidation loop**: periodically replays high-utility traces into the knowledge graph while pruning low-utility items [9].
- **CrewAI's automatic consolidation**: the LLM decides whether to keep, update, delete, or insert new records when similarity exceeds thresholds [14].

The consolidation process typically involves: (1) scoring existing memories for utility, (2) identifying redundancies and conflicts, (3) abstracting specific instances into general principles, and (4) pruning low-value entries to maintain capacity bounds.

### 3.6 Scratchpad / Working Memory

Working memory operates within a single session as disposable short-term context. Nowaczyk (2025) describes it as scratchpads and chain-of-thought traces that improve reasoning decomposition but should not persist to durable knowledge [17]. This separation is critical: speculative reasoning in working memory must not contaminate long-term stores.

LangGraph implements this cleanly through thread-scoped state isolated from the persistent Store. Claude Code's conversation context serves the same function, with `/clear` explicitly resetting it between unrelated tasks. The Externalization survey emphasizes that working memory changes rapidly and loses value if stale [3].

### 3.7 Episodic vs. Semantic vs. Procedural

The three memory types from cognitive science map to distinct implementation patterns:

| Memory Type | Content | Implementation Examples |
|-------------|---------|------------------------|
| **Episodic** | What happened -- decision points, tool calls, outcomes | Letta recall memory, Zep episodes, CraniMem episodic buffer |
| **Semantic** | What is true -- domain facts, user preferences, project conventions | Letta core blocks, Mem0 user memory, Claude CLAUDE.md |
| **Procedural** | How to do things -- reusable methods, workflows | Claude Code skills, LangGraph procedural prompts, LLM weights |

The Externalization survey notes a critical distinction: episodic memory provides raw material for abstraction, while semantic memory represents the abstractions themselves. The graduation from episodic to semantic to procedural mirrors a learning progression [3].

---

## 4. Neurocognitive Approaches

### 4.1 CraniMem: Gated and Bounded Memory

**CraniMem** (Mody et al., 2026) brings neurocognitive principles directly into agent architecture [9]:

**Attentional gating** suppresses low-salience inputs before they reach memory, analogous to brainstem and thalamic control systems. Input gating uses semantic similarity between incoming messages and agent goals; if similarity falls below a noise threshold (tau_noise), content is discarded [9].

**Three interconnected components**:
- **Episodic Buffer**: A bounded FIFO store holding the most recent N turns, supporting high-fidelity short-range continuity and providing candidates for consolidation.
- **Knowledge Graph**: Structured long-term storage for user preferences, constraints, and stable plans, enabling consistent multi-hop retrieval.
- **Consolidation Loop**: Periodic optimization that replays high-utility traces into the graph while pruning low-utility items [9].

**Utility scoring** combines three LLM-generated signals: importance (I), surprise (S), and emotion (E), aggregated as `BaseUtility = 1/3(I + S + E)` [9].

**Capacity and overflow**: the episodic buffer maintains strict FIFO size limits. Consolidation applies selective filtering: only items with ReplayScore above a consolidation threshold transfer to long-term memory. Low-utility traces are discarded to keep memory bounded [9].

**Experimental results** on HotpotQA with noise injection:

| Metric | CraniMem | Mem0 | Vanilla RAG |
|--------|----------|------|------------|
| F1 (Clean) | 0.323 | 0.234 | 0.095 |
| Noise Drop | 0.011 | 0.036 | 0.027 |

CraniMem achieved smaller noise drop across four LLM backbones, demonstrating superior robustness at the cost of increased consolidation overhead [9].

### 4.2 Human-Like Memory Recall and Consolidation

Hou et al. (2024) model memory strength using a modified sigmoid function [18]:

`p_n(t) = [1 - exp(-r * e^(-t/g_n))] / [1 - e^(-1)]`

Where r = relevance (cosine similarity), t = elapsed time, and g_n = consolidation factor increasing with recall frequency.

Key human-like aspects:
- **Cued recall**: contextual triggers rather than automatic retrieval
- **Consolidation through time**: memories never completely decay; appropriate triggers enable recall of distant events
- **Spaced repetition effects**: extended intervals between recalls strengthen memories more than frequent short-term recalls [18]

Testing against Generative Agents baseline showed statistically significant improvements (t-value: -5.687, p-value: 0.000299). The system struggles with novel contexts deviating from established patterns, as it prioritizes long-term behavioral patterns over situational changes [18].

---

## 5. Design Decisions and Trade-offs

### 5.1 Explicit vs. Implicit Memory Formation

| Approach | How it works | Examples | Pros | Cons |
|----------|-------------|----------|------|------|
| **Explicit** | Agent decides what to save via tool calls | MemGPT, Mem0, LangMem, Claude Code auto memory | Transparent, controllable, auditable | Agent may miss important info or over-store |
| **Implicit** | System automatically extracts memories | OpenAI ChatGPT, CrewAI's auto-extraction | Lower cognitive overhead on agent | Less transparent, harder to debug |
| **Hybrid** | Automatic extraction with agent override | CrewAI (auto-extract + manual), Zep (auto-ingest + explicit search) | Balances convenience and control | More complex implementation |

The trend is toward hybrid approaches. CrewAI's design is instructive: after each task, the crew automatically extracts facts (implicit), but agents can manually store information (explicit). The LLM-driven consolidation layer manages conflicts and duplicates [14].

### 5.2 Centralized vs. Distributed Memory

| Approach | Architecture | Examples | Best for |
|----------|-------------|----------|----------|
| **Centralized** | Single store shared by all agents | CrewAI shared crew memory, Zep per-user graph | Teams needing consistent shared context |
| **Distributed** | Per-agent memory stores | LangGraph thread-scoped state | Privacy, isolation, independent agents |
| **Federated** | Shared + private scopes | CrewAI scoped views, LangGraph namespaces | Multi-agent teams with role-specific knowledge |

### 5.3 Context Window Management Strategies

Context window saturation is the fundamental constraint. Strategies include:

1. **Eviction/paging** (MemGPT): move less-relevant content to lower tiers when context fills.
2. **Compaction/summarization** (Claude Code, LangGraph): compress conversation history while preserving key decisions and file states.
3. **Selective retrieval** (Mem0, Zep): only inject relevant memories rather than full history.
4. **Gating** (CraniMem): prevent low-salience inputs from entering memory at all [9].
5. **Subagent delegation** (Claude Code): offload research to separate context windows, reporting back summaries [15].
6. **Cost-sensitive routing** (Gaikwad, 2026): optimize which memory stores to query to maximize accuracy per context token spent [7a].

Gaikwad's routing study provides quantitative evidence: uniform retrieval across all stores (787 context tokens) achieves only 81.3% accuracy, while oracle routing (299 tokens, 62% fewer) achieves 86.7%. Despite 94% routing coverage, hybrid heuristics achieve only 70% downstream accuracy versus 86.7% for oracle routing -- selecting correct stores does not guarantee successful answer extraction [7a].

### 5.4 Memory Capacity and Overflow Handling

| Framework | Capacity Mechanism | Overflow Handling |
|-----------|-------------------|-------------------|
| MemGPT | Fixed main context; unbounded archival | Paging/eviction from main context |
| CraniMem | Bounded FIFO episodic buffer | FIFO eviction + selective consolidation + input gating |
| CrewAI | Configurable consolidation threshold | Deduplication (0.85), batch dedup (0.98) |
| Claude Code | 200 lines / 25KB auto-memory limit | Topic files for overflow; lazy loading |
| Zep/Graphiti | Graph growth | Temporal invalidation (facts, not deletion) |
| LangGraph | Configurable store | Compaction via summarization |

CraniMem's approach is the most principled: it combines FIFO eviction (hard bound) with selective consolidation (only high-utility items transfer) and input gating (preventing noise from entering). This mirrors the biological principle that memory systems must forget most inputs to function effectively [9].

### 5.5 Storage Backend Choices

| Backend Type | Examples | Strengths | Weaknesses |
|-------------|----------|-----------|------------|
| **Vector databases** | Mem0, CrewAI, LangMem | Semantic similarity search | Lose structural relationships |
| **Knowledge graphs** | Zep/Graphiti, CraniMem, Mem0 graph | Entity relationships, multi-hop reasoning | Complexity, maintenance overhead |
| **Relational databases** | LangGraph checkpoints | Transactional guarantees, structured queries | No semantic search without extensions |
| **File systems** | Claude Code, Memory-as-a-Tool | Transparency, human editability, git integration | No sophisticated search |
| **Hybrid** | Mem0, Zep | Complementary retrieval | Dual storage path complexity |

The Memory-as-a-Tool paper provides an important counterpoint: file-system-based memory with meaningful filenames and LLM-driven retrieval reasoning achieved competitive performance with substantially lower infrastructure complexity [6].

### 5.6 Memory Reliability Safeguards

Nowaczyk (2025) identifies requirements for reliable memory [17]:

- **Provenance tracking**: source identifiers, content hashes, timestamps, retrieval policy versions
- **Freshness policies**: TTL limits and refresh-on-access procedures by source type
- **Hygiene defenses**: sanitization of untrusted content with strict allow/deny lists
- **Compaction strategies**: layered retention with lossless cold storage, summarized mid-term, and hot caches
- **Separation**: isolated working memory from long-term stores to prevent speculative reasoning contamination

---

## 6. Cross-Framework Comparison

### Memory Ownership Model

| Framework | Who manages memory? | Where is it stored? |
|-----------|-------------------|---------------------|
| **Letta/MemGPT** | The agent (via tool calls) | Server-side (Letta platform) |
| **Mem0** | External service (with LLM extraction) | Mem0 Cloud or self-hosted |
| **Zep/Graphiti** | External service (automatic extraction) | Zep Cloud / Neo4j |
| **LangGraph** | Developer or agent (via tools) | Developer-chosen backend |
| **CrewAI** | LLM-driven unified system | LanceDB (local default) |
| **OpenAI ChatGPT** | The model (autonomous) | OpenAI servers |
| **Claude Code** | The agent (via file I/O) | Client-side filesystem |

### Retrieval Capabilities

| Framework | Vector Search | Graph Traversal | Keyword/BM25 | Temporal Queries |
|-----------|:---:|:---:|:---:|:---:|
| **Letta** | Yes (archival) | No | No | No |
| **Mem0** | Yes | Yes | No | Timestamps |
| **Zep/Graphiti** | Yes | Yes | Yes | Yes (bi-temporal) |
| **LangGraph** | Yes (store) | No | No | No |
| **CrewAI** | Yes | No | No | Yes (recency decay) |
| **OpenAI** | Unknown | Unknown | Unknown | No |
| **Claude Code** | No (file-based) | No | No | No |

### Comprehensive Comparison

| Framework | Memory Model | Storage | Formation | Multi-Agent | Temporal | Key Innovation |
|-----------|-------------|---------|-----------|-------------|----------|----------------|
| **MemGPT/Letta** | 3-tier virtual context | Vector DB + relational | Explicit (tool calls) | Shared blocks | No | OS-inspired paging |
| **Mem0** | Vector + Graph hybrid | Vector DB + Neo4j/etc. | Explicit (CRUD API) | Scoped IDs | Timestamps | Entity graph enrichment |
| **Zep/Graphiti** | Temporal knowledge graph | Neo4j + embeddings | Auto-ingest | Per-user graphs | Bi-temporal | Fact invalidation |
| **LangGraph** | State + Store | DB-backed stores | Hybrid (hot/background) | Namespaces | Checkpoints | Graph-native state |
| **LangMem** | Extractive + Store | LangGraph Store | Hybrid | Via LangGraph | No | Memory manager patterns |
| **CrewAI** | Unified scored memory | LanceDB | Hybrid (auto + manual) | Shared + scoped | Recency decay | Composite scoring |
| **ChatGPT** | Extracted facts | Proprietary | Implicit + user request | No | No | Zero-config simplicity |
| **Claude Code** | File-based markdown | Filesystem | Hybrid (manual + auto) | No | No | Git-native transparency |
| **CraniMem** | Gated episodic + KG | FIFO buffer + KG | Gated (utility scoring) | No | Consolidation | Neurocognitive gating |

---

## 7. Cross-Cutting Themes

### Theme 1: From Storage to Control

The most significant trend is the shift from memory as passive storage to memory as active control infrastructure. Zhou et al. frame this as the central transition: memory determines what past context the model can effectively act upon [3].

### Theme 2: Agent-Managed vs. Service-Managed Memory

| Approach | Advantages | Disadvantages |
|----------|-----------|---------------|
| **Agent-managed** (Letta, Claude) | Contextual decisions about storage; no dependency on external extraction quality | Consumes tokens for memory management; may forget to save |
| **Service-managed** (Mem0, Zep) | Automatic extraction without agent overhead; consistent quality | May extract irrelevant information; adds infrastructure complexity |

### Theme 3: Vector-Only vs. Hybrid

Pure vector search is the default for most frameworks. Mem0 and Zep/Graphiti add graph storage. **When graph matters**: queries requiring relationship understanding, temporal evolution, or multi-hop reasoning. **When vector suffices**: queries that are essentially "find things similar to this."

### Theme 4: Temporal Awareness

Only Zep/Graphiti (bi-temporal validity windows) and CrewAI (recency decay) have explicit temporal models. Most frameworks treat memories as timeless facts. CraniMem's consolidation loop adds a weak temporal signal through utility decay.

### Theme 5: Consolidation and Deduplication

| Framework | Strategy |
|-----------|----------|
| **Mem0** | LLM-based conflict resolution during `add` pipeline |
| **CrewAI** | Similarity threshold (0.85) + LLM decision |
| **Zep** | Fact invalidation (old facts marked with `invalid_at`) |
| **CraniMem** | Utility-based replay scoring + pruning |
| **Letta** | Agent's responsibility via `core_memory_replace` |
| **LangGraph, Claude Code** | Developer's responsibility |
| **OpenAI** | Opaque (handled internally) |

### Theme 6: Graceful Degradation

Production systems handle LLM failures without breaking the primary workflow:
- **CrewAI**: Falls back to default scope and vector-only search
- **Claude Code**: Memory is optional; agent functions without it
- **Letta**: Memory tools are optional; agent can respond without archival/recall

### Theme 7: Developer Control vs. Convenience

A clear spectrum:

```
Most Automatic                                              Most Explicit
    |                                                            |
    OpenAI     Zep      Mem0     CrewAI    Letta    Claude   LangGraph
    ChatGPT
```

Consumer products favor automation; enterprise agents favor control and auditability; developer tools favor flexibility.

### Theme 8: Memory Couples with Other Agent Components

The Externalization survey highlights that memory does not exist in isolation [3]:
- **With Skills**: Memory stores execution evidence (traces, outcomes, failures); skills represent promotion of evidence into explicit reusable procedures.
- **With Protocols**: Tool results and state transitions arrive through protocolized interfaces but become memory once normalized into persistent state.
- **With Harness**: The survey advocates treating memory as managed state infrastructure with read/write permissions, conflict resolution, and access quotas.

---

## 8. Sources

### Research Papers

[1] Sumers, T.R., Yao, S., Narasimhan, K., and Griffiths, T.L. "Cognitive Architectures for Language Agents." arXiv:2309.02427 (2023). https://arxiv.org/abs/2309.02427

[2] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S.G., Stoica, I., and Gonzalez, J.E. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560 (2023). https://arxiv.org/abs/2310.08560

[3] Zhou, C., Chai, H., Chen, W., et al. "Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering." arXiv:2604.08224 (2026). https://arxiv.org/abs/2604.08224

[6] Gallego, V. "Distilling Feedback into Memory-as-a-Tool." arXiv:2601.05960 (2026). https://arxiv.org/abs/2601.05960

[7a] Gaikwad, M. "Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents." arXiv:2603.15658 (2026). https://arxiv.org/abs/2603.15658

[9] Mody, P., Panchal, M., Kar, R., Bhowmick, K., and Karani, R. "CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems." arXiv:2603.15642 (2026). https://arxiv.org/abs/2603.15642

[17] Nowaczyk, S. "Architectures for Building Agentic AI." arXiv:2512.09458 (2025). https://arxiv.org/abs/2512.09458

[18] Hou, Y., Tamoto, H., and Miyashita, H. "My agent understands me better: Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents." arXiv:2404.00573 (2024). https://arxiv.org/abs/2404.00573

[19] Plaat, A., van Duijn, M., van Stein, N., Preuss, M., van der Putten, P., et al. "Agentic Large Language Models, a survey." arXiv:2503.23037 (2025). https://arxiv.org/abs/2503.23037

[20] Wang, L., Ma, C., Feng, X., et al. "A survey on large language model based autonomous agents." Frontiers of Computer Science (2024). DOI: 10.1007/s11704-024-40231-1

[21] Yu, Y., Li, H., Chen, Z., et al. "FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design." AAAI Symposium Series (2024). DOI: 10.1609/aaaiss.v3i1.31290

[22] Honda, Y., Fujita, Y., Zempo, K., and Fukushima, S. "Human-Like Remembering and Forgetting in LLM Agents: An ACT-R-Inspired Memory Architecture." ACM (2025). DOI: 10.1145/3765766.3765803

### Framework Documentation

[4] Mem0 GitHub: https://github.com/mem0ai/mem0

[5] Mem0 Documentation: https://docs.mem0.ai

[7] Zep GitHub: https://github.com/getzep/zep

[8] Graphiti GitHub: https://github.com/getzep/graphiti

[10] Mem0 Graph Memory: https://docs.mem0.ai/features/graph-memory

[11] Zep / Graphiti Documentation: https://help.getzep.com/concepts; https://github.com/getzep/graphiti

[12] LangGraph Memory Docs: https://docs.langchain.com/oss/python/langgraph/memory

[13] LangMem Documentation: https://langchain-ai.github.io/langmem/

[14] CrewAI Memory Docs: https://docs.crewai.com/concepts/memory

[15] Claude Code Documentation: https://code.claude.com/docs/en/memory; https://code.claude.com/docs/en/best-practices

[16] Letta Documentation: https://docs.letta.com; https://github.com/letta-ai/letta
