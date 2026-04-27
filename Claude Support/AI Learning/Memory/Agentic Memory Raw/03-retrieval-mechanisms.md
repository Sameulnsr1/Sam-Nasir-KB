# Retrieval Mechanisms for Agentic Memory Systems

**Research Chunk 03** | Compiled 2026-04-13 | Revised with source code analysis and implementation details

How agents find and surface relevant memories at the right time is arguably the most consequential design decision in any memory-augmented system. A memory store is only as useful as its retrieval layer. This document surveys seven retrieval strategies -- from vector similarity search to graph traversal to meta-cognitive self-querying -- with emphasis on exact implementations, source code, formulas, and benchmark results from 2023-2026.

---

## 1. Semantic Similarity Search (Baseline Vector Retrieval)

### 1.1 How It Works

The foundation of most modern memory retrieval: encode text into dense vector embeddings, store them in a vector database, and at query time encode the query into the same embedding space and retrieve the nearest neighbors by cosine similarity (or dot product / Euclidean distance).

```
similarity(q, d) = cos(q, d) = dot(q, d) / (||q|| * ||d||)
```

This is operationally a Maximum Inner Product Search (MIPS) or approximate nearest neighbor (ANN) problem using algorithms like HNSW, FAISS, ScaNN, ANNOY, or LSH. HNSW is the dominant choice in production vector databases (Pinecone, Weaviate, pgvector) due to its balance of speed and recall.

Source: https://lilianweng.github.io/posts/2023-06-23-agent/

### 1.2 Embedding Models (2023-2026 Landscape)

#### OpenAI text-embedding-3 (January 2024)

Two variants with native dimension reduction via Matryoshka Representation Learning:

| Model | Max Dimensions | Shortening Support | Max Input |
|-------|---------------|-------------------|-----------|
| text-embedding-3-small | 1,536 | Yes (e.g., 256, 512) | 8,191 tokens |
| text-embedding-3-large | 3,072 | Yes (e.g., 256, 512, 1024) | 8,191 tokens |

Key insight: text-embedding-3-large at 256 dimensions outperforms ada-002 at 1,536 dimensions on MTEB. A Galileo study on Nvidia 10-K financial documents found text-embedding-3-small achieved a 7% attribution improvement over all-MiniLM-L6-v2 at 384 dimensions, while text-embedding-3-large at the same dimension performed nearly identically -- suggesting diminishing returns beyond a quality threshold. Their recommendation: "save money while maintaining performance" by using the smaller model.

Source: https://www.galileo.ai/blog/mastering-rag-how-to-select-an-embedding-model

#### Cohere Embed v3 / v4 (November 2023 / 2025)

| Model | Dimensions | Notes |
|-------|-----------|-------|
| embed-english-v3.0 | 1,024 | English-optimized |
| embed-multilingual-v3.0 | 1,024 | 100+ languages |
| Embed v4.0 (pro/fast) | 256-1,536 (configurable via `output_dimension`) | Multimodal (images + text) |

Distinctive features:

- **Input type parameter** (`search_document`, `search_query`, `classification`, `clustering`, `image`) -- asymmetric embeddings where the same text produces different vectors based on intended use. This is critical for memory systems where the storage embedding and the query embedding should be optimized differently.
- **Compression types**: `float`, `int8`, `uint8`, `binary`, `ubinary`, `base64`. Binary quantization reduces storage to 1/8 the float length with minimal recall loss.
- **Max 96 texts per API call**, with truncation options: `NONE` (error), `START` (discard beginning), `END` (discard ending, default).

```python
# Cohere Embed v2 API
import cohere
co = cohere.Client(api_key)

# Storage: embed as search_document
doc_embeddings = co.embed(
    texts=["memory content here"],
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["float", "binary"]  # get both for hybrid indexing
).embeddings

# Query: embed as search_query
query_embedding = co.embed(
    texts=["what did the user say about X?"],
    model="embed-english-v3.0",
    input_type="search_query",
    embedding_types=["float"]
).embeddings
```

Source: https://docs.cohere.com/v2/reference/embed

#### BGE-M3 (BAAI, February 2024)

The most versatile open-source embedding model, supporting three retrieval modes simultaneously from a single model:

| Feature | Specification |
|---------|--------------|
| Dimensions | 1,024 |
| Max Sequence Length | 8,192 tokens |
| Base Architecture | XLM-RoBERTa (extended) |
| Languages | 100+ |
| Downloads | 15.5M+/month on HuggingFace |

**Three retrieval modes in one model**:
1. **Dense retrieval**: Standard single-vector per text
2. **Sparse retrieval**: Learned token-level lexical weights (like SPLADE), competitive with BM25
3. **Multi-vector retrieval (ColBERT)**: Multiple vectors per document for fine-grained token-level matching

```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences_1 = ["What is BGE M3?", "Definition of BM25"]
sentences_2 = ["BGE M3 is an embedding model...", "BM25 is a bag-of-words retrieval function..."]

# Dense embeddings
embeddings_1 = model.encode(sentences_1, max_length=8192)['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
# Output: [[0.6265, 0.3477], [0.3499, 0.6780]]

# Sparse retrieval (lexical weights)
output_1 = model.encode(sentences_1, return_sparse=True)
print(model.convert_id_to_token(output_1['lexical_weights']))
# [{'What': 0.08356, 'is': 0.0814, 'BGE': 0.252, ...}]

# Multi-vector (ColBERT) retrieval
output = model.encode(sentences_1, return_colbert_vecs=True)
colbert_score = model.colbert_score(output['colbert_vecs'][0], output['colbert_vecs'][1])

# Combined score with configurable weights [dense, sparse, colbert]
scores = model.compute_score(sentence_pairs, weights_for_different_modes=[0.4, 0.2, 0.4])
```

Recommended pipeline: BGE-M3 hybrid retrieval (dense + sparse) + BGE-reranker-v2-m3 cross-encoder reranking.

Source: https://huggingface.co/BAAI/bge-m3

#### SPECTER2 (Allen AI, 2023)

Specialized for scientific documents. Uses modular adapters on SciBERT:

| Adapter | Task | Use Case |
|---------|------|----------|
| `allenai/specter2` | Proximity/Retrieval | Link prediction, nearest neighbor search |
| `allenai/specter2_adhoc_query` | Ad-hoc Search | Short text queries |
| `allenai/specter2_classification` | Classification | Feed embeddings into classifiers |
| `allenai/specter2_regression` | Regression | Feed embeddings into regressors |

Specifications: 768 dimensions, 512 max tokens, trained on 6M+ citation triplets from SciRepEval.

**Benchmarks** (vs. predecessors):

| Model | SciRepEval In-Train | SciRepEval Out-of-Train | Avg | MDCR (MAP, Recall@5) |
|-------|-------------------|----------------------|-----|----------------------|
| BM-25 | -- | -- | -- | (33.7, 28.5) |
| SPECTER | 54.7 | 72.0 | 67.5 | (30.6, 25.5) |
| SciNCL | 55.6 | 73.4 | 68.8 | (32.6, 27.3) |
| **SPECTER2** | **62.3** | **74.1** | **71.1** | **(38.4, 33.0)** |

```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

text = paper['title'] + tokenizer.sep_token + paper.get('abstract', '')
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
embedding = model(**inputs).last_hidden_state[:, 0, :]  # CLS token, shape [batch, 768]
```

Source: https://huggingface.co/allenai/specter2

### 1.3 Limitations of Pure Semantic Similarity

1. **Vocabulary mismatch**: Semantic search misses exact keyword matches. A query for "CUDA error 11" may not retrieve a document about "device-side assert triggered."
2. **No temporal awareness**: A memory from 5 minutes ago and 5 months ago are treated equally if equidistant in embedding space.
3. **No importance weighting**: Trivial observations can outrank critical insights if semantically closer to the query.
4. **Information loss in compression**: Bi-encoders compress all meaning into a single vector. As Pinecone notes, this means the encoder "must compress all of the possible meanings of a document into a single vector."
5. **Domain sensitivity**: Models trained on web text underperform on domain-specific content without fine-tuning.
6. **Context window blindness**: Standard chunking loses document-level context -- a chunk about "the committee's decision" retrieves poorly without knowing which committee. (Addressed by Anthropic's Contextual Retrieval -- see Section 6.)

Source: https://www.pinecone.io/learn/series/rag/rerankers/

---

## 2. Temporal Decay / Recency Weighting

### 2.1 How It Works

Temporal decay ensures recent memories are preferred over older ones, mimicking human memory's recency bias. The mechanism applies a monotonically decreasing function to reduce a memory's retrieval score based on age or position in a chronological ordering.

### 2.2 The Generative Agents Approach (Park et al. 2023) -- Actual Implementation

The reference implementation from the Generative Agents source code reveals that recency is computed by **chronological position** rather than absolute time:

```python
# From generative_agents/reverie/backend_server/persona/cognitive_modules/retrieve.py

# Nodes are ordered chronologically. Position 1 = most recent.
recency_vals = [persona.scratch.recency_decay ** i
                for i in range(1, len(nodes) + 1)]

# recency_decay is a configurable per-persona parameter, typically 0.995
# Position 1 (most recent):  0.995^1   = 0.995
# Position 10:               0.995^10  = 0.951
# Position 100:              0.995^100 = 0.606
# Position 1000:             0.995^1000 = 0.0067

# All recency values are then min-max normalized to [0, 1]
recency_out = normalize_dict_floats(recency_out, 0, 1)
```

This is a subtle but important design choice: the 10th most recent memory always gets the same recency score regardless of whether it was created 1 hour or 1 week ago. What matters is **rank position**, not clock time. This makes the system robust to variable activity rates.

Source: https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py

### 2.3 When Recency Weighting Outperforms Alternatives

- **Conversational agents**: Recent messages are almost always more relevant than old ones in multi-turn dialogue.
- **Rapidly changing environments**: Simulation agents, game agents, or any system where world state evolves continuously.
- **Task-oriented agents**: Current task context matters more than historical tasks.
- **Reducing stale memory pollution**: Prevents outdated facts from dominating retrieval after the world has changed.

### 2.4 Alternative Temporal Approaches

| Approach | Mechanism | Tradeoff |
|----------|-----------|----------|
| Absolute time decay | `score = e^(-lambda * hours)` | More intuitive but requires tuning lambda |
| Sliding window | Only consider last N hours/days | Simple but hard cutoff |
| Temporal bucketing | Weight by time bucket (last hour: 1.0, last day: 0.8, last week: 0.5) | Simple, no tuning, but coarse-grained |
| Bi-temporal tracking | Track both when a fact was *recorded* and when it was *valid* | Most powerful but complex (see Graphiti) |

### 2.5 Zep/Graphiti's Bi-Temporal Model

Graphiti tracks two temporal dimensions for every fact:
1. **Transaction time**: When the fact was recorded in the system
2. **Valid time**: When the fact was true in the real world

This enables queries like:
- "What is true right now?" (current valid time)
- "What was true on March 1st?" (historical valid time)
- "What did we know as of last Tuesday?" (historical transaction time)

When information changes, old facts are **invalidated rather than deleted**, preserving full temporal history with explicit validity windows.

Source: https://github.com/getzep/graphiti | https://arxiv.org/abs/2501.13956

---

## 3. Importance Scoring

### 3.1 How It Works

Importance scoring distinguishes high-value memories from mundane observations. Without it, "ate breakfast at 8am" could outrank "received a promotion" if the breakfast memory happens to be more semantically similar to the query.

### 3.2 Generative Agents: LLM-as-Judge for Poignancy

Each memory node has a pre-computed `poignancy` score (1-10) assigned by the LLM at creation time:

```python
# From retrieve.py -- importance uses the pre-stored poignancy score
importance_out[node.node_id] = node.poignancy

# Min-max normalized to [0, 1]
importance_out = normalize_dict_floats(importance_out, 0, 1)
```

The scoring prompt asks: "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a breakup, college acceptance), rate the likely poignancy of the following piece of memory."

This is a **single LLM call per memory, amortized at write time** -- no cost at retrieval time.

Source: https://github.com/joonspk-research/generative_agents

### 3.3 The Complete Retrieval Score Formula

The Generative Agents source code reveals the exact combination formula with specific weights:

```python
# The master retrieval score per memory node
master_out[key] = (persona.scratch.recency_w * recency_out[key] * gw[0]
                 + persona.scratch.relevance_w * relevance_out[key] * gw[1]
                 + persona.scratch.importance_w * importance_out[key] * gw[2])

# Global weights (hardcoded):
gw = [0.5, 3, 2]
# gw[0] = 0.5 (recency weight)
# gw[1] = 3.0 (relevance weight)  
# gw[2] = 2.0 (importance weight)

# persona.scratch.recency_w, relevance_w, importance_w are per-persona multipliers (typically 1.0)
```

**Effective weighting**: Relevance is weighted **6x** more than recency. Importance is weighted **4x** more than recency. The system strongly prioritizes semantic relevance, uses importance as a secondary signal, and treats recency as a tiebreaker.

**Relevance** is computed via cosine similarity between the focal point embedding and each memory node's embedding:

```python
focal_embedding = get_embedding(focal_pt)
relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)

def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))
```

The top `n_count` (default 30) nodes by combined score are returned.

Source: https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py

### 3.4 MemGPT/Letta: Agent-as-Importance-Judge

Letta takes a fundamentally different approach: instead of scoring importance externally, the **agent itself decides** what is important through active memory management.

**Three-tier memory hierarchy**:

| Tier | Always in Context? | Access Method | Importance Implication |
|------|-------------------|---------------|----------------------|
| Core Memory (blocks) | Yes, pinned to system prompt | Direct read/write via tools | Highest -- agent chose to keep it visible |
| Recall Memory | No | `conversation_search`, `conversation_search_date` | Medium -- full chat history, agent searches on demand |
| Archival Memory | No | `archival_memory_insert`, `archival_memory_search` | Lower -- useful but not always needed |

**Memory management tools**:

| Tool | Function | Concurrency Behavior |
|------|----------|---------------------|
| `memory_insert` | Append to in-context block | Additive, safe for multi-agent |
| `memory_replace` | Edit in-context block | Validates content before replacing |
| `memory_rethink` | Full rewrite of block | "Most recent write wins", destructive |
| `archival_memory_insert` | Store to long-term vector store | Semantic search indexed |
| `archival_memory_search` | Query archival store | Returns paginated results |
| `conversation_search` | Text search message history | Paginated |
| `conversation_search_date` | Date-filtered history search | Time-bounded |

The agent maintains awareness of its own memory limitations: "prior messages have been hidden from view due to conversation memory constraints." Tool results are truncated to "prevent bad functions from overflowing the agent context window."

When context exceeds thresholds, automatic summarization triggers, "trimming messages length from prior_len to current_length."

Source: https://github.com/letta-ai/letta | https://docs.letta.com/guides/agents/memory

### 3.5 Mem0: Lifecycle-Based Importance

Mem0 manages importance through the memory lifecycle rather than explicit scoring:

- **Automatic deduplication**: System detects semantic overlap and merges rather than creating duplicates
- **Multi-level scoping**: Memories scoped to user_id, agent_id, session_id -- user-level memories implicitly persist longer than session-level
- **Reranking**: Search results can be reranked for contextual relevance, acting as a proxy for importance

```python
memory = Memory.from_config(config)
memory.add(messages, user_id="alice")

# Reranking acts as importance-aware retrieval
results = memory.search(
    query="preferences", 
    user_id="alice", 
    top_k=3, 
    rerank=True
)
```

Source: https://github.com/mem0ai/mem0 | https://docs.mem0.ai/features/graph-memory

---

## 4. Reflection and Synthesis

### 4.1 How It Works

Reflection is the process by which an agent periodically pauses to synthesize higher-level insights from accumulated memories. Instead of only retrieving raw observations, the agent generates abstract conclusions that can themselves be retrieved, creating a hierarchical memory structure.

### 4.2 Generative Agents: Reflection Trees -- Exact Implementation

The complete reflection pipeline from the source code:

**Step 1 -- Trigger mechanism**: Reflections fire when accumulated importance scores deplete a counter.

```python
# From reflect.py
def reflection_trigger(persona):
    if (persona.scratch.importance_trigger_curr <= 0 and
        [] != persona.a_mem.seq_event + persona.a_mem.seq_thought):
        return True
```

Each new observation's poignancy score is subtracted from `importance_trigger_curr`. When it hits zero or below, and the memory stream is non-empty, reflection activates.

**Step 2 -- Focal point generation**: Select the N most recent non-idle memories and ask the LLM to generate 3 salient high-level questions.

```python
def generate_focal_points(persona, n=3):
    nodes = [[i.last_accessed, i]
             for i in persona.a_mem.seq_event + persona.a_mem.seq_thought
             if "idle" not in i.embedding_key]
    nodes = sorted(nodes, key=lambda x: x[0])
    # Pass to LLM via run_gpt_prompt_focal_pt()
    # Returns: ["What is Klaus most passionate about?", ...]
```

**Step 3 -- Insight generation**: For each focal point, retrieve relevant memories and generate insights with evidence.

```python
def generate_insights_and_evidence(persona, nodes, n=5):
    ret = run_gpt_prompt_insight_and_guidance(persona, statements, n)[0]
    # Returns: [(insight_text, [evidence_node_ids]), ...]
```

**Step 4 -- Memory integration**: Reflections are stored as "thoughts" with their own embeddings, poignancy scores, and evidence links.

```python
persona.a_mem.add_thought(
    created,           # Timestamp
    expiration,        # 30 days from creation
    s, p, o,           # Subject-predicate-object triple
    thought,           # The insight text
    keywords,          # Extracted keywords
    thought_poignancy, # LLM-rated importance
    thought_embedding_pair,  # Embedding for retrieval
    evidence           # List of source memory node IDs
)
```

**Step 5 -- Counter reset**: After reflection completes, the importance counter resets to its maximum value.

```python
reset_reflection_counter(persona)
# Resets importance_trigger_curr to importance_trigger_max
```

This creates a **hierarchical memory structure**: raw observations at the bottom, first-order reflections above them, and potentially meta-reflections above those. All levels participate in the same retrieval scoring (recency + importance + relevance), so reflections compete with raw observations for retrieval slots.

Source: https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/reflect.py

### 4.3 Self-RAG: Self-Reflective Retrieval (Asai et al. 2023)

Self-RAG trains a single LM to adaptively decide when to retrieve and to self-critique its generations using four **reflection tokens**:

| Token | Purpose | Values |
|-------|---------|--------|
| **Retrieve** | Should I retrieve for this segment? | yes / no / continue |
| **IsRel** | Is the retrieved passage relevant to the query? | relevant / irrelevant |
| **IsSup** | Does the passage support my generation? | fully supported / partially / no support |
| **IsUse** | How useful is my overall response? | 1-5 scale |

**Inference workflow**:
1. Model generates a segment of text
2. **Retrieve** token decides whether retrieval is needed for the next segment
3. If yes, passages are retrieved and each scored by **IsRel**
4. Model generates a response grounded in relevant passages
5. **IsSup** and **IsUse** tokens evaluate generation quality
6. **Segment-level beam search** explores multiple retrieval/generation paths in parallel

**Training**: GPT-4 generates reflection token labels on training data, which are then distilled into the target model (7B or 13B parameters) via supervised fine-tuning. The model learns to generate reflection tokens as part of its normal token sequence.

**Results**: 7B and 13B Self-RAG models "significantly outperform state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks" including ChatGPT on open-domain QA, reasoning, and fact verification, with "significant gains in improving factuality and citation accuracy for long-form generations."

Source: https://arxiv.org/abs/2310.11511

### 4.4 CRAG: Corrective Retrieval Augmented Generation (Yan et al. 2024)

CRAG adds a retrieval quality evaluation step that corrects poor retrievals before generation:

**Architecture**:

```
Query -> Retrieve -> Evaluate Confidence -> [Correct] -> Use documents directly
                                         -> [Incorrect] -> Discard, web search
                                         -> [Ambiguous] -> Supplement with web search
         -> Decompose-Recompose -> Generate
```

Components:
1. **Lightweight retrieval evaluator**: Generates a confidence score for retrieved documents
2. **Three action triggers** based on confidence thresholds:
   - **Correct**: High confidence -- proceed with retrieved documents
   - **Incorrect**: Low confidence -- discard local retrieval, use web search instead
   - **Ambiguous**: Medium confidence -- combine local retrieval with web search
3. **Decompose-then-recompose algorithm**: Breaks documents into knowledge strips, "selectively focuses on key information and filters out irrelevant information"
4. **Web search augmentation**: When local retrieval fails, queries are rewritten for web search

CRAG is "plug-and-play" -- it layers on top of any existing RAG system.

Source: https://arxiv.org/abs/2401.15884

### 4.5 LangGraph Implementation of Self-RAG and CRAG

LangGraph implements these patterns as **state machines** with conditional edges rather than fixed chains:

```
Self-RAG/CRAG State Machine:

[Query] -> [Route] -> [Retrieve] -> [Grade Documents]
                                          |
                    [All relevant] -------+------- [Any irrelevant]
                          |                              |
                    [Generate]                   [Rewrite Query]
                          |                              |
                    [Grade Generation]           [Web Search]
                          |                              |
                    [Supported?] ----[No]----> [Rewrite Query]
                          |
                      [Output]
```

"A state machine simply lets us define a set of steps and set the transition options between them" -- enabling loops and conditional branching. Binary decisions use "Pydantic to model the output and supply this function as an OpenAI tool" for consistent routing logic.

Source: https://blog.langchain.com/agentic-rag-with-langgraph/

---

## 5. Multi-Hop Retrieval

### 5.1 How It Works

Multi-hop retrieval answers complex queries requiring information from multiple memory nodes or documents. Instead of single-shot retrieve-and-generate, the system performs multiple retrieval rounds where each round's output informs the next round's query.

### 5.2 Microsoft GraphRAG (Edge et al. 2024)

GraphRAG builds a hierarchical knowledge graph from source text for corpus-level question answering.

**Indexing pipeline**:
1. LLMs extract entities and relationships from text chunks
2. Entities become nodes, relationships become edges in a knowledge graph
3. **Leiden community detection** clusters related entities hierarchically
4. LLM generates summaries for each community at each hierarchy level

**Three query modes**:

| Mode | Mechanism | Best For |
|------|-----------|----------|
| **Local Search** | Start from query-relevant entities, traverse neighbors | Specific factual questions ("What does Alice work on?") |
| **Global Search** | Map-reduce across all community summaries | Corpus-wide questions ("What are the main themes?") |
| **DRIFT Search** | Entity-neighbor traversal + community context | Hybrid specific + contextual queries |

**Key results**: "Substantial improvements in comprehensiveness and diversity of answers" vs. naive RAG on global sensemaking queries. "Similar level of faithfulness to baseline RAG" on factual accuracy -- it improves breadth without sacrificing correctness.

**Limitations**: Batch-oriented, expensive to index (many LLM calls per document), static once built. Not suited for continuously evolving agent memory.

Source: https://arxiv.org/abs/2404.16130 | https://microsoft.github.io/graphrag/

### 5.3 Zep/Graphiti: Temporal Knowledge Graphs (Rasmussen et al. 2025)

Graphiti is specifically designed for the multi-hop needs of agentic memory, addressing GraphRAG's limitations:

**Graph components**:
- **Entities (nodes)**: People, products, policies with summaries that evolve temporally
- **Facts/Relationships (edges)**: Triplets with **temporal validity windows** (when facts became/stopped being true)
- **Episodes (provenance)**: Raw ingested data serving as ground truth; every derived fact traceable to source
- **Custom Types (ontology)**: Developer-defined entity/edge types via Pydantic models

**Incremental construction**: New episodes are incorporated immediately without full graph recomputation. When information conflicts, old facts are **invalidated** (not deleted), preserving full history.

**Hybrid retrieval** combining:
- Semantic embeddings for concept matching
- BM25 keyword search for exact terms
- Graph traversal for structural/relational queries
- Reranking using graph distance

**Benchmark results**:

| Benchmark | Zep | MemGPT | Improvement |
|-----------|-----|--------|-------------|
| Deep Memory Retrieval (DMR) | 94.8% | 93.4% | +1.4% |
| LongMemEval (accuracy) | -- | -- | Up to +18.5% |
| LongMemEval (latency) | -- | -- | 90% reduction |

Comparison with GraphRAG:

| Aspect | GraphRAG | Graphiti |
|--------|----------|----------|
| Data Model | Batch, static summaries | Continuous, incremental updates |
| Temporal Handling | Basic timestamps | Explicit bi-temporal with auto-invalidation |
| Contradiction Resolution | LLM-driven judgments | Automatic fact invalidation preserving history |
| Query Latency | Seconds to tens of seconds | Sub-second typical |

Source: https://github.com/getzep/graphiti | https://arxiv.org/abs/2501.13956

### 5.4 Mem0 Graph Memory: Parallel Vector + Graph Retrieval

Mem0 runs graph retrieval **in parallel** with vector search, enriching results without replacing them:

**Entity extraction**: An extraction LLM processes `memory.add()` payloads to identify entities, relationships, and timestamps. Customizable via `custom_prompt` to filter what gets captured (e.g., "Only capture people, organisations, project links").

**Dual storage**: Embeddings go to vector store; nodes and edges go to graph backend.

**Parallel search**: During `memory.search()`, vector similarity and graph queries run concurrently. Results include both vector hits and a `relations` array.

**Critical design choice**: "Graph edges do not reorder those hits automatically" -- vector ranking determines result order. Graph relationships appear as supplementary context.

```python
config = {
    "graph_store": {
        "provider": "neo4j",  # Also: Memgraph, Neptune, Kuzu, Apache AGE
        "config": {
            "url": "neo4j+s://<instance>.databases.neo4j.io",
            "username": "neo4j",
            "password": "<PASSWORD>",
        },
        "custom_prompt": "Only capture people, organisations, project links",
        "threshold": 0.75  # Confidence filter for noisy edges
    }
}
memory = Memory.from_config(config)
memory.add(conversation, user_id="demo-user")

# Search returns vector hits + graph relations
results = memory.search("Who did Alice meet?", user_id="demo-user", top_k=3, rerank=True)
# results["results"][i]["relations"] -> contextual graph connections
```

**Supported graph backends**: Neo4j (recommended, requires APOC plugin), Memgraph (Docker, Bolt protocol), Neptune Analytics, Neptune DB, Kuzu (embedded, in-process), Apache AGE (PostgreSQL extension).

Source: https://docs.mem0.ai/open-source/graph_memory/overview | https://docs.mem0.ai/features/graph-memory

### 5.5 Multi-Vector Retriever Pattern (LangChain)

Decouples the retrieval unit from the synthesis unit for multi-hop reasoning:

1. Create **summaries** of documents/tables optimized for search
2. Store summaries as embeddings, link to full original documents in a docstore
3. Retrieve via **summary similarity**, but pass the **full original document** to the LLM
4. Use retrieved documents to identify related entities, then retrieve those entities' documents

This enables multi-hop chains where each step retrieves at a different granularity. As LangChain describes it: "we can create a summary of a verbose document optimized to vector-based similarity search, but still pass the full document into the LLM."

The pattern extends to multi-modal retrieval:
- **Option 1**: CLIP embeddings retrieve images; raw images go to multimodal LLMs
- **Option 2**: Generate text summaries from images, retrieve via text, pass raw images for synthesis
- **Option 3**: Combine image summaries with raw image references

Source: https://blog.langchain.com/semi-structured-multi-modal-rag/

---

## 6. Hybrid Retrieval

### 6.1 How It Works

Hybrid retrieval combines multiple retrieval signals -- typically keyword-based (sparse) and semantic (dense) -- to overcome the limitations of either alone. The key insight: semantic search captures meaning but misses exact matches; keyword search catches exact matches but misses semantic equivalents.

### 6.2 BM25 Algorithm

BM25 (Best Matching 25) is the dominant keyword scoring function, extending TF-IDF with document length normalization:

```
score(D, Q) = SUM[ IDF(qi) * f(qi,D) * (k1+1) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl)) ]
```

Where:
- `k1` and `b` are calibration parameters for dataset-specific tuning
- `IDF(qi)` weights keyword uniqueness
- `|D|/avgdl` normalizes by document length relative to collection average
- `f(qi,D)` is term frequency in document D

BM25F extends this to multiple text fields with different weights (e.g., title weighted higher than body).

Source: https://weaviate.io/blog/hybrid-search-explained

### 6.3 Score Fusion Methods

#### Reciprocal Rank Fusion (RRF)

Merges ranked lists from multiple retrievers without requiring score normalization:

```
RRF_score(d) = SUM[ 1 / (k + rank_i(d)) ]
```

Where `k` is a smoothing constant (default 0 in Weaviate, 60 in the original RRF paper).

**Worked example** (k=0):

| Document | BM25 Rank | Dense Rank | RRF Score | Final Rank |
|----------|-----------|-----------|-----------|------------|
| A | 1 | 3 | 1/1 + 1/3 = **1.33** | 2 |
| B | 2 | 1 | 1/2 + 1/1 = **1.50** | **1** |
| C | 3 | 2 | 1/3 + 1/2 = **0.83** | 3 |

Document B wins because it ranks highly in **both** lists, even though it's not #1 in either.

Source: https://weaviate.io/blog/hybrid-search-explained

#### Alpha Weighting (Linear Interpolation)

Direct score combination with a tunable alpha:

```
hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
```

| Alpha | Behavior |
|-------|----------|
| 1.0 | Pure semantic search |
| 0.75 | Weaviate default, slightly favors dense |
| 0.5 | Equal weighting |
| 0.3 | Emphasizes sparse/keyword |
| 0.0 | Pure keyword search |

**Pinecone implementation** (unified sparse-dense index):
```python
def hybrid_scale(dense, sparse, alpha: float):
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

# Upsert both vectors in single record
index.upsert([{
    'id': doc_id,
    'sparse_values': sparse_embedding,
    'values': dense_embedding,
    'metadata': {...}
}])

# Query with both
index.query(
    vector=dense_query,
    sparse_vector=sparse_query,
    top_k=10,
    include_metadata=True
)
```

The unified index eliminates the need for separate sparse and dense engines.

Source: https://www.pinecone.io/learn/hybrid-search-intro/

### 6.4 Reranking as the Third Stage

A two-stage pipeline maximizes both recall and precision:

1. **Stage 1 (Fast retrieval)**: Retrieve many candidates (high top_k) via hybrid search
2. **Stage 2 (Reranking)**: Cross-encoder scores each query-document pair, keeping only the best (smaller top_n)

**Why reranking matters**: Bi-encoders compress document meaning into a single vector (information loss). Cross-encoder rerankers "receive raw information directly into the large transformer computation" with the full query-document pair, avoiding compression loss. But computational cost is prohibitive for first-stage: "using BERT on V100 GPU with 40M records would require >50 hours to return a single query result" vs. <100ms for vector search.

**Available rerankers**:

| Model | Provider | Notes |
|-------|----------|-------|
| rerank-v4.0-pro / fast | Cohere | 100+ languages, scores 0-1, max 96 docs/call |
| rerank-v3.5 | Cohere | Unified multilingual |
| bge-reranker-v2-m3 | BAAI | Open-source, pairs with BGE-M3 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | HuggingFace | Fast, English |

Source: https://www.pinecone.io/learn/series/rag/rerankers/ | https://docs.cohere.com/v2/docs/rerank-overview

### 6.5 Anthropic's Contextual Retrieval (September 2024)

The most impactful single-technique improvement for retrieval quality documented in this period.

**Core problem**: Chunks lose document-level context when embedded. A chunk saying "the committee decided to approve the project" retrieves poorly without knowing which committee.

**Solution**: Before embedding, prepend each chunk with a short context passage generated by Claude:

```
Prompt template:
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of
the chunk. Answer only with the succinct context and nothing else.
```

Generated context is 50-100 tokens. The contextualized chunk is used for **both** embedding and BM25 indexing ("Contextual BM25").

**Results** across multiple knowledge domains:

| Configuration | Top-20 Retrieval Failure Rate | Reduction vs. Baseline |
|--------------|-------------------------------|------------------------|
| Standard embeddings (baseline) | 5.7% | -- |
| Contextual Embeddings alone | 3.7% | **35%** reduction |
| Contextual Embeddings + Contextual BM25 | 2.9% | **49%** reduction |
| + Reranking | 1.9% | **67%** reduction |

**Cost**: ~$1.02 per million document tokens using prompt caching (the whole document is cached, only the per-chunk prompt changes). This is a **write-time** optimization -- cost is amortized at indexing, with zero additional cost at query time.

Source: https://www.anthropic.com/news/contextual-retrieval

---

## 7. Agentic Retrieval

### 7.1 How It Works

Agentic retrieval inverts the traditional RAG pattern: instead of always retrieving on every query, the **agent itself decides** when, what, and how to retrieve. Memory access is a tool the agent can choose to invoke, not a mandatory preprocessing step.

### 7.2 MemGPT/Letta: Self-Directed Memory Search

The canonical implementation of agentic retrieval, inspired by operating system virtual memory.

**Architecture**: The agent has explicit tools for memory operations and decides through its own reasoning when to use them. The system prompt instructs the agent about its memory architecture:

```
The agent is told:
- "Your core memory has limited capacity. Use memory management tools to save important information."
- "When you need information not in your current context, use archival_memory_search or conversation_search."
- "Prior messages have been hidden from view due to conversation memory constraints."
```

**When does the agent retrieve?** It is not triggered by confidence scores or reflection tokens. The agent uses its own judgment through the LLM's reasoning process. If a user asks about something the agent doesn't see in its current context, it infers it should search archival or recall memory.

**Context compilation**: Letta compiles context by combining system prompt + embedded memory blocks + external memory metadata (passage counts, timestamps) + in-context message history. The system tracks "system message token count and token_counts_no_system separately" to manage the context window budget.

**Automatic summarization**: When context exceeds thresholds, the agent auto-summarizes and evicts older messages, analogous to OS page-swapping.

Source: https://arxiv.org/abs/2310.08560 | https://github.com/letta-ai/letta

### 7.3 Agentic RAG with LangGraph: State Machine Retrieval

LangGraph formalizes agentic retrieval as state machines with conditional edges:

```
State Machine Architecture:

[Query] --> [Route Decision]
              |            |
        [Vector Store]  [Web Search]
              |            |
        [Grade Documents]  |
         /       \         |
    [Relevant] [Irrelevant]|
        |          \       |
   [Generate]    [Rewrite Query]
        |              |
   [Grade Generation]  [Re-retrieve or Web Search]
        |
   [Supported + Useful?]
     /          \
  [Output]   [Loop back]
```

**Key mechanisms**:
- **Routing**: LLM classifies each query and directs to appropriate source
- **Document grading**: Binary relevance assessment using Pydantic-modeled outputs bound as LLM tools
- **Query rewriting**: If documents are irrelevant, reformulate and re-retrieve
- **Generation grading**: Check if response is supported by documents and actually useful
- **Conditional edges**: "If any document lacks relevance, trigger web search via Tavily API"

Source: https://blog.langchain.com/agentic-rag-with-langgraph/

### 7.4 Self-RAG's Adaptive Retrieval Decision

Self-RAG represents a lightweight form of agentic retrieval where the decision is trained into model weights:

The **Retrieve** token decides per-segment:
- `[Retrieve: yes]` -- pause generation, retrieve passages, continue
- `[Retrieve: no]` -- use parametric knowledge
- `[Retrieve: continue]` -- use already-retrieved context

This is faster than MemGPT's tool-calling approach (no external function call overhead) but less flexible (can't route to different sources, can't decide search parameters).

Source: https://arxiv.org/abs/2310.11511

### 7.5 When Agentic Retrieval Outperforms Always-On RAG

| Scenario | Why Agentic Wins |
|----------|-----------------|
| Simple factual questions | No retrieval needed -- agent uses parametric knowledge, saving latency |
| Multi-step reasoning | Agent retrieves, reasons, retrieves again with refined query |
| Memory maintenance | Agent decides to update/consolidate memories, not just read them |
| Context budget management | Agent decides how many memories to retrieve based on available space |
| Mixed-source queries | Agent routes sub-questions to different stores (archival vs. web vs. history) |

---

## 8. Comparative Analysis

### 8.1 System Comparison

| System | Retrieval Method | Temporal Awareness | Importance Scoring | Reflection | Benchmark |
|--------|-----------------|-------------------|-------------------|------------|-----------|
| **Generative Agents** | Weighted sum: 0.5*recency + 3*relevance + 2*importance | Positional decay (0.995^rank) | LLM 1-10 poignancy at write time | Threshold-triggered insight synthesis | Qualitative (Turing-test style) |
| **MemGPT/Letta** | Agent-directed tool calls across 3 tiers | Implicit via conversation ordering | Agent curates core vs. archival | Self-directed via tool calls | DMR: 93.4% |
| **Zep/Graphiti** | Hybrid: semantic + BM25 + graph traversal | Bi-temporal validity windows | Graph structure implicit | Entity resolution + fact invalidation | DMR: 94.8%, LongMemEval: +18.5% |
| **Mem0** | Vector search + parallel graph enrichment | Timestamp metadata | Graph confidence thresholds | LLM extraction at ingest | -- |
| **Self-RAG** | Adaptive on-demand via reflection tokens | Not explicit | IsRel/IsSup/IsUse tokens | Per-segment retrieve/critique | Outperforms ChatGPT on QA |
| **CRAG** | Confidence-gated with web fallback | Not explicit | Evaluator confidence score | Decompose-then-recompose | Significant improvement on 4 datasets |
| **Contextual Retrieval** | Hybrid contextual BM25 + contextual embeddings + reranking | Not explicit | Not explicit | Not applicable (indexing technique) | 67% reduction in retrieval failures |
| **GraphRAG** | Community-based map-reduce summarization | Basic timestamps | Community hierarchy | Leiden community summaries | Outperforms naive RAG on global queries |
| **BGE-M3** | Dense + sparse + ColBERT in one model | Not applicable (embedding model) | Not applicable | Not applicable | SOTA on MIRACL multilingual |

### 8.2 When to Use What

| Strategy | Best Scenario | Worst Scenario |
|----------|--------------|----------------|
| **Pure semantic** | Simple factual lookup, single-turn | Keyword-heavy queries, temporal questions |
| **Temporal decay** | Conversational agents, evolving environments | Historical research, "what did I say last month?" |
| **Importance scoring** | Long-lived agents with diverse memory types | Short sessions, uniform importance |
| **Reflection** | Agents needing high-level understanding over time | Real-time systems (too slow), short sessions |
| **Multi-hop/graph** | Complex relational queries, entity-centric domains | Simple questions (overkill), sparse data |
| **Hybrid retrieval** | Production systems needing both precision and recall | Extremely simple use cases (unnecessary complexity) |
| **Agentic retrieval** | Autonomous agents with diverse memory needs | Fixed pipelines, predictable query patterns |
| **Contextual Retrieval** | Any system with document-level context loss | Already short, self-contained chunks |

### 8.3 The State of the Art (2025-2026)

The most capable systems layer multiple strategies:

1. **Anthropic's Contextual Retrieval** (hybrid BM25 + contextual embeddings + reranking) achieves the single strongest documented improvement: 67% reduction in top-20 retrieval failures. This is a write-time optimization that improves all downstream retrieval.

2. **Zep/Graphiti** (temporal knowledge graph + hybrid retrieval) achieves 94.8% on DMR and 18.5% improvement on LongMemEval -- the strongest results for agentic memory specifically, with 90% latency reduction.

3. **The Generative Agents formula** (`0.5*recency + 3*relevance + 2*importance` with reflection) remains the most influential architecture for autonomous agent memory. The specific global weights [0.5, 3, 2] reveal that the original designers found relevance to be the dominant signal.

4. **MemGPT/Letta's self-directed approach** represents the purest form of agentic retrieval: the agent maintains its own three-tier memory hierarchy and decides when/what to search.

5. **BGE-M3** represents the embedding frontier: dense + sparse + ColBERT in one model, enabling hybrid search without separate BM25 infrastructure.

The clear trend: **composable, multi-signal retrieval** where no single mechanism dominates, and **agent-controlled memory access** where the agent decides its own retrieval strategy at runtime.

---

## 9. Open Questions and Research Frontiers

1. **Retrieval-aware training**: Should agents be fine-tuned on retrieval tasks over their own memory stores? Self-RAG suggests yes, but this hasn't been explored for persistent agent memory.

2. **Multi-agent shared retrieval**: When multiple agents share a memory store, how should retrieval be scoped? Letta's multi-agent memory blocks are a start, but cross-agent retrieval for collaboration remains underexplored.

3. **Forgetting as retrieval policy**: Active forgetting (archiving or removing memories) as a retrieval optimization. Fewer memories means faster, more precise retrieval. The tension between completeness and efficiency is unresolved.

4. **Embedding model co-evolution**: As models improve, should stores be periodically re-embedded? The operational cost is high, but drift degrades quality over time.

5. **Causal retrieval**: Moving beyond correlation (semantic similarity) to causation -- retrieving memories that caused or were caused by the current situation. No current system does this well.

6. **Memory routing intelligence**: Lightweight classifiers that dispatch queries to the optimal retrieval backend (vector store vs. graph vs. conversation history vs. web) before executing any search.

---

## Sources

### Papers

1. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *arXiv:2304.03442*. https://arxiv.org/abs/2304.03442

2. Packer, C., Fang, V., Patil, S. G., Lin, K., Wooders, S., & Gonzalez, J. E. (2023). "MemGPT: Towards LLMs as Operating Systems." *arXiv:2310.08560*. https://arxiv.org/abs/2310.08560

3. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *arXiv:2310.11511*. https://arxiv.org/abs/2310.11511

4. Yan, S., Gu, J., Zhu, Y., & Ling, Z. (2024). "Corrective Retrieval Augmented Generation." *arXiv:2401.15884*. https://arxiv.org/abs/2401.15884

5. Edge, D., Trinh, H., Cheng, N., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130*. https://arxiv.org/abs/2404.16130

6. Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." *arXiv:2501.13956*. https://arxiv.org/abs/2501.13956

7. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings." *arXiv:2402.03216*. https://arxiv.org/abs/2402.03216

8. Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997*. https://arxiv.org/abs/2312.10997

9. Zhang, Z., et al. (2024). "A Survey on the Memory Mechanism of Large Language Model based Agents." *arXiv:2404.13501*. https://arxiv.org/abs/2404.13501

### Source Code

10. Generative Agents - retrieve.py (retrieval scoring formula). https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py

11. Generative Agents - reflect.py (reflection mechanism). https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/reflect.py

12. Letta (formerly MemGPT) - agent.py. https://github.com/letta-ai/letta

13. Graphiti - temporal knowledge graph engine. https://github.com/getzep/graphiti

14. Mem0 - memory layer for AI agents. https://github.com/mem0ai/mem0

### Documentation and Blogs

15. Anthropic (2024). "Introducing Contextual Retrieval." https://www.anthropic.com/news/contextual-retrieval

16. Weng, L. (2023). "LLM Powered Autonomous Agents." *Lil'Log*. https://lilianweng.github.io/posts/2023-06-23-agent/

17. LangChain (2024). "Agentic RAG with LangGraph." https://blog.langchain.com/agentic-rag-with-langgraph/

18. LangChain (2023). "Semi-structured Multi-Modal RAG." https://blog.langchain.com/semi-structured-multi-modal-rag/

19. Pinecone (2024). "Rerankers." https://www.pinecone.io/learn/series/rag/rerankers/

20. Pinecone (2023). "Hybrid Search Intro." https://www.pinecone.io/learn/hybrid-search-intro/

21. Weaviate (2024). "Hybrid Search Explained." https://weaviate.io/blog/hybrid-search-explained

22. Cohere - Embed API v2 Reference. https://docs.cohere.com/v2/reference/embed

23. Cohere - Rerank Overview. https://docs.cohere.com/v2/docs/rerank-overview

24. Letta Documentation - Memory. https://docs.letta.com/guides/agents/memory

25. Mem0 Documentation - Graph Memory. https://docs.mem0.ai/features/graph-memory

26. Mem0 Documentation - Graph Memory (OSS). https://docs.mem0.ai/open-source/graph_memory/overview

27. Microsoft GraphRAG Documentation. https://microsoft.github.io/graphrag/

28. BAAI/bge-m3 - HuggingFace Model Card. https://huggingface.co/BAAI/bge-m3

29. Allen AI SPECTER2 - HuggingFace Model Card. https://huggingface.co/allenai/specter2

30. Galileo (2024). "Mastering RAG: How to Select an Embedding Model." https://www.galileo.ai/blog/mastering-rag-how-to-select-an-embedding-model

31. Microsoft Research (2024). "GraphRAG: Unlocking LLM Discovery on Narrative Private Data." https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
