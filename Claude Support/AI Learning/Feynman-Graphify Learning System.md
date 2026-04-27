# The Feynman-Graphify Learning System

A personal knowledge system that combines structured research workflows with knowledge graph construction to deeply learn any topic. Built on Claude Code with 7 research skills and a graph engine.

---

## The Core Idea

Most people learn by reading — passively consuming articles, papers, and videos. This system forces active learning through two complementary mechanisms:

1. **Feynman skills** break a topic apart through structured research, multi-level explanation, source comparison, and peer review. They force you to engage with a topic from every angle — not just read about it.
2. **Graphify** reassembles everything you've researched into a navigable knowledge graph, revealing connections you'd never find by reading linearly.

The result is a persistent, queryable knowledge base that grows with every research session. You don't just learn a topic — you build infrastructure that makes the next topic easier to learn.

---

## Architecture

```
                        You ask a question
                              |
                    +---------+---------+
                    |                   |
              Need to find          Need to understand
              sources first?        a concept first?
                    |                   |
           /feynman-paper-search   /feynman-eli5
                    |                   |
          +---------+---------+         |
          |         |         |         |
       Deep      Literature  Source     |
       Research   Review     Comparison |
          |         |         |         |
          +---------+---------+---------+
                    |
              Write it up?
                    |
           /feynman-paper-writing
                    |
              Review it?
                    |
           /feynman-peer-review
                    |
            All research artifacts
            saved to ~/Desktop/
                    |
              /graphify ~/Desktop/research-topic/
                    |
            Interactive knowledge graph
            (HTML, Obsidian vault, JSON)
                    |
              /graphify query "What connects X to Y?"
              /graphify path "Concept A" "Concept B"
              /graphify explain "Node"
```

### The Skills

| Skill | What It Does | When To Use It |
|-------|-------------|----------------|
| `/feynman-paper-search` | Search 250M+ academic papers across arXiv, Semantic Scholar, and OpenAlex | Starting a new topic. Need to find what's been published. |
| `/feynman-deep-research` | Multi-source investigation with parallel subagents and claim verification | Deep dive on a broad topic. Want a comprehensive brief. |
| `/feynman-literature-review` | Systematic survey organized by theme with gap analysis | Need to map the full landscape of a field. |
| `/feynman-source-comparison` | Side-by-side matrix of multiple sources with agreement/disagreement analysis | Evaluating competing approaches or contradictory claims. |
| `/feynman-eli5` | Explain a concept at multiple complexity levels (5-year-old through expert) | Need intuition before going deep. Or need to teach someone else. |
| `/feynman-paper-writing` | Draft a structured paper with literature grounding and citations | Synthesizing your understanding into a formal document. |
| `/feynman-peer-review` | Structured critique with severity ratings (FATAL/MAJOR/MINOR) | Stress-testing a paper, proposal, or your own writing. |
| `/graphify` | Transform any corpus into a knowledge graph with community detection | Connecting everything you've learned. Finding surprises. |

### The Backbone: Academic API

All Feynman skills share a common research engine (`academic_api.py`) that searches three databases simultaneously:

- **arXiv** — preprints, cutting-edge research (3-second rate limit between calls)
- **Semantic Scholar** — 214M papers with AI-generated TLDRs and citation context
- **OpenAlex** — 250M+ works, broadest coverage including non-CS fields

This means every skill has access to the same paper discovery, citation graph traversal, and related-paper algorithms. A paper found in one skill is immediately usable in another.

---

## How To Learn Any Topic

### Phase 1: Orientation (30 minutes)

**Goal**: Get the lay of the land. What is this field? Who are the key people? What are the main debates?

```
/feynman-eli5 <your topic>
```

This produces a multi-level explanation:
- **One-Sentence Summary** — what a smart 12-year-old would understand
- **Big Idea** — core intuition with analogies
- **How It Works** — step-by-step mechanics
- **Why It Matters** — real-world impact
- **What To Be Skeptical Of** — limitations and common misconceptions
- **If You Remember 3 Things** — the essence

Output: `~/Desktop/<topic>-eli5.md`

Then find the foundational papers:

```
/feynman-paper-search <your topic>
```

This searches all three academic databases, deduplicates results, and gives you a table of papers with citation counts, TLDRs, and links. From here you can:

- Drill into any paper: `paper <arxiv-id>` for full metadata
- Explore who cites it: `citations <id> --direction both`
- Find related work: `related <id>`
- Read the full text: fetches from arXiv HTML

Output: Saved to `~/Desktop/` as markdown tables.

### Phase 2: Deep Dive (1-2 hours)

**Goal**: Thorough understanding with verified claims and multiple perspectives.

Pick one of two paths depending on your needs:

**Path A: Broad topic, need a research brief**

```
/feynman-deep-research <your topic>
```

This spawns parallel research subagents that investigate sub-questions simultaneously. Each agent searches the web, reads academic papers, and writes findings to separate files. The main agent then synthesizes everything into a single brief with:

- Executive Summary
- Background and key terminology
- Findings organized by theme
- Cross-source analysis
- Open questions and gaps
- Full source list with inline citations

Every critical claim requires 2+ independent sources. Single-source findings get explicit caveats. No fabricated URLs.

Output: `~/Desktop/<topic>-research.md` + `~/Desktop/research-plans/<topic>.md`

**Path B: Need to map an entire field**

```
/feynman-literature-review <your topic>
```

This is more systematic — it traverses citation graphs in both directions, categorizes sources by methodology, and identifies consensus vs. disagreements. The output is organized by theme (not chronologically), with explicit gap analysis showing what's missing from the literature.

Output: `~/Desktop/<topic>-literature-review.md`

### Phase 3: Compare and Critique (30-60 minutes)

**Goal**: Develop an informed opinion. Understand trade-offs.

If you found competing approaches or contradictory claims:

```
/feynman-source-comparison <list of sources or approaches to compare>
```

This builds a comparison matrix with confidence levels:

| Dimension | Source A | Source B | Agreement |
|-----------|----------|----------|-----------|
| Performance | claim + evidence [HIGH] | claim + evidence [MEDIUM] | Disagree |
| Scalability | claim + evidence [HIGH] | claim + evidence [HIGH] | Agree |

Each cell contains the claim, supporting evidence, and a confidence rating. Divergences are explained (different methodologies? different datasets? different definitions?).

Output: `~/Desktop/<topic>-comparison.md`

Then stress-test the strongest source:

```
/feynman-peer-review <path to paper or arXiv ID>
```

This produces an academic-style review with severity tags:
- **[FATAL]** — fundamental flaws that invalidate conclusions
- **[MAJOR]** — significant issues that need addressing
- **[MINOR]** — suggestions for improvement

Plus: questions for the authors, missing references, and a verdict (ACCEPT / REVISE / REJECT).

Output: `~/Desktop/<topic>-review.md`

### Phase 4: Synthesize (optional, 1-2 hours)

**Goal**: Cement your understanding by writing about it.

```
/feynman-paper-writing <your thesis or contribution>
```

This drafts a structured paper with proper citations, methodology, and literature grounding. Even if you never publish it, the act of writing forces you to organize your thinking. The Feynman technique in practice — if you can explain it clearly enough to write it down, you understand it.

Output: `~/Desktop/<topic>-paper.md`

### Phase 5: Build the Knowledge Graph

**Goal**: Connect everything and discover surprises.

Collect all your research artifacts into a folder:

```bash
mkdir ~/Desktop/learning/<topic>
mv ~/Desktop/<topic>-*.md ~/Desktop/learning/<topic>/
```

Then build the graph:

```
/graphify ~/Desktop/learning/<topic>
```

Graphify reads every file, extracts entities and relationships, runs community detection, and produces:

1. **`graph.html`** — an interactive visualization you open in your browser. Nodes are concepts, edges are relationships. Communities are color-coded clusters. Click any node to see its connections.

2. **`GRAPH_REPORT.md`** — an audit trail that answers:
   - What are the central concepts? (god nodes — highest connectivity)
   - What surprising connections exist? (bridge nodes crossing community boundaries)
   - What questions can this graph answer?
   - How cohesive is each topic cluster?

3. **`graph.json`** — the raw graph data that persists across sessions. This is key — weeks later, you can query this graph without re-reading anything.

Every edge in the graph is tagged with its provenance:
- **EXTRACTED** — explicitly stated in the source material
- **INFERRED** — reasonable inference from context
- **AMBIGUOUS** — uncertain, flagged for your review

This honesty layer means you know exactly what the graph found versus what it guessed.

### Phase 6: Query and Grow

The graph is now a persistent knowledge base. Use it:

```
/graphify query "What is the relationship between attention mechanisms and memory networks?"
```

This runs a graph traversal (BFS for broad context, DFS for tracing specific paths) and explains what it finds in plain language.

```
/graphify path "Transformer" "Retrieval-Augmented Generation"
```

This finds the shortest path between two concepts — the chain of relationships connecting them.

```
/graphify explain "Self-Attention"
```

This gives a plain-language explanation of one node and all its connections in the graph.

And the graph grows incrementally:

```
/graphify add https://arxiv.org/abs/2401.12345
```

This fetches the paper, transcribes it if needed, extracts entities, and merges them into the existing graph. The community structure updates automatically.

---

## Replicating This System

### Prerequisites

1. **Claude Code** — Anthropic's CLI tool. The skills run inside Claude Code sessions.
2. **Python 3.10+** — for the academic API and Graphify scripts.
3. **Internet access** — for paper search, web research, and full-text fetching.

### Step 1: Install the Academic API

The backbone that all Feynman skills use for paper discovery:

```bash
# The file lives at ~/.claude/scripts/lib/academic_api.py
# It's a standalone script with no dependencies beyond the standard library + urllib

# Test it:
python3 ~/.claude/scripts/lib/academic_api.py search "transformer attention" --max 5
```

The API wraps three free academic databases. No API keys required for basic usage, but you get better rate limits with:

```bash
# Optional: add to ~/.claude/.env
SEMANTIC_SCHOLAR_API_KEY=<free key from semanticscholar.org>
OPENALEX_EMAIL=<your email for polite pool>
```

### Step 2: Install the Skills

Each skill is a markdown file (`SKILL.md`) in its own directory under `~/.claude/skills/`. Claude Code reads these automatically. The 7 Feynman skills and Graphify are:

```
~/.claude/skills/
  feynman-deep-research/SKILL.md
  feynman-eli5/SKILL.md
  feynman-literature-review/SKILL.md
  feynman-paper-search/SKILL.md
  feynman-paper-writing/SKILL.md
  feynman-peer-review/SKILL.md
  feynman-source-comparison/SKILL.md
  graphify/SKILL.md
```

Each skill file defines: the trigger command, allowed tools, workflow steps, output format, and guidelines. Claude Code loads the relevant skill when you invoke it.

### Step 3: Set Up Your Learning Directory

```bash
mkdir -p ~/Desktop/learning
mkdir -p ~/Desktop/research-plans
```

All skill outputs go to `~/Desktop/` by default. Organize by topic when you're ready to graphify.

### Step 4: (Optional) Connect to Obsidian

If you use Obsidian as your knowledge management tool, Graphify can output directly to an Obsidian vault:

```
/graphify ~/Desktop/learning/<topic> --obsidian
```

This creates one markdown note per graph node with backlinks, plus a `graph.canvas` file for Obsidian's Canvas view. The graph becomes part of your vault, searchable and linked to everything else you know.

---

## Design Principles

### Why "Feynman"?

Richard Feynman's learning technique: (1) pick a concept, (2) explain it in simple language, (3) identify gaps in your explanation, (4) go back to the source material. This system operationalizes that cycle:

- **ELI5** forces simple explanation (step 2)
- **Deep Research** and **Literature Review** are structured source reading (step 4)
- **Source Comparison** and **Peer Review** expose gaps (step 3)
- **Paper Writing** is the ultimate test of understanding (step 2, harder mode)
- **Graphify** makes the connections visible so you can see what you still don't understand

### Why Knowledge Graphs?

Reading is linear. Understanding is not. When you research a topic across 20 papers, 10 blog posts, and 5 videos, the connections between ideas are invisible unless you build the map yourself. Graphify builds that map automatically and honestly (every edge tagged with its provenance).

The community detection is especially valuable — it finds clusters of related concepts and, more importantly, the **bridge nodes** that connect clusters. These bridges are often the most interesting insights: the unexpected connection between two subfields, the shared technique used in different contexts, the common assumption that different authors make.

### Why Persistent Artifacts?

Every skill saves its output to disk. Nothing lives only in the conversation context. This means:

- **You can review later.** Research briefs, comparisons, and reviews are markdown files you can read, edit, and share.
- **You can graphify later.** Collect a week's worth of research into a folder and build the graph when you're ready.
- **You can query later.** Graphify's `graph.json` persists across sessions. Ask questions about research you did months ago.
- **You can chain skills.** The output of one skill is a file that another skill can read. Literature review feeds paper writing. Deep research feeds source comparison. Everything feeds the graph.

### Why Verification?

Every Feynman skill has built-in claim verification:

- **Deep Research**: Every critical finding needs 2+ independent sources. Single-source claims are explicitly flagged.
- **Paper Search**: Citations are verified against actual databases, not hallucinated.
- **Peer Review**: Referenced papers are checked for existence and accuracy.
- **Graphify**: Every edge is tagged EXTRACTED / INFERRED / AMBIGUOUS.

This isn't academic paranoia — it's how you build a knowledge base you can actually trust. If you can't tell which claims are well-supported and which are speculative, the whole system is unreliable.

---

## Example: Learning "Retrieval-Augmented Generation" From Scratch

Here's what a full learning cycle looks like in practice.

**1. Orientation** (15 min)

```
/feynman-eli5 retrieval-augmented generation
```

Read the output. Now you know what RAG is, why it matters, and what to be skeptical of.

**2. Find the papers** (10 min)

```
/feynman-paper-search retrieval-augmented generation
```

You get a table of 20+ papers. The original RAG paper (Lewis et al., 2020) is there, plus recent work on chunking strategies, hybrid search, and evaluation.

**3. Deep dive** (45 min)

```
/feynman-deep-research retrieval-augmented generation architectures and evaluation
```

Parallel subagents investigate: embedding models, chunking strategies, retrieval methods (dense vs. sparse vs. hybrid), generation quality metrics, hallucination rates. You get a 3000-word brief with inline citations.

**4. Compare approaches** (20 min)

```
/feynman-source-comparison naive RAG vs advanced RAG vs modular RAG
```

A comparison matrix shows where they agree (retrieval improves factuality), where they disagree (optimal chunk size, reranking necessity), and what's still unsettled.

**5. Build the graph** (5 min)

```bash
mkdir ~/Desktop/learning/rag
mv ~/Desktop/rag-*.md ~/Desktop/learning/rag/
```

```
/graphify ~/Desktop/learning/rag
```

Open `graph.html` in your browser. You see clusters: "Embedding Models", "Chunking Strategies", "Evaluation Metrics", "Hybrid Search". Bridge nodes reveal that chunk overlap strategy and embedding model choice are tightly coupled — something no single paper stated explicitly but the graph discovered from cross-referencing.

**6. Query it next week** (anytime)

```
/graphify query "What evaluation metrics are used for RAG systems?"
```

The graph answers from your accumulated research without re-reading anything.

**7. Grow it** (ongoing)

Found a new paper? Add it:

```
/graphify add https://arxiv.org/abs/2407.xxxxx
```

The graph updates incrementally. New connections appear. Your knowledge base compounds.

---

## Quick Reference

| I want to... | Run this |
|--------------|----------|
| Get a quick intuition for a concept | `/feynman-eli5 <concept>` |
| Find academic papers on a topic | `/feynman-paper-search <query>` |
| Do a thorough research investigation | `/feynman-deep-research <topic>` |
| Map an entire research field | `/feynman-literature-review <topic>` |
| Compare multiple approaches or sources | `/feynman-source-comparison <sources>` |
| Critique a paper or proposal | `/feynman-peer-review <file or arXiv ID>` |
| Write up my understanding as a paper | `/feynman-paper-writing <thesis>` |
| Build a knowledge graph from my research | `/graphify <folder>` |
| Add a new source to my graph | `/graphify add <URL>` |
| Query my knowledge base | `/graphify query "<question>"` |
| Find how two concepts connect | `/graphify path "A" "B"` |
| Get a plain explanation of a graph node | `/graphify explain "Node"` |
