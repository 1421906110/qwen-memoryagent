# CogniMem — Cognitive Memory Agent

> **A memory system that thinks like a human brain — not just storage, but understanding, forgetting, questioning, and abstracting.**

---

## What is CogniMem?

CogniMem is not a chatbot with a vector database. It's a **cognitive memory system** that stores everything as structured **fact triples** (subject-predicate-object) instead of raw text. It remembers, forgets, questions contradictions, and abstracts fragments into knowledge — just like how human memory works.

## Track

**MemoryAgent** — Building an intelligent agent with persistent memory.

## Who Built It

- **Xiaoqi (baikai)** — Full-stack development, architecture, deployment
- A solo project built for the Global AI Hackathon Series with Qwen Cloud

## Key Innovations

### 1. 🧠 Structured Triple Memory (SPO)
Not "the user likes iced American coffee" as text — but `(user, likes, iced American coffee)` as a structured relationship with confidence, evidence chain, and importance. This enables contradiction detection, abstraction, and intelligent forgetting that raw-text systems cannot do.

### 2. 🔄 Five-Level Recall Router
| Level | Method | Speed | Cost |
|-------|--------|-------|------|
| L0 | Semantic cache | <1ms | 0 token |
| L1 | Exact triple match | <3ms | 0 token |
| L1.5 | BM25 fuzzy match | <5ms | 0 token |
| L2 | Semantic expansion (tags/topics) | <10ms | 0 token |
| L3 | Pure-Python vector similarity | <20ms | 0 token |

The router picks the fastest path first — cache → exact → fuzzy → semantic → vector. Most queries are satisfied at L0/L1 with zero LLM calls and zero token cost.

### 3. 🎓 Memory Governance (6-Signal Scoring)
Every memory is scored on 6 signals before entering context:
- **Importance × Confidence** (base relevance)
- **Type priority** (preference > fact > goal > observation > action)
- **Ebbinghaus decay** (scientific forgetting curve)
- **Freshness** (recently accessed facts get a boost)
- **Contradiction penalty** (conflicting facts are demoted)
- **Relevance boost** (matches current query)

Blocked memories (credentials, ultra-low confidence, ≥3 contradictions) never enter context. Graduated memories (frequent + unused for 7+ days) auto-retire.

### 4. 🚀 Autonomous Agent Loop
Unlike most memory agents that are just REST APIs, CogniMem has a complete **Think → Act → Observe** loop with:
- 12 built-in tools (web search, file I/O, shell, memory operations, self-diagnosis)
- Goal-driven task planning
- Self-correction (SelfReflector fixes errors automatically)
- Completion marker system for clean task termination

### 5. 💡 Innovation: Contradiction-Driven Learning
CogniMem actively detects contradictions between stored facts. When contradictory information is found, the system flags it, adjusts confidence scores, and can challenge the user to clarify — driving cognitive improvement over time.

### 6. 🔬 Innovation: Pure-Python Vector Search
No external embedding models or vector databases required. Our multi-granularity n-gram hashing + cosine similarity achieves competitive semantic search with zero external dependencies. Works on any system with PostgreSQL.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Interface                     │
│  Chat (Streaming SSE) | Dashboard | REST API         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Agent Engine (Think→Act→Observe)         │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Planner │  │ Executor │  │ SelfReflector    │    │
│  │(goal    │→│(12 tools)│→│(auto-correct)    │    │
│  │ driven) │  │          │  │                  │    │
│  └─────────┘  └──────────┘  └──────────────────┘    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              CogniMem Brain (Core Engine)             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │Extractor │ │Fact      │ │Recall Router     │    │
│  │(rule+LLM)│→│Network   │→│(L0→L1→L2→L3)     │    │
│  └──────────┘ └──────────┘ └──────────────────┘    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │Memory    │ │Contradict│ │Consolidation     │    │
│  │Governance│ │ion Detect│ │(decay+graduate)  │    │
│  └──────────┘ └──────────┘ └──────────────────┘    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              PostgreSQL (Persistence)                 │
│  facts | contradictions | versions | audit_log       │
└─────────────────────────────────────────────────────┘
```

## Built With

- **Qwen Cloud (DashScope)** — LLM inference via OpenAI-compatible API
- **DeepSeek** — Alternative LLM provider
- **Python 3.10+** — Core language
- **FastAPI + Uvicorn** — Web server + SSE streaming
- **PostgreSQL** — Persistence layer
- **Alibaba Cloud ECS** — Production deployment
- **Jinja2 + vanilla JS** — Dashboard UI

## Deployment

Live demo: `http://47.99.151.253:8000/`
Source: `https://github.com/1421906110/qwen-memoryagent` (AGPL-3.0)

## What Makes CogniMem Different?

| Capability | CogniMem | Typical RAG Systems |
|------------|----------|-------------------|
| Memory structure | Triple SPO (structured) | Text chunks (unstructured) |
| Agent loop | Think→Act→Observe (autonomous) | Query→Response (passive) |
| Tools | 12 built-in tools | None |
| Forgetting | Ebbinghaus curve + graduation | Everything or nothing |
| Contradiction | Active detection + flagging | None |
| Self-correction | Auto-reflect on errors | None |
| Search cost | 0-token cache → BM25 → vector | Always vector (costly) |
| Embedding | Pure Python, no model needed | Requires external API/model |
