# CogniMem Architecture

```mermaid
graph TB
    subgraph "User Interface"
        CHAT["Chat UI (SSE Streaming)"]
        DASH["Dashboard"]
        API["REST API"]
    end

    subgraph "Agent Engine"
        AGENT["Agent Loop<br/>Think → Act → Observe"]
        TOOLS["12 Tools<br/>web_search/web_fetch<br/>read_file/write_file/shell<br/>memory_remember/recall<br/>memory_diagnose/forget"]
        REFLECT["SelfReflector<br/>Auto Error Correction"]
        PLANNER["Goal-Driven Planner<br/>Task Decomposition"]
    end

    subgraph "CogniMem Brain"
        EXTRACT["Triple Extractor<br/>Rule-based + LLM"]
        FNET["Fact Network<br/>SPO Storage"]
        ROUTER["Recall Router<br/>L0 Cache → L1 Exact<br/>→ L1.5 BM25 → L2 Tags → L3 Vector"]
        GOV["Memory Governance<br/>6-Signal Scoring<br/>Type Diversity<br/>Blocked/Graduated"]
        CONTRADICT["Contradiction Detector<br/>Deny/Conflict/Context"]
        CONSOL["Consolidation<br/>Ebbinghaus Decay<br/>Abstraction<br/>Graduation"]
    end

    subgraph "Persistence"
        PG[("PostgreSQL<br/>facts / contradictions<br/>versions / audit_log")]
        STM[("Short-Term Memory<br/>Sliding Window Cache")]
    end

    subgraph "LLM Providers"
        DS["DeepSeek API"]
        QW["Qwen Cloud (DashScope)"]
    end

    CHAT --> AGENT
    DASH --> API
    API --> AGENT
    AGENT --> TOOLS
    AGENT --> REFLECT
    AGENT --> PLANNER
    AGENT --> EXTRACT
    EXTRACT --> FNET
    FNET --> ROUTER
    FNET --> CONTRADICT
    FNET --> GOV
    GOV --> CONSOL
    ROUTER --> STM
    CONSOL --> PG
    CONTRADICT --> PG
    FNET --> PG
    AGENT --> DS
    AGENT --> QW
```

## Data Flow

```
User Message
    │
    ▼
┌─────────────────────┐
│  Simple or Complex?  │
│  (rule-based check)  │
└──────┬──────┬───────┘
       │      │
    Simple  Complex
       │      │
       ▼      ▼
  ┌────────┐  ┌──────────────────┐
  │ Direct  │  │ Agent Loop       │
  │ LLM     │  │ 1. Recall Memory │
  │ Reply   │  │ 2. Plan Steps    │
  └────────┘  │ 3. Execute Tools  │
              │ 4. Store Results  │
              │ 5. Self-Reflect   │
              └──────────────────┘
                      │
                      ▼
              ┌──────────────────┐
              │  Consolidation   │
              │ • Ebbinghaus     │
              │ • Contradiction  │
              │ • Graduation     │
              └──────────────────┘
```

## Memory Lifecycle

```
New Info → Extract Triple → Score (6 signals) → Store
    ↓
Cache (L0) → Exact Match (L1) → BM25 (L1.5) → Tags (L2) → Vector (L3)
    ↓
After session: Consolidation → Ebbinghaus Decay → Contradiction Scan
    ↓
After 7+ days unused + 10+ accesses: Graduation (retired from context)
    ↓
Contradicted 3+ times OR confidence < 0.15: Blocked
```

## Deployment

```
┌──────────────┐     HTTPS/SSE      ┌──────────────────┐
│  User Browser │ ◄──────────────► │  Alibaba Cloud    │
│  (Chat UI)    │                   │  ECS (47.99.x.x)  │
└──────────────┘                   │  Port 8000        │
                                    │  systemd service  │
                                    │  PostgreSQL 13    │
                                    └──────────────────┘
```
