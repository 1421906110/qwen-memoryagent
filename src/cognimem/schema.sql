-- ================================================================
-- CogniMem Schema — CockroachDB
-- AI Agent 认知记忆系统
-- ================================================================

-- ── 事实表 (Fact Network 核心) ──
CREATE TABLE facts (
    fact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id STRING NOT NULL,
    subject STRING NOT NULL,          -- 主体: "用户"
    predicate STRING NOT NULL,        -- 谓词: "喜欢"
    object STRING NOT NULL,           -- 客体: "冰美式"

    -- 元数据
    fact_type STRING NOT NULL DEFAULT 'general',  -- preference | fact | goal | decision | observation | skill
    confidence FLOAT NOT NULL DEFAULT 0.6,
    importance FLOAT NOT NULL DEFAULT 0.5,

    -- 证据链
    evidence JSONB DEFAULT '[]'::JSONB,  -- [{"source":"session_xxx","statement":"..."}]
    contradictions JSONB DEFAULT '[]'::JSONB,  -- [fact_id, ...]
    connected_facts JSONB DEFAULT '[]'::JSONB,  -- [fact_id, ...]

    -- 上下文
    context_tags STRING[] DEFAULT ARRAY[],
    source_session STRING,

    -- 时序
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_confirmed TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count INT NOT NULL DEFAULT 1,
    expires_at TIMESTAMPTZ,

    -- 编码层级: raw | compressed | core
    encoding_level STRING NOT NULL DEFAULT 'raw',

    -- 去重约束: 同一 agent 内相同三元组只存一次
    UNIQUE INDEX idx_fact_triple (agent_id, subject, predicate, object),

    -- 检索索引
    INDEX idx_fact_agent_type (agent_id, fact_type),
    INDEX idx_fact_confidence (agent_id, confidence DESC),
    INDEX idx_fact_subject (agent_id, subject),
    INDEX idx_fact_tags (agent_id, context_tags),
    INDEX idx_fact_accessed (agent_id, accessed_at DESC),
    INDEX idx_fact_expires (agent_id, expires_at) WHERE expires_at IS NOT NULL
);


-- ── 矛盾记录表 ──
CREATE TABLE contradictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id STRING NOT NULL,
    fact_a_id UUID NOT NULL REFERENCES facts(fact_id),
    fact_b_id UUID NOT NULL REFERENCES facts(fact_id),

    -- 矛盾详情
    description STRING NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolution STRING NOT NULL DEFAULT 'pending',  -- pending | resolved_a | resolved_b | both_false

    INDEX idx_contradiction_agent (agent_id, resolution),
    INDEX idx_contradiction_fact (fact_a_id, fact_b_id)
);


-- ── 时序事件表 (Episodic Memory) ──
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id STRING NOT NULL,
    session_id STRING,
    summary STRING NOT NULL,           -- 事件摘要
    fact_refs UUID[] DEFAULT ARRAY[],  -- 关联的事实
    importance FLOAT DEFAULT 0.5,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    INDEX idx_episode_agent (agent_id, occurred_at DESC)
);


-- ── 工作记忆缓存 (L0) ──
-- 这个存在 Redis 或内存里，表结构仅用于持久化快照
CREATE TABLE working_memory_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id STRING NOT NULL,
    snapshot JSONB NOT NULL,
    saved_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    INDEX idx_wm_agent (agent_id, saved_at DESC)
);


-- ── 信念日志 (Confidence History) ──
CREATE TABLE confidence_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_id UUID NOT NULL REFERENCES facts(fact_id),
    agent_id STRING NOT NULL,
    old_confidence FLOAT,
    new_confidence FLOAT NOT NULL,
    reason STRING NOT NULL,           -- confirmed|contradicted|decayed|consolidated
    changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    INDEX idx_conf_fact (fact_id, changed_at DESC)
);


-- ── Agent 元数据 ──
CREATE TABLE agents (
    agent_id STRING PRIMARY KEY,
    name STRING,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT now(),
    config JSONB DEFAULT '{}'::JSONB
);
