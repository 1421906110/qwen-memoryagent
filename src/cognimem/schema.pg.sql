-- ================================================================
-- CogniMem Schema — PostgreSQL (本地开发用)
-- AI Agent 认知记忆系统
-- ================================================================

-- ── pgvector 扩展（向量搜索 L3 依赖）──
CREATE EXTENSION IF NOT EXISTS vector;

-- ── 事实表 (Fact Network 核心) ──
CREATE TABLE facts (
    fact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    subject TEXT NOT NULL,          -- 主体: "用户"
    predicate TEXT NOT NULL,        -- 谓词: "喜欢"
    object TEXT NOT NULL,           -- 客体: "冰美式"

    -- 元数据
    fact_type TEXT NOT NULL DEFAULT 'general',  -- preference | fact | goal | decision | observation | skill
    confidence FLOAT NOT NULL DEFAULT 0.6,
    importance FLOAT NOT NULL DEFAULT 0.5,

    -- 证据链
    evidence JSONB DEFAULT '[]'::JSONB,
    contradictions JSONB DEFAULT '[]'::JSONB,
    connected_facts JSONB DEFAULT '[]'::JSONB,

    -- 上下文
    context_tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    source_session TEXT,

    -- 时序
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_confirmed TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count INT NOT NULL DEFAULT 1,
    expires_at TIMESTAMPTZ,

    -- 编码层级: raw | compressed | core
    encoding_level TEXT NOT NULL DEFAULT 'raw',

    -- 向量嵌入 (L3 向量搜索，384维 n-gram 哈希)
    embedding vector(384),

    -- 去重约束
    UNIQUE (agent_id, subject, predicate, object)
);

-- 检索索引
CREATE INDEX idx_fact_agent_type ON facts (agent_id, fact_type);
CREATE INDEX idx_fact_confidence ON facts (agent_id, confidence DESC);
CREATE INDEX idx_fact_subject ON facts (agent_id, subject);
CREATE INDEX idx_fact_tags ON facts (agent_id, context_tags);
CREATE INDEX idx_fact_accessed ON facts (agent_id, accessed_at DESC);
CREATE INDEX idx_fact_expires ON facts (agent_id, expires_at) WHERE expires_at IS NOT NULL;

-- 向量搜索索引 (ivfflat，适合 <100K 条数据场景)
CREATE INDEX idx_fact_embedding ON facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);


-- ── 矛盾记录表 ──
CREATE TABLE contradictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    fact_a_id UUID NOT NULL REFERENCES facts(fact_id),
    fact_b_id UUID NOT NULL REFERENCES facts(fact_id),

    -- 矛盾详情
    description TEXT NOT NULL,
    contradiction_type TEXT NOT NULL DEFAULT 'deny',  -- deny(直接否定) | conflict(间接冲突) | context(上下文变化)
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolution TEXT NOT NULL DEFAULT 'pending',  -- pending | resolved_a | resolved_b | both_false

    UNIQUE (fact_a_id, fact_b_id)
);
CREATE INDEX idx_contradiction_agent ON contradictions (agent_id, resolution);
CREATE INDEX idx_contradiction_fact ON contradictions (fact_a_id, fact_b_id);


-- ── 时序事件表 (Episodic Memory) ──
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    session_id TEXT,
    summary TEXT NOT NULL,
    fact_refs UUID[] DEFAULT ARRAY[]::UUID[],
    importance FLOAT DEFAULT 0.5,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_episode_agent ON episodes (agent_id, occurred_at DESC);


-- ── 工作记忆缓存快照 (L0 持久化) ──
CREATE TABLE working_memory_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    snapshot JSONB NOT NULL,
    saved_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_wm_agent ON working_memory_snapshots (agent_id, saved_at DESC);


-- ── 信念日志 (Confidence History) ──
CREATE TABLE confidence_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_id UUID NOT NULL REFERENCES facts(fact_id),
    agent_id TEXT NOT NULL,
    old_confidence FLOAT,
    new_confidence FLOAT NOT NULL,
    reason TEXT NOT NULL,           -- confirmed|contradicted|decayed|consolidated
    changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_conf_fact ON confidence_log (fact_id, changed_at DESC);


-- ── Agent 元数据 ──
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT now(),
    config JSONB DEFAULT '{}'::JSONB
);

-- ── 事实版本链 ──
CREATE TABLE fact_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_id UUID NOT NULL REFERENCES facts(fact_id),
    agent_id TEXT NOT NULL,
    old_subject TEXT,
    old_predicate TEXT,
    old_object TEXT,
    old_confidence FLOAT,
    old_importance FLOAT,
    old_encoding_level TEXT,
    new_subject TEXT,
    new_predicate TEXT,
    new_object TEXT,
    new_confidence FLOAT NOT NULL,
    new_importance FLOAT NOT NULL,
    new_encoding_level TEXT,
    change_reason TEXT NOT NULL,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_fv_fact ON fact_versions (fact_id, changed_at DESC);


-- ── 审计日志表 (Audit Trail) ──
-- 受 DREAM 审计日志启发，记录所有记忆操作的完整追踪。
-- 每条记录：谁操作了哪条记忆、做了什么、什么时候做的。
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    fact_id UUID,                              -- 操作的目标事实（可选，治理操作无特定事实）
    operation TEXT NOT NULL,                    -- create|read|update|delete|confirm|challenge|contradiction|governance|consolidation
    detail TEXT NOT NULL,                       -- 操作详情（自然语言描述）
    metadata JSONB DEFAULT '{}'::JSONB,         -- 附加数据（旧值/新值/来源等）
    caller TEXT DEFAULT '',                     -- 调用方标识（api|mcp|agent|system|user）
    ip_address TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_audit_agent ON audit_log (agent_id, created_at DESC);
CREATE INDEX idx_audit_fact ON audit_log (fact_id, created_at DESC);
CREATE INDEX idx_audit_operation ON audit_log (operation, created_at DESC);
-- 按时间范围查询（最常用）
CREATE INDEX idx_audit_created ON audit_log (created_at DESC);
