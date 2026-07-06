"""
CogniMem 数据库适配器 — PostgreSQL / CockroachDB

提供 FactNetwork 所需的 db_adapter 接口。
本地用 PostgreSQL，CockroachDB Serverless 换连接串即可。
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.pool

from .models import FactTriple, Contradiction, EvidenceItem, Episode

logger = logging.getLogger(__name__)

# ── timezone 感知 ──
psycopg2.extras.register_default_jsonb(loads=json.loads)


class DatabaseAdapter:
    """
    CockroachDB / PostgreSQL 适配器

    使用方法:
        db = DatabaseAdapter(dsn="postgresql://...")
        db.connect()
        db.create_tables()   # 首次部署
    """

    def __init__(self, dsn: str = "", pool_min: int = 1, pool_max: int = 5):
        self.dsn = dsn or "postgresql://localhost/cognimem"
        self._pool: psycopg2.pool.ThreadedConnectionPool | None = None

    # ═══════════════════════════════════════════
    # 连接管理
    # ═══════════════════════════════════════════

    def connect(self):
        """创建连接池"""
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=self._pool_min,
            maxconn=self._pool_max,
            dsn=self.dsn,
        )
        logger.info("Database pool connected: %s", self.dsn)

    def close(self):
        """关闭连接池"""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Database pool closed")

    @property
    def _pool_min(self) -> int:
        return 1

    @property
    def _pool_max(self) -> int:
        return 5

    def _get_conn(self):
        if not self._pool:
            self.connect()
        return self._pool.getconn()

    def _put_conn(self, conn):
        if self._pool and conn:
            self._pool.putconn(conn)

    @staticmethod
    def _is_conn_alive(conn) -> bool:
        """健康检查：用 SELECT 1 验证连接是否存活"""
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _get_healthy_conn(self):
        """获取健康连接 — 自动检测并修复坏连接

        ⭐ v0.10 稳定性: PostgreSQL 重启/网络抖动后自动恢复，
        避免坏连接在池中毒化。
        """
        conn = self._get_conn()
        if not self._is_conn_alive(conn):
            logger.warning("⚠️ 连接已断开，丢弃坏连接并重建...")
            # 坏连接：从池中移除（不 putconn 回去，避免毒化池）
            try:
                if hasattr(self._pool, '_used'):
                    self._pool._used.discard(conn)
            except (ValueError, AttributeError, TypeError):
                pass
            try:
                conn.close()
            except Exception:
                pass
            # 拿个新连接
            conn = self._get_conn()
            if not self._is_conn_alive(conn):
                logger.error("❌ 新连接也不可用，尝试重建连接池...")
                self.close()
                self.connect()
                conn = self._get_conn()
        return conn

    @contextmanager
    def _conn_ctx(self):
        """安全连接上下文 — 自动健康检查 + 异常时归还连接"""
        conn = self._get_healthy_conn()
        try:
            yield conn
            conn.commit()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            conn.rollback()
            logger.warning("⚠️ 数据库操作错误，丢弃坏连接: %s", e)
            try:
                if hasattr(self._pool, '_used'):
                    self._pool._used.discard(conn)
            except Exception:
                pass
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    @contextmanager
    def _cursor_ctx(self):
        """游标上下文 — 自动健康检查 + 坏连接自动修复"""
        conn = self._get_healthy_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                yield cur
            conn.commit()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            conn.rollback()
            logger.warning("⚠️ 数据库操作错误（连接层），丢弃坏连接: %s", e)
            try:
                if hasattr(self._pool, '_used'):
                    self._pool._used.discard(conn)
            except Exception:
                pass
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    @contextmanager
    def _plain_cursor_ctx(self):
        """无 DictCursor 的游标上下文 — 自动健康检查"""
        conn = self._get_healthy_conn()
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            conn.rollback()
            logger.warning("⚠️ 数据库操作错误（连接层），丢弃坏连接: %s", e)
            try:
                if hasattr(self._pool, '_used'):
                    self._pool._used.discard(conn)
            except Exception:
                pass
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    # ═══════════════════════════════════════════
    # DDL
    # ═══════════════════════════════════════════

    def create_tables(self):
        """执行 schema.pg.sql 建表"""
        import os
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "schema.pg.sql"
        )
        with open(schema_path) as f:
            sql = f.read()
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Tables created from schema.pg.sql")
        finally:
            self._put_conn(conn)

    # ═══════════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════════

    def _row_to_fact(self, row: tuple, cursor) -> FactTriple | None:
        """数据库行 → FactTriple"""
        if not row:
            return None
        col_names = [desc[0] for desc in cursor.description]
        d = dict(zip(col_names, row))
        return self._dict_to_fact(d)

    def _dict_to_fact(self, d: dict) -> FactTriple:
        """字典 → FactTriple"""
        evidence = []
        for ev in d.get("evidence", []) or []:
            if isinstance(ev, dict):
                evidence.append(EvidenceItem(
                    source=ev.get("source", ""),
                    statement=ev.get("statement", ""),
                    timestamp=ev.get("timestamp", ""),
                ))

        def _ts(val: Any) -> str:
            if isinstance(val, datetime):
                return val.isoformat()
            return str(val) if val else ""

        return FactTriple(
            fact_id=str(d["fact_id"]),
            agent_id=d["agent_id"],
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            fact_type=d.get("fact_type", "general"),
            confidence=float(d.get("confidence", 0.6)),
            importance=float(d.get("importance", 0.5)),
            encoding_level=d.get("encoding_level", "raw"),
            evidence=evidence,
            contradictions=list(d.get("contradictions", []) or []),
            connected_facts=list(d.get("connected_facts", []) or []),
            context_tags=list(d.get("context_tags", []) or []),
            source_session=d.get("source_session", ""),
            created_at=_ts(d.get("created_at")),
            accessed_at=_ts(d.get("accessed_at")),
            last_confirmed=_ts(d.get("last_confirmed")),
            access_count=int(d.get("access_count", 1)),
            expires_at=_ts(d.get("expires_at")) if d.get("expires_at") else "",
        )

    def _fact_to_row(self, fact: FactTriple) -> dict:
        """FactTriple → 数据库行字典"""
        evidence_json = []
        for ev in fact.evidence:
            if isinstance(ev, EvidenceItem):
                evidence_json.append({
                    "source": ev.source,
                    "statement": ev.statement,
                    "timestamp": ev.timestamp,
                })
            elif isinstance(ev, dict):
                evidence_json.append(ev)

        def _parse_ts(ts: str) -> datetime | None:
            if not ts:
                return None
            if isinstance(ts, datetime):
                return ts
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                return datetime.now(timezone.utc)

        return {
            "fact_id": fact.fact_id,
            "agent_id": fact.agent_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "fact_type": fact.fact_type,
            "confidence": fact.confidence,
            "importance": fact.importance,
            "encoding_level": fact.encoding_level,
            "evidence": json.dumps(evidence_json, ensure_ascii=False),
            "contradictions": json.dumps(list(fact.contradictions)),
            "connected_facts": json.dumps(list(fact.connected_facts)),
            "context_tags": list(fact.context_tags),
            "source_session": fact.source_session,
            "created_at": _parse_ts(fact.created_at),
            "accessed_at": _parse_ts(fact.accessed_at),
            "last_confirmed": _parse_ts(fact.last_confirmed),
            "access_count": fact.access_count,
            "expires_at": _parse_ts(fact.expires_at) if fact.expires_at else None,
        }

    # ═══════════════════════════════════════════
    # FactNetwork 接口方法
    # ═══════════════════════════════════════════

    def save_version(self, fact_id: str, agent_id: str,
                     old_fact: FactTriple | None,
                     new_fact: FactTriple, reason: str):
        """保存事实版本快照"""
        if old_fact is None:
            return
        with self._plain_cursor_ctx() as cur:
            cur.execute("""
                INSERT INTO fact_versions
                    (fact_id, agent_id,
                     old_subject, old_predicate, old_object,
                     old_confidence, old_importance, old_encoding_level,
                     new_subject, new_predicate, new_object,
                     new_confidence, new_importance, new_encoding_level,
                     change_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s)
            """, (
                fact_id, agent_id,
                old_fact.subject, old_fact.predicate, old_fact.object,
                old_fact.confidence, old_fact.importance, old_fact.encoding_level,
                new_fact.subject, new_fact.predicate, new_fact.object,
                new_fact.confidence, new_fact.importance, new_fact.encoding_level,
                reason,
            ))

    def get_versions(self, fact_id: str) -> list[dict]:
        """获取事实的版本历史"""
        with self._cursor_ctx() as cur:
            cur.execute("""
                SELECT * FROM fact_versions
                WHERE fact_id = %s
                ORDER BY changed_at DESC
            """, (fact_id,))
            return [dict(row) for row in cur.fetchall()]

    def save_fact(self, fact: FactTriple):
        """插入新事实"""
        row = self._fact_to_row(fact)
        with self._plain_cursor_ctx() as cur:
            cur.execute("""
                INSERT INTO facts (
                    fact_id, agent_id, subject, predicate, object,
                    fact_type, confidence, importance, encoding_level,
                    evidence, contradictions, connected_facts,
                    context_tags, source_session,
                    created_at, accessed_at, last_confirmed,
                    access_count, expires_at
                ) VALUES (
                    %(fact_id)s, %(agent_id)s, %(subject)s, %(predicate)s, %(object)s,
                    %(fact_type)s, %(confidence)s, %(importance)s, %(encoding_level)s,
                    %(evidence)s::JSONB, %(contradictions)s::JSONB, %(connected_facts)s::JSONB,
                    %(context_tags)s, %(source_session)s,
                    %(created_at)s, %(accessed_at)s, %(last_confirmed)s,
                    %(access_count)s, %(expires_at)s
                )
                ON CONFLICT (agent_id, subject, predicate, object)
                DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    importance = EXCLUDED.importance,
                    evidence = EXCLUDED.evidence,
                    contradictions = EXCLUDED.contradictions,
                    connected_facts = EXCLUDED.connected_facts,
                    context_tags = EXCLUDED.context_tags,
                    accessed_at = EXCLUDED.accessed_at,
                    access_count = EXCLUDED.access_count,
                    last_confirmed = EXCLUDED.last_confirmed
            """, row)
            try:
                txt = f"{fact.subject} {fact.predicate} {fact.object}"
                self.update_embedding(fact.fact_id, txt)
            except Exception:
                pass

    def update_fact(self, fact: FactTriple):
        """更新已有事实（合并后）"""
        row = self._fact_to_row(fact)
        with self._plain_cursor_ctx() as cur:
            cur.execute("""
                UPDATE facts SET
                    confidence = %(confidence)s,
                    importance = %(importance)s,
                    encoding_level = %(encoding_level)s,
                    evidence = %(evidence)s::JSONB,
                    contradictions = %(contradictions)s::JSONB,
                    connected_facts = %(connected_facts)s::JSONB,
                    context_tags = %(context_tags)s,
                    source_session = %(source_session)s,
                    accessed_at = %(accessed_at)s,
                    last_confirmed = %(last_confirmed)s,
                    access_count = %(access_count)s,
                    expires_at = %(expires_at)s
                WHERE fact_id = %(fact_id)s
            """, row)

    def find_by_triple_key(self, triple_key: str) -> FactTriple | None:
        """按 JSON 编码的三元组 key 查找"""
        try:
            parts = json.loads(triple_key)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parts, list) or len(parts) != 4:
            return None
        agent_id, subject, predicate, object_ = parts
        with self._cursor_ctx() as cur:
            cur.execute("""
                SELECT * FROM facts
                WHERE agent_id = %s AND subject = %s
                  AND predicate = %s AND "object" = %s
            """, (agent_id, subject, predicate, object_))
            row = cur.fetchone()
            return self._dict_to_fact(dict(row)) if row else None

    def get_fact(self, fact_id: str) -> FactTriple | None:
        """按 fact_id 获取事实"""
        try:
            with self._cursor_ctx() as cur:
                cur.execute("SELECT * FROM facts WHERE fact_id = %s", (fact_id,))
                row = cur.fetchone()
                return self._dict_to_fact(dict(row)) if row else None
        except Exception as e:
            logger.warning("get_fact error for %s: %s", fact_id, e)
            return None

    def get_agent_facts(self, agent_id: str) -> list[FactTriple]:
        """获取某 agent 的所有事实"""
        with self._cursor_ctx() as cur:
            cur.execute(
                "SELECT * FROM facts WHERE agent_id = %s ORDER BY confidence DESC",
                (agent_id,)
            )
            return [self._dict_to_fact(dict(row)) for row in cur.fetchall()]

    def search_facts(self, agent_id: str, subject: str = "",
                     object: str = "", predicate: str = "",
                     tag: str = "",
                     limit: int = 20) -> list[FactTriple]:
        """
        灵活搜索事实。

        多个条件之间是 OR 关系：subject 匹配 OR object 匹配 OR
        predicate 匹配 OR tag 匹配。
        如果所有条件都为空，返回该 agent 所有事实。
        """
        with self._cursor_ctx() as cur:
            params: dict[str, Any] = {"agent_id": agent_id, "limit": limit}
            filters: list[str] = []

            if subject:
                filters.append("subject ILIKE %(subject)s")
                esc = subject.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
                params["subject"] = f"%{esc}%"
            if predicate:
                filters.append("predicate ILIKE %(predicate)s")
                esc = predicate.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
                params["predicate"] = f"%{esc}%"
            if object:
                filters.append('"object" ILIKE %(object)s')
                esc = object.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
                params["object"] = f"%{esc}%"
            if tag:
                filters.append("%(tag)s = ANY(context_tags)")
                params["tag"] = tag

            if filters:
                where = f"agent_id = %(agent_id)s AND ({' OR '.join(filters)})"
            else:
                where = "agent_id = %(agent_id)s"

            sql = f"SELECT * FROM facts WHERE {where} ORDER BY confidence DESC LIMIT %(limit)s"
            cur.execute(sql, params)
            return [self._dict_to_fact(dict(row)) for row in cur.fetchall()]

    def delete_fact(self, fact_id: str):
        """删除事实（含级联的矛盾、版本、日志）"""
        with self._plain_cursor_ctx() as cur:
            cur.execute("DELETE FROM contradictions WHERE fact_a_id = %s OR fact_b_id = %s",
                        (fact_id, fact_id))
            cur.execute("DELETE FROM confidence_log WHERE fact_id = %s", (fact_id,))
            cur.execute("DELETE FROM fact_versions WHERE fact_id = %s", (fact_id,))
            cur.execute("DELETE FROM facts WHERE fact_id = %s", (fact_id,))

    # ═══════════════════════════════════════════
    # 矛盾记录
    # ═══════════════════════════════════════════

    def save_contradiction(self, c: Contradiction):
        """保存矛盾记录"""
        with self._plain_cursor_ctx() as cur:
            cur.execute("""
                INSERT INTO contradictions (id, agent_id, fact_a_id, fact_b_id,
                                           description, contradiction_type,
                                           detected_at, resolution)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fact_a_id, fact_b_id) DO NOTHING
            """, (c.id, c.agent_id, c.fact_a_id, c.fact_b_id,
                  c.description, c.contradiction_type,
                  c.detected_at, c.resolution))

    def get_contradictions(self, agent_id: str,
                           resolution: str = "pending") -> list[Contradiction]:
        """获取待处理的矛盾"""
        with self._cursor_ctx() as cur:
            cur.execute("""
                SELECT * FROM contradictions
                WHERE agent_id = %s AND resolution = %s
                ORDER BY detected_at DESC
            """, (agent_id, resolution))
            return [
                Contradiction(
                    id=str(row["id"]),
                    agent_id=row["agent_id"],
                    fact_a_id=str(row["fact_a_id"]),
                    fact_b_id=str(row["fact_b_id"]),
                    description=row["description"],
                    contradiction_type=row.get("contradiction_type", "deny"),
                    detected_at=row["detected_at"].isoformat(),
                    resolution=row["resolution"],
                )
                for row in cur.fetchall()
            ]

    # ═══════════════════════════════════════════
    # 置信度日志
    # ═══════════════════════════════════════════

    def log_confidence_change(self, fact_id: str, agent_id: str,
                               old_conf: float, new_conf: float, reason: str):
        """记录置信度变更"""
        with self._plain_cursor_ctx() as cur:
            cur.execute("""
                INSERT INTO confidence_log
                    (fact_id, agent_id, old_confidence, new_confidence, reason)
                VALUES (%s, %s, %s, %s, %s)
            """, (fact_id, agent_id, old_conf, new_conf, reason))

    # ═══════════════════════════════════════════
    # Agent 管理
    # ═══════════════════════════════════════════

    def ensure_agent(self, agent_id: str, name: str = ""):
        """确保 agent 记录存在"""
        with self._plain_cursor_ctx() as cur:
            cur.execute("""
                INSERT INTO agents (agent_id, name, last_active)
                VALUES (%s, %s, now())
                ON CONFLICT (agent_id) DO UPDATE SET
                    last_active = now(),
                    name = COALESCE(NULLIF(%s, ''), agents.name)
            """, (agent_id, name, name))

    # ═══════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════

    def get_stats(self, agent_id: str) -> dict:
        """获取 agent 的数据库统计"""
        with self._cursor_ctx() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total_facts,
                    COUNT(*) FILTER (WHERE confidence >= 0.9) AS core_beliefs,
                    COUNT(*) FILTER (WHERE confidence < 0.2) AS unreliable,
                    AVG(confidence)::FLOAT AS avg_confidence,
                    COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at < now()) AS expired
                FROM facts WHERE agent_id = %s
            """, (agent_id,))
            row = cur.fetchone()
            return dict(row) if row else {}

    # ═══════════════════════════════════════════
    # 向量搜索 (pgvector)
    # ═══════════════════════════════════════════

    def compute_embedding(self, text: str) -> list[float]:
        """纯 Python n-gram 哈希嵌入，不依赖外部模型"""
        import hashlib
        dim = 384
        vec = [0.0] * dim
        text = text.lower().strip()
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i:i+n]
                h = hashlib.md5(gram.encode()).digest()
                idx = int.from_bytes(h[:4], 'little') % dim
                vec[idx] += 1.0
        mag = sum(x*x for x in vec) ** 0.5
        if mag > 0:
            vec = [x/mag for x in vec]
        return vec

    def update_embedding(self, fact_id: str, text: str):
        """计算并存储 embedding"""
        with self._plain_cursor_ctx() as cur:
            cur.execute(
                "UPDATE facts SET embedding = %s WHERE fact_id = %s",
                (self.compute_embedding(text), fact_id)
            )

    def search_facts_vector(self, agent_id: str, query: str,
                            limit: int = 10) -> list[FactTriple]:
        """向量相似度搜索

        使用纯 Python n-gram 哈希 embedding（无外部模型依赖）。
        哈希嵌入的余弦相似度天然偏低（0.5+ 才算强匹配），
        所以阈值设为 0.10 以捕获弱语义关联。
        L3 是最后兜底层，宁可低阈值多召回，由上层 ranking 来过滤。
        """
        vec = self.compute_embedding(query)
        try:
            with self._cursor_ctx() as cur:
                cur.execute("""
                    SELECT *, 1 - (embedding <=> %s::vector) AS similarity
                    FROM facts
                    WHERE agent_id = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vec, agent_id, vec, limit))
                return [
                    self._dict_to_fact(dict(r))
                    for r in cur.fetchall()
                    if r.get("similarity", 0) > 0.10
                ]
        except Exception as e:
            logger.error("[L3] vector search failed: %s", e)
            return []
