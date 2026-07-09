"""
Module System — 纯指令系统的沉淀引擎。

What happens here:
  1. AI handles NEW tasks (costly, flexible)
  2. CogniMem tracks HOW OFTEN each task type occurs
  3. When frequency passes a threshold → suggest making a MODULE
  4. Module = a named, reusable, zero-cost capability
  5. Modules auto-register as tools in the agent's ToolRegistry

Key concept from the research report:
                  沉淀率 = AI 任务 → 模块的转化率
  沉淀率越高 → AI 费用越低 → 系统越"聪明"（常用操作秒出）

Module file structure:
  modules/
  └── <category>/
      └── <module_name>.py   # Standard template

Module interface:

  # module_name.py
  \"\"\"metadata
  name: module_name
  description: What this module does
  params: {"param1": {"type": "string", "description": "..."}, ...}
  \"\"\"

  def run(params: dict) -> dict:
      \"\"\"Execute the module. Return a dict with results.\"\"\"
      ...
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from memory_agent.agent import ToolRegistry

if TYPE_CHECKING:
    from memory_agent.agent import AgentContext

logger = logging.getLogger("agent.modules")

# Default module directory
DEFAULT_MODULE_DIR = Path.home() / ".qwen-memory" / "modules"


# ---------------------------------------------------------------------------
#  Frequency tracking (via CogniMem or local SQLite)
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    """Record of an AI-handled task — used to determine when to 沉淀."""
    task_type: str          # Normalized category, e.g. "web_scrape", "data_analysis"
    description: str        # Brief description of what the task was
    timestamp: float
    session_id: str
    outcome: str = "success"
    num_tool_calls: int = 0
    module_name: str | None = None  # Set if this task was handled by a module


class FrequencyTracker:
    """Tracks how often task types occur and suggests 沉淀 (module creation).

    Uses CogniMem for persistent storage (via agent memory tools)
    and an in-memory window for real-time frequency calculation.
    """

    def __init__(self, cogni_client=None, agent_id: str = "default"):
        self.agent_id = agent_id
        self.cogni = cogni_client
        self._window: list[TaskRecord] = []  # Recent tasks in memory
        self._max_window = 1000

    def record_task(self, task_type: str, description: str,
                    outcome: str = "success", num_tool_calls: int = 0,
                    module_name: str | None = None) -> TaskRecord:
        """Record that a task of this type was handled (by AI or module)."""
        record = TaskRecord(
            task_type=task_type,
            description=description,
            timestamp=time.time(),
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            outcome=outcome,
            num_tool_calls=num_tool_calls,
            module_name=module_name,
        )

        self._window.append(record)
        # Keep window bounded
        if len(self._window) > self._max_window:
            self._window = self._window[-self._max_window:]

        # Persist to CogniMem
        if self.cogni:
            try:
                module_tag = f" via module:{module_name}" if module_name else " (AI handled)"
                self.cogni.remember(
                    text=f"[task_tracking] {task_type}: {description}{module_tag}",
                    agent_id=self.agent_id,
                    source=f"task_log:{task_type}",
                )
            except Exception as e:
                logger.warning("Failed to persist task record: %s", e)

        return record

    def get_frequency(self, task_type: str, window_hours: float = 72) -> int:
        """Count how many times a task type occurred in the time window."""
        cutoff = time.time() - (window_hours * 3600)
        return sum(
            1 for r in self._window
            if r.task_type == task_type and r.timestamp >= cutoff
        )

    def get_all_frequencies(self, window_hours: float = 72) -> dict[str, int]:
        """Get frequency for all task types seen in the window."""
        cutoff = time.time() - (window_hours * 3600)
        freq: dict[str, int] = {}
        for r in self._window:
            if r.timestamp >= cutoff:
                freq[r.task_type] = freq.get(r.task_type, 0) + 1
        return freq

    def suggest_modules(self, threshold: int = 3,
                        window_hours: float = 72) -> list[dict]:
        """Suggest task types that should become modules.

        Returns list of {task_type, frequency, description, reason}.
        """
        freq = self.get_all_frequencies(window_hours)
        suggestions = []
        for task_type, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count >= threshold:
                # Find the most recent description for context
                recent_desc = ""
                for r in reversed(self._window):
                    if r.task_type == task_type and r.description:
                        recent_desc = r.description[:100]
                        break

                suggestions.append({
                    "task_type": task_type,
                    "frequency": count,
                    "window_hours": window_hours,
                    "recent_example": recent_desc,
                    "reason": f"出现了 {count} 次，建议沉淀为模块",
                    "recommendation": (
                        f"**{task_type}** — {count} 次/{window_hours}h\n"
                        f"最近示例: {recent_desc}\n"
                        f"建议: 花 30 分钟写个模块，以后零成本执行"
                    ),
                })
        return suggestions

    def classify_task(self, ai_response: dict, user_message: str) -> str:
        """Classify an AI task into a normalized task type based on tools used.

        Args:
            ai_response: The response dict from Agent.chat()
            user_message: The original user message

        Returns:
            Normalized task type string, e.g. "web_scrape", "file_edit"
        """
        tools_used = ai_response.get("tools_called", 0)
        tool_seq = ai_response.get("tool_sequence", [])

        # Check what tools were used to classify the task
        tools_used_names = set()
        # Extract tool names from the sequence
        for item in tool_seq:
            if isinstance(item, str) and "🛠️" not in item:
                continue  # skip thinking text

        # Classify by user message content
        msg_lower = user_message.lower()

        if any(w in msg_lower for w in ["爬", "抓", "fetch", "scrape", "crawl"]):
            return "web_scrape"
        elif any(w in msg_lower for w in ["搜", "搜索", "search", "find", "查询"]):
            return "web_search"
        elif any(w in msg_lower for w in ["读文件", "看文件", "read", "打开文件"]):
            return "file_read"
        elif any(w in msg_lower for w in ["写文件", "写入", "write", "保存"]):
            return "file_write"
        elif any(w in msg_lower for w in ["改", "编辑", "edit", "修改"]):
            return "file_edit"
        elif any(w in msg_lower for w in ["跑命令", "运行", "run", "执行"]):
            return "shell_exec"
        elif any(w in msg_lower for w in ["分析", "analyze", "统计"]):
            return "data_analysis"
        elif any(w in msg_lower for w in ["对比", "比较", "compare", "diff"]):
            return "comparison"
        elif any(w in msg_lower for w in ["生成", "generate", "写", "write"]):
            return "content_generation"
        elif any(w in msg_lower for w in ["翻译", "translate"]):
            return "translation"
        else:
            return "general_ai"


# ---------------------------------------------------------------------------
#  Module Loader
# ---------------------------------------------------------------------------

class ModuleLoader:
    """Discovers and loads modules from the filesystem.

    Module directory structure:
      modules/
      ├── web/
      │   ├── scrape_jd_prices.py
      │   └── check_website_status.py
      ├── file/
      │   ├── batch_rename.py
      │   └── convert_format.py
      ├── data/
      │   ├── csv_to_json.py
      │   └── analyze_sales.py
      └── custom/
          └── my_workflow.py
    """

    def __init__(self, module_dir: str | Path = DEFAULT_MODULE_DIR):
        self.module_dir = Path(module_dir).expanduser()
        self.module_dir.mkdir(parents=True, exist_ok=True)

    def discover_modules(self) -> list[dict]:
        """Scan module directory and return metadata for all modules found.

        Each module file must have:
          - A module-level docstring with YAML-like metadata
          - A `run(params: dict) -> dict` function

        Returns list of {name, description, params, category, filepath, error}
        """
        modules = []
        for py_file in sorted(self.module_dir.rglob("*.py")):
            try:
                meta = self._parse_metadata(py_file)
                if meta:
                    modules.append(meta)
            except Exception as e:
                logger.warning("Failed to load module %s: %s", py_file, e)
                modules.append({
                    "name": py_file.stem,
                    "error": str(e),
                    "filepath": str(py_file),
                })
        return modules

    def _parse_metadata(self, filepath: Path) -> dict | None:
        """Parse module metadata from its docstring."""
        content = filepath.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Expect first line to be """metadata
        if not lines or '"""' not in lines[0]:
            return None

        # Parse simple key: value metadata from first few lines
        meta = {"name": filepath.stem, "filepath": str(filepath)}
        for line in lines[1:6]:
            if '"""' in line:
                break
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()

        # Ensure required fields
        meta.setdefault("description", f"Module: {filepath.stem}")
        meta.setdefault("params", "{}")
        meta.setdefault("category", filepath.parent.name)

        # Try to parse params JSON
        if isinstance(meta.get("params"), str):
            try:
                meta["params"] = json.loads(meta["params"])
            except (json.JSONDecodeError, TypeError):
                meta["params"] = {}

        return meta

    def load_and_run(self, module_name: str, params: dict) -> dict:
        """Load a module by name and execute its run() function.

        Searches recursively in the module directory.
        """
        for py_file in self.module_dir.rglob(f"{module_name}.py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"module_{module_name}", py_file
                )
                if not spec or not spec.loader:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                if not hasattr(mod, "run"):
                    return {"error": f"Module {module_name} has no run() function"}

                result = mod.run(params)
                logger.info("📦 Module %s executed successfully", module_name)
                return {"success": True, "result": result}
            except Exception as e:
                logger.exception("Module %s failed", module_name)
                return {"error": str(e)}

        return {"error": f"Module not found: {module_name}"}

    def generate_template(self, name: str, description: str,
                          params: dict) -> str:
        """Generate a module template file.

        This is what an AI task gets "沉淀" into — the user tweaks the
        AI-generated body and saves it as a permanent module.
        """
        params_str = json.dumps(params, indent=2, ensure_ascii=False)
        return (
            f'"""\n'
            f'name: {name}\n'
            f'description: {description}\n'
            f'params: {params_str}\n'
            f'category: custom\n'
            f'"""\n'
            f'\n'
            f'def run(params: dict) -> dict:\n'
            f'    """Execute the module.\n'
            f'\n'
            f'    Args:\n'
            f'        params: Input parameters matching the schema above.\n'
            f'\n'
            f'    Returns:\n'
            f'        dict with results. Use "success": True/False and "result": ...\n'
            f'    """\n'
            f'    try:\n'
            f'        # TODO: implement module logic\n'
            f'        # Example:\n'
            f'        #   url = params.get("url")\n'
            f'        #   result = scrape_page(url)\n'
            f'        \n'
            f'        return {{\n'
            f'            "success": True,\n'
            f'            "result": f"Module {name} executed with {{params}}",\n'
            f'        }}\n'
            f'    except Exception as e:\n'
            f'        return {{"success": False, "error": str(e)}}\n'
        )

    def save_module(self, name: str, content: str,
                    category: str = "custom") -> dict:
        """Save a new module to the modules directory.

        If a module with the same name exists, version it.
        """
        category_dir = self.module_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        filepath = category_dir / f"{name}.py"

        # Version if exists
        if filepath.exists():
            version = 1
            while True:
                v_path = category_dir / f"{name}.v{version}.py"
                if not v_path.exists():
                    filepath.rename(v_path)
                    break
                version += 1

        filepath.write_text(content, encoding="utf-8")
        logger.info("📦 Saved module: %s", filepath)
        return {
            "success": True,
            "path": str(filepath),
            "name": name,
            "category": category,
        }


# ---------------------------------------------------------------------------
#  Integration: Register modules as agent tools
# ---------------------------------------------------------------------------

def register_modules_as_tools(registry: ToolRegistry,
                              module_loader: ModuleLoader) -> int:
    """Discover all modules and register them as callable tools.

    Returns the number of registered modules.
    """
    modules = module_loader.discover_modules()
    count = 0
    for mod in modules:
        if mod.get("error"):
            continue

        name = mod["name"]
        description = mod.get("description", f"Module: {name}")
        params = mod.get("params", {})

        # Create a closure that captures the module name
        def make_executor(mod_name: str):
            def executor(tool_call_id: str, args: dict,
                         ctx: "AgentContext") -> dict:
                result = module_loader.load_and_run(mod_name, args)
                return result
            return executor

        registry.register(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
            executor=make_executor(name),
            tool_type="module",
            category=mod.get("category", "module"),
        )
        count += 1

    if count:
        logger.info("🔄 Registered %d modules as agent tools", count)
    return count


# ---------------------------------------------------------------------------
#  Quick-help: 沉淀 suggestion from CogniMem history
# ---------------------------------------------------------------------------

def suggest_from_cogni(cogni_client, agent_id: str = "default",
                       threshold: int = 3) -> list[dict]:
    """Query CogniMem history to suggest modules for 沉淀.

    Scans memory for task_tracking entries and groups by task_type frequency.
    """
    if not cogni_client:
        return []

    try:
        result = cogni_client.recall(
            query="task_tracking",
            agent_id=agent_id,
            top_k=50,
        )
        facts = result.get("facts", [])

        # Group by task_type
        type_counts: dict[str, dict] = {}
        for f in facts:
            content = f.get("fact", "") or f"{f.get('subject','')} {f.get('predicate','')} {f.get('object','')}"
            if "[task_tracking]" not in content:
                continue

            # Extract task_type: "[task_tracking] web_scrape: ..."
            parts = content.replace("[task_tracking]", "").strip().split(":", 1)
            if len(parts) >= 1:
                task_type = parts[0].strip()
                if task_type not in type_counts:
                    type_counts[task_type] = {"count": 0, "examples": []}
                type_counts[task_type]["count"] += 1
                if len(parts) > 1:
                    type_counts[task_type]["examples"].append(parts[1].strip())

        # Build suggestions
        suggestions = []
        for task_type, info in sorted(
            type_counts.items(), key=lambda x: -x[1]["count"]
        ):
            if info["count"] >= threshold:
                example = info["examples"][0][:100] if info["examples"] else ""
                suggestions.append({
                    "task_type": task_type,
                    "frequency": info["count"],
                    "source": "cognimem_history",
                    "recent_example": example,
                    "recommendation": (
                        f"**{task_type}** — 历史累计 {info['count']} 次\n"
                        f"最近示例: {example}\n"
                        f"建议沉淀为模块！"
                    ),
                })
        return suggestions
    except Exception as e:
        logger.warning("Failed to query CogniMem for 沉淀 suggestions: %s", e)
        return []
