# modules/memory.py

import json
import os
import time
from typing import Any, List, Optional
from pydantic import BaseModel

# Optional fallback logger
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

class MemoryItem(BaseModel):
    """Represents a single memory entry for a session."""
    timestamp: float
    type: str  # run_metadata, tool_call, tool_output, final_answer
    text: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[dict] = None
    final_answer: Optional[str] = None
    tags: Optional[List[str]] = []
    success: Optional[bool] = None
    metadata: Optional[dict] = {}  # ✅ ADD THIS LINE BACK


class MemoryManager:
    """Manages session memory (read/write/append) and optional historical records."""

    def __init__(
        self,
        session_id: str,
        memory_dir: str = "memory",
        historical_config: Optional[dict] = None,
    ):
        self.session_id = session_id
        self.memory_dir = memory_dir

        session_parts = session_id.split("/")
        session_filename = f"{session_parts[-1]}.json" if session_parts else f"{session_id}.json"
        self.memory_path = os.path.join(
            self.memory_dir,
            *session_parts[:-1],
            session_filename,
        )

        self.items: List[MemoryItem] = []
        self.historical_items: List[MemoryItem] = []

        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir, exist_ok=True)

        self.historical_enabled = False
        self.historical_path: Optional[str] = None
        self.historical_max_items: Optional[int] = None
        self.historical_extensions: Optional[List[str]] = None
        self.transcript_enabled: bool = False
        self.transcript_file: Optional[str] = None

        if historical_config:
            self.historical_enabled = bool(historical_config.get("enabled", False))
            self.historical_path = historical_config.get("path")
            self.historical_max_items = historical_config.get("max_items")
            extensions = historical_config.get("file_extensions")
            if isinstance(extensions, list):
                self.historical_extensions = [ext.lower() for ext in extensions]
            transcripts_cfg = historical_config.get("transcripts", {})
            if isinstance(transcripts_cfg, dict):
                self.transcript_enabled = bool(transcripts_cfg.get("enabled", False))
                filename = transcripts_cfg.get("filename", "transcripts.jsonl")
                base_dir = self.historical_path or self.memory_dir
                if filename and base_dir:
                    self.transcript_file = os.path.join(base_dir, filename)
                    if self.transcript_enabled:
                        log(
                            "memory",
                            f"✅ Transcript enabled: will write to {self.transcript_file}",
                        )

        self.load()
        if self.historical_enabled and self.historical_path:
            self.historical_items = self._load_historical_items(
                self.historical_path,
                max_items=self.historical_max_items,
                allowed_extensions=self.historical_extensions,
            )
            log(
                "memory",
                f"Historical index active: loaded {len(self.historical_items)} item(s) from {self.historical_path}",
            )
            log(
                "memory",
                f"Loaded {len(self.historical_items)} historical memory items "
                f"from {self.historical_path}",
            )

    def load(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.items = [MemoryItem(**item) for item in raw]
        else:
            self.items = []

    def save(self):
        # Before opening the file for writing
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            raw = [item.dict() for item in self.items]
            json.dump(raw, f, indent=2)

    def _load_historical_items(
        self,
        directory: str,
        max_items: Optional[int] = None,
        allowed_extensions: Optional[List[str]] = None,
    ) -> List[MemoryItem]:
        items: List[MemoryItem] = []

        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                log("memory", f"ℹ️ Created historical directory: {directory}")
            except Exception as exc:
                log("memory", f"⚠️ Failed to create historical directory {directory}: {exc}")
                return items

        candidate_files: List[str] = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if allowed_extensions:
                    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
                        continue
                candidate_files.append(os.path.join(root, filename))

        candidate_files.sort(key=os.path.getmtime, reverse=True)

        for file_path in candidate_files:
            if max_items is not None and len(items) >= max_items:
                break

            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                remaining = None if max_items is None else max(0, max_items - len(items))
                if remaining == 0:
                    break
                if file_ext == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    items.extend(self._normalize_json_entries(data, file_path, remaining))
                elif file_ext == ".jsonl":
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            items.extend(self._normalize_json_entries(data, file_path, remaining))
                            if max_items is not None and len(items) >= max_items:
                                break
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        items.append(
                            MemoryItem(
                                timestamp=os.path.getmtime(file_path),
                                type="historical_note",
                                text=content,
                                tags=["historical"],
                                metadata={"source_file": file_path},
                            )
                        )
                if max_items is not None and len(items) >= max_items:
                    break
            except Exception as exc:
                log("memory", f"⚠️ Failed to load historical memory from {file_path}: {exc}")
                continue

        if max_items is not None and len(items) > max_items:
            items = items[:max_items]

        items.sort(key=lambda item: item.timestamp)
        return items

    def _normalize_json_entries(
        self,
        data: Any,
        source_file: str,
        max_items: Optional[int],
    ) -> List[MemoryItem]:
        normalized: List[MemoryItem] = []

        def create_item(entry: dict, parent_session: Optional[str] = None) -> MemoryItem:
            text = entry.get("text") or entry.get("content") or json.dumps(entry, ensure_ascii=False)
            timestamp = entry.get("timestamp") or os.path.getmtime(source_file)
            metadata = entry.get("metadata", {}) or {}
            metadata = {**metadata, "source_file": source_file}
            if parent_session:
                metadata.setdefault("session_id", parent_session)
            return MemoryItem(
                timestamp=timestamp,
                type=entry.get("type", "historical"),
                text=text,
                tool_name=entry.get("tool_name"),
                tool_args=entry.get("tool_args"),
                tool_result=entry.get("tool_result"),
                final_answer=entry.get("final_answer"),
                tags=list(set((entry.get("tags") or []) + ["historical"])),
                success=entry.get("success"),
                metadata=metadata,
            )

        def extend_with_entry(entry: Any, parent_session: Optional[str] = None):
            nonlocal normalized
            if max_items is not None and len(normalized) >= max_items:
                return

            if isinstance(entry, dict):
                items_field = entry.get("items")
                session_id_field = entry.get("session_id")
                session_context = parent_session or session_id_field
                if isinstance(items_field, list):
                    for sub_entry in items_field:
                        if max_items is not None and len(normalized) >= max_items:
                            break
                        if isinstance(sub_entry, dict):
                            normalized.append(create_item(sub_entry, parent_session=session_context))
                        else:
                            normalized.append(
                                MemoryItem(
                                    timestamp=os.path.getmtime(source_file),
                                    type="historical",
                                    text=str(sub_entry),
                                    tags=["historical"],
                                    metadata={"source_file": source_file, "session_id": session_context},
                                )
                            )
                else:
                    normalized.append(create_item(entry, parent_session=session_context))
            else:
                normalized.append(
                    MemoryItem(
                        timestamp=os.path.getmtime(source_file),
                        type="historical",
                        text=str(entry),
                        tags=["historical"],
                        metadata={"source_file": source_file, "session_id": parent_session},
                    )
                )

        if isinstance(data, list):
            for entry in data:
                if max_items is not None and len(normalized) >= max_items:
                    break
                extend_with_entry(entry)
        elif isinstance(data, dict):
            extend_with_entry(data)
        else:
            extend_with_entry(data)

        return normalized

    def add(self, item: MemoryItem):
        self.items.append(item)
        self.save()

    def add_tool_call(
        self, tool_name: str, tool_args: dict, tags: Optional[List[str]] = None
    ):
        item = MemoryItem(
            timestamp=time.time(),
            type="tool_call",
            text=f"Called {tool_name} with {tool_args}",
            tool_name=tool_name,
            tool_args=tool_args,
            tags=tags or [],
        )
        self.add(item)

    def add_tool_output(
        self, tool_name: str, tool_args: dict, tool_result: dict, success: bool, tags: Optional[List[str]] = None
    ):
        item = MemoryItem(
            timestamp=time.time(),
            type="tool_output",
            text=f"Output of {tool_name}: {tool_result}",
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            success=success,  # 🆕 Track success!
            tags=tags or [],
        )
        self.add(item)

    def add_final_answer(self, text: str):
        item = MemoryItem(
            timestamp=time.time(),
            type="final_answer",
            text=text,
            final_answer=text,
        )
        self.add(item)

    def find_recent_successes(self, limit: int = 5) -> List[str]:
        """Find tool names which succeeded recently."""
        tool_successes = []

        # Search from newest to oldest
        for item in reversed(self.items):
            if item.type == "tool_output" and item.success:
                if item.tool_name and item.tool_name not in tool_successes:
                    tool_successes.append(item.tool_name)
            if len(tool_successes) >= limit:
                break

        return tool_successes

    def add_tool_success(self, tool_name: str, success: bool):
        """Patch last tool call or output for a given tool with success=True/False."""

        # Search backwards for latest matching tool call/output
        for item in reversed(self.items):
            if item.tool_name == tool_name and item.type in {"tool_call", "tool_output"}:
                item.success = success
                log("memory", f"✅ Marked {tool_name} as success={success}")
                self.save()
                return

        log("memory", f"⚠️ Tried to mark {tool_name} as success={success} but no matching memory found.")

    def get_session_items(self) -> List[MemoryItem]:
        """
        Return all memory items for current session.
        """
        combined = list(self.items)
        if self.historical_enabled:
            combined.extend(self.historical_items)
        return combined

    def get_historical_items(self) -> List[MemoryItem]:
        return list(self.historical_items)

    def persist_transcript(self):
        if not self.transcript_enabled or not self.transcript_file:
            return

        transcript_entry = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "items": [item.dict() for item in self.items],
        }
        os.makedirs(os.path.dirname(self.transcript_file), exist_ok=True)
        try:
            with open(self.transcript_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(transcript_entry, ensure_ascii=False))
                f.write("\n")
            log(
                "memory",
                f"✅ Successfully persisted transcript to {self.transcript_file} (session: {self.session_id})",
            )
        except Exception as exc:
            log("memory", f"⚠️ Failed to persist transcript: {exc}")
