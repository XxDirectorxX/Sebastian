# core/plugin_manager.py

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Optional


class SessionManager:
    def __init__(self, context: dict, context_file: Path, current_session_id: str):
        self.context = context
        self.context_file = context_file
        self.current_session_id = current_session_id

    def save_context(self) -> None:
        """
        Persist the current context state to disk.
        """
        with open(self.context_file, "w", encoding="utf-8") as f:
            json.dump(self.context, f, indent=2, ensure_ascii=False)

    def forget_session(self, session_id: str) -> bool:
        """
        Securely remove a session from active context, archiving minimal metadata.

        Args:
            session_id: Identifier of the session to forget.

        Returns:
            True if the session was found and archived; False otherwise.
        """
        sessions = self.context.setdefault("sessions", {})
        archived = self.context.setdefault("archived_sessions", {})

        if session_id not in sessions:
            return False

        session = sessions[session_id]

        archived[session_id] = {
            "start_time": session.get("start_time"),
            "interaction_count": len(session.get("interactions", [])),
            "topics": session.get("active_topics", []),
            "archived_at": datetime.now(timezone.utc).isoformat()
        }

        del sessions[session_id]
        self.save_context()
        return True

    def compress_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Compress sessions older than max_age_days to reduce storage.

        Compressing entails removing detailed interactions while preserving summaries.

        Args:
            max_age_days: Threshold in days to consider sessions old.

        Returns:
            Number of sessions compressed.
        """
        sessions = self.context.setdefault("sessions", {})
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        compressed_count = 0

        for session_id, session in list(sessions.items()):
            if session_id == self.current_session_id:
                continue

            try:
                start_time_str = session.get("start_time")
                if not start_time_str:
                    continue

                start_time = datetime.fromisoformat(start_time_str)
                if start_time > cutoff:
                    continue

                interactions = session.get("interactions", [])
                summary = (
                    interactions[:1] + interactions[-1:]
                    if len(interactions) > 1
                    else interactions
                )

                compressed_session = {
                    "start_time": start_time_str,
                    "end_time": interactions[-1]["timestamp"] if interactions else start_time_str,
                    "interaction_count": len(interactions),
                    "topics": session.get("active_topics", []),
                    "entities_mentioned": session.get("entities_mentioned", []),
                    "summary_interactions": summary,
                    "compressed_at": datetime.now(timezone.utc).isoformat(),
                    "compressed": True,
                }

                sessions[session_id] = compressed_session
                compressed_count += 1

            except (KeyError, ValueError, IndexError):
                continue  # skip malformed session silently

        if compressed_count:
            self.save_context()

        return compressed_count

    def export_session(self, session_id: Optional[str] = None, format_type: str = "json") -> str:
        """
        Export a session in either JSON or plaintext format.

        Args:
            session_id: Session to export. Defaults to current session if None.
            format_type: 'json' or 'txt' export format.

        Returns:
            Absolute path to the exported file.

        Raises:
            ValueError: If the session does not exist or format unsupported.
        """
        sessions = self.context.setdefault("sessions", {})
        session_id = session_id or self.current_session_id

        if session_id not in sessions:
            raise ValueError(f"Session '{session_id}' not found.")

        session = sessions[session_id]

        export_dir = self.context_file.parent / "exports"
        export_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_prefix = f"session_{session_id[:8]}_{timestamp}"

        if format_type.lower() == "json":
            export_path = export_dir / f"{file_prefix}.json"
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)

        elif format_type.lower() == "txt":
            export_path = export_dir / f"{file_prefix}.txt"
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Start time: {session.get('start_time', 'N/A')}\n")
                topics = session.get("active_topics", [])
                f.write(f"Topics: {', '.join(topics) if topics else 'None'}\n\n")
                f.write("Conversation:\n\n")

                for interaction in session.get("interactions", []):
                    user = interaction.get("user_input", "[No user input]")
                    response = interaction.get("sebastian_response", "[No response]")
                    f.write(f"User: {user}\nSebastian: {response}\n\n")

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        return str(export_path)
