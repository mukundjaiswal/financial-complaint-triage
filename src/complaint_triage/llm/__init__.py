"""Model adapters."""

from complaint_triage.llm.fake import ScriptedChatModel
from complaint_triage.llm.gguf import GgufChatModel

__all__ = ["GgufChatModel", "ScriptedChatModel"]
