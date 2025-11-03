"""Memory manager for conversation storage and retrieval."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation memory with vector search capabilities."""

    def __init__(
        self,
        persist_directory: str = "./memory",
        use_vector_db: bool = True,
    ) -> None:
        """Initialize memory manager.

        Args:
            persist_directory: Directory for persistent storage
            use_vector_db: Whether to use vector database for semantic search
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.conversations_file = self.persist_directory / "conversations.json"
        self.use_vector_db = use_vector_db

        self.vector_store: Optional[VectorStore]
        if use_vector_db:
            vector_db_path = str(self.persist_directory / "vector_db")
            self.vector_store = VectorStore(persist_directory=vector_db_path)
        else:
            self.vector_store = None  # type: ignore[assignment]

        # Load existing conversations
        self.conversations: Dict[str, Dict[str, Any]] = self._load_conversations()

    def save_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save a conversation to memory.

        Args:
            conversation_id: Unique conversation identifier
            messages: List of message dictionaries
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            metadata = metadata or {}
            conversation_data = {
                "id": conversation_id,
                "messages": messages,
                "metadata": metadata,
                "created_at": metadata.get("created_at", datetime.now().isoformat()),
                "updated_at": datetime.now().isoformat(),
            }

            self.conversations[conversation_id] = conversation_data

            # Save to JSON file
            self._save_conversations()

            # Save to vector store
            if self.vector_store:
                self.vector_store.add_conversation(
                    conversation_id=conversation_id,
                    messages=messages,
                    metadata=metadata,
                )

            logger.info(f"Saved conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation data or None
        """
        return self.conversations.get(conversation_id)

    def list_conversations(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List all conversations.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip

        Returns:
            List of conversation metadata
        """
        conversations = list(self.conversations.values())

        # Sort by updated_at (most recent first)
        conversations.sort(
            key=lambda x: x.get("updated_at", ""),
            reverse=True
        )

        # Apply pagination
        if limit:
            return conversations[offset:offset + limit]
        return conversations[offset:]

    def search_conversations(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search conversations semantically.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of search results
        """
        if not self.vector_store:
            logger.warning("Vector store not enabled, falling back to keyword search")
            return self._keyword_search(query, n_results)

        results = self.vector_store.search(query=query, n_results=n_results)

        # Enhance results with full conversation data
        enhanced_results = []
        for result in results:
            conv_id = result["metadata"].get("conversation_id")
            if conv_id and conv_id in self.conversations:
                enhanced_results.append({
                    "conversation_id": conv_id,
                    "matched_content": result["content"],
                    "similarity": 1 - result.get("distance", 1) if result.get("distance") else None,
                    "conversation": self.conversations[conv_id],
                    "metadata": result["metadata"],
                })

        return enhanced_results

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if successful
        """
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                self._save_conversations()

            if self.vector_store:
                self.vector_store.delete_conversation(conversation_id)

            logger.info(f"Deleted conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

    def export_conversation(
        self,
        conversation_id: str,
        format: str = "json",
    ) -> Optional[str]:
        """Export a conversation to various formats.

        Args:
            conversation_id: Conversation identifier
            format: Export format (json, markdown, txt)

        Returns:
            Exported conversation string or None
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None

        if format == "json":
            return json.dumps(conversation, indent=2)
        elif format == "markdown":
            return self._export_to_markdown(conversation)
        elif format == "txt":
            return self._export_to_text(conversation)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None

    def clear_all(self) -> bool:
        """Clear all conversations.

        Returns:
            True if successful
        """
        try:
            self.conversations = {}
            self._save_conversations()

            if self.vector_store:
                self.vector_store.clear_all()

            logger.info("Cleared all conversations")
            return True
        except Exception as e:
            logger.error(f"Error clearing conversations: {e}")
            return False

    def _load_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Load conversations from JSON file."""
        if self.conversations_file.exists():
            try:
                with open(self.conversations_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading conversations: {e}")
        return {}

    def _save_conversations(self) -> None:
        """Save conversations to JSON file."""
        try:
            with open(self.conversations_file, "w") as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")

    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple keyword search fallback."""
        results = []
        query_lower = query.lower()

        for conv_id, conv_data in self.conversations.items():
            for message in conv_data.get("messages", []):
                content = message.get("content", "").lower()
                if query_lower in content:
                    results.append({
                        "conversation_id": conv_id,
                        "matched_content": message.get("content"),
                        "conversation": conv_data,
                    })
                    break

        return results[:limit]

    def _export_to_markdown(self, conversation: Dict[str, Any]) -> str:
        """Export conversation to Markdown format."""
        lines = [
            f"# Conversation: {conversation['id']}",
            "",
            f"**Created:** {conversation.get('created_at', 'Unknown')}",
            f"**Updated:** {conversation.get('updated_at', 'Unknown')}",
            "",
        ]

        if conversation.get("metadata"):
            lines.append("## Metadata")
            for key, value in conversation["metadata"].items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        lines.append("## Messages")
        lines.append("")

        for message in conversation.get("messages", []):
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            lines.append(f"### {role}")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def _export_to_text(self, conversation: Dict[str, Any]) -> str:
        """Export conversation to plain text format."""
        lines = [
            f"Conversation: {conversation['id']}",
            f"Created: {conversation.get('created_at', 'Unknown')}",
            f"Updated: {conversation.get('updated_at', 'Unknown')}",
            "",
            "=" * 80,
            "",
        ]

        for message in conversation.get("messages", []):
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
            lines.append("")

        return "\n".join(lines)
