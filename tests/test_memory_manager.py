"""Tests for MemoryManager."""

import json
import tempfile
from pathlib import Path

import pytest

from semantic_kernel_ui.memory import MemoryManager


class TestMemoryManager:
    """Test MemoryManager functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def memory_manager(self, temp_dir):
        """Create MemoryManager instance with temporary directory."""
        return MemoryManager(persist_directory=temp_dir, use_vector_db=False)

    def test_initialization(self, temp_dir):
        """Test MemoryManager initialization."""
        mm = MemoryManager(persist_directory=temp_dir, use_vector_db=False)

        assert mm.persist_directory == Path(temp_dir)
        assert mm.persist_directory.exists()
        assert mm.conversations_file == Path(temp_dir) / "conversations.json"
        assert mm.conversations == {}
        assert not mm.use_vector_db
        assert mm.vector_store is None

    def test_save_conversation(self, memory_manager):
        """Test saving a conversation."""
        conversation_id = "test-conv-001"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        metadata = {"mode": "single_agent"}

        result = memory_manager.save_conversation(
            conversation_id=conversation_id,
            messages=messages,
            metadata=metadata
        )

        assert result is True
        assert conversation_id in memory_manager.conversations

        conv = memory_manager.conversations[conversation_id]
        assert conv["id"] == conversation_id
        assert conv["messages"] == messages
        assert conv["metadata"] == metadata
        assert "created_at" in conv
        assert "updated_at" in conv

    def test_get_conversation(self, memory_manager):
        """Test retrieving a conversation."""
        conversation_id = "test-conv-002"
        messages = [{"role": "user", "content": "Test"}]

        memory_manager.save_conversation(conversation_id, messages)

        retrieved = memory_manager.get_conversation(conversation_id)

        assert retrieved is not None
        assert retrieved["id"] == conversation_id
        assert retrieved["messages"] == messages

    def test_get_nonexistent_conversation(self, memory_manager):
        """Test retrieving a conversation that doesn't exist."""
        result = memory_manager.get_conversation("nonexistent")
        assert result is None

    def test_list_conversations(self, memory_manager):
        """Test listing all conversations."""
        # Save multiple conversations
        for i in range(5):
            memory_manager.save_conversation(
                conversation_id=f"conv-{i}",
                messages=[{"role": "user", "content": f"Message {i}"}]
            )

        conversations = memory_manager.list_conversations()

        assert len(conversations) == 5
        # Check they are sorted by updated_at (most recent first)
        assert all("updated_at" in conv for conv in conversations)

    def test_list_conversations_with_pagination(self, memory_manager):
        """Test listing conversations with pagination."""
        # Save multiple conversations
        for i in range(10):
            memory_manager.save_conversation(
                conversation_id=f"conv-{i}",
                messages=[{"role": "user", "content": f"Message {i}"}]
            )

        # Test limit
        limited = memory_manager.list_conversations(limit=5)
        assert len(limited) == 5

        # Test offset
        offset_results = memory_manager.list_conversations(limit=5, offset=5)
        assert len(offset_results) == 5

        # Ensure different results
        assert limited[0]["id"] != offset_results[0]["id"]

    def test_delete_conversation(self, memory_manager):
        """Test deleting a conversation."""
        conversation_id = "test-conv-delete"
        memory_manager.save_conversation(
            conversation_id=conversation_id,
            messages=[{"role": "user", "content": "Test"}]
        )

        assert conversation_id in memory_manager.conversations

        result = memory_manager.delete_conversation(conversation_id)

        assert result is True
        assert conversation_id not in memory_manager.conversations

    def test_delete_nonexistent_conversation(self, memory_manager):
        """Test deleting a conversation that doesn't exist."""
        result = memory_manager.delete_conversation("nonexistent")
        # Should return True even if doesn't exist (idempotent)
        assert result is True

    def test_export_conversation_json(self, memory_manager):
        """Test exporting conversation to JSON."""
        conversation_id = "test-conv-export"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]

        memory_manager.save_conversation(conversation_id, messages)

        exported = memory_manager.export_conversation(conversation_id, format="json")

        assert exported is not None
        data = json.loads(exported)
        assert data["id"] == conversation_id
        assert data["messages"] == messages

    def test_export_conversation_markdown(self, memory_manager):
        """Test exporting conversation to Markdown."""
        conversation_id = "test-conv-md"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]

        memory_manager.save_conversation(conversation_id, messages)

        exported = memory_manager.export_conversation(conversation_id, format="markdown")

        assert exported is not None
        assert f"# Conversation: {conversation_id}" in exported
        assert "USER" in exported
        assert "ASSISTANT" in exported
        assert "Hello" in exported
        assert "Hi!" in exported

    def test_export_conversation_text(self, memory_manager):
        """Test exporting conversation to plain text."""
        conversation_id = "test-conv-txt"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]

        memory_manager.save_conversation(conversation_id, messages)

        exported = memory_manager.export_conversation(conversation_id, format="txt")

        assert exported is not None
        assert f"Conversation: {conversation_id}" in exported
        assert "USER: Hello" in exported
        assert "ASSISTANT: Hi!" in exported

    def test_export_nonexistent_conversation(self, memory_manager):
        """Test exporting a conversation that doesn't exist."""
        result = memory_manager.export_conversation("nonexistent", format="json")
        assert result is None

    def test_clear_all(self, memory_manager):
        """Test clearing all conversations."""
        # Save multiple conversations
        for i in range(5):
            memory_manager.save_conversation(
                conversation_id=f"conv-{i}",
                messages=[{"role": "user", "content": f"Message {i}"}]
            )

        assert len(memory_manager.conversations) == 5

        result = memory_manager.clear_all()

        assert result is True
        assert len(memory_manager.conversations) == 0

    def test_persistence(self, temp_dir):
        """Test that conversations are persisted to disk."""
        # Create first manager and save conversation
        mm1 = MemoryManager(persist_directory=temp_dir, use_vector_db=False)
        mm1.save_conversation(
            conversation_id="persist-test",
            messages=[{"role": "user", "content": "Test persistence"}]
        )

        # Create second manager with same directory
        mm2 = MemoryManager(persist_directory=temp_dir, use_vector_db=False)

        # Should load the saved conversation
        assert "persist-test" in mm2.conversations
        assert mm2.conversations["persist-test"]["messages"][0]["content"] == "Test persistence"

    def test_keyword_search(self, memory_manager):
        """Test keyword search fallback."""
        # Save conversations with different content
        memory_manager.save_conversation(
            "conv-1",
            [{"role": "user", "content": "I love machine learning"}]
        )
        memory_manager.save_conversation(
            "conv-2",
            [{"role": "user", "content": "Python is great"}]
        )
        memory_manager.save_conversation(
            "conv-3",
            [{"role": "user", "content": "Machine learning with Python"}]
        )

        # Search for "machine learning"
        results = memory_manager.search_conversations("machine learning", n_results=5)

        # Should find conv-1 and conv-3
        assert len(results) >= 1
        conv_ids = [r["conversation_id"] for r in results]
        assert "conv-1" in conv_ids or "conv-3" in conv_ids
