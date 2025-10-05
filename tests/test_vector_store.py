"""Tests for VectorStore."""

import tempfile
from datetime import datetime

import pytest

try:
    from semantic_kernel_ui.memory import VectorStore
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestVectorStore:
    """Test VectorStore functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create VectorStore instance with temporary directory."""
        return VectorStore(persist_directory=temp_dir)

    def test_initialization(self, temp_dir):
        """Test VectorStore initialization."""
        vs = VectorStore(persist_directory=temp_dir)

        assert vs.persist_directory == temp_dir
        assert vs.client is not None
        assert vs.collection is not None

    def test_add_conversation(self, vector_store):
        """Test adding a conversation to the vector store."""
        conversation_id = "test-conv-001"
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI."}
        ]
        metadata = {"topic": "AI"}

        vector_store.add_conversation(
            conversation_id=conversation_id,
            messages=messages,
            metadata=metadata
        )

        # Verify conversation was added
        results = vector_store.get_conversation(conversation_id)
        assert len(results) == 2
        assert results[0]["content"] == "What is machine learning?"

    def test_add_empty_conversation(self, vector_store):
        """Test adding an empty conversation."""
        # Should handle gracefully
        vector_store.add_conversation(
            conversation_id="empty-conv",
            messages=[],
            metadata={}
        )

        results = vector_store.get_conversation("empty-conv")
        assert len(results) == 0

    def test_search(self, vector_store):
        """Test semantic search."""
        # Add multiple conversations
        vector_store.add_conversation(
            "conv-1",
            [
                {"role": "user", "content": "Tell me about Python programming"},
                {"role": "assistant", "content": "Python is a high-level programming language"}
            ]
        )
        vector_store.add_conversation(
            "conv-2",
            [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "ML is a subset of artificial intelligence"}
            ]
        )

        # Search for Python-related content
        results = vector_store.search("Python programming language", n_results=5)

        assert len(results) > 0
        # First result should be most relevant
        assert "Python" in results[0]["content"] or "python" in results[0]["content"].lower()

    def test_search_with_metadata_filter(self, vector_store):
        """Test search with metadata filtering."""
        # Add conversations with different metadata
        vector_store.add_conversation(
            "conv-tech",
            [{"role": "user", "content": "Python is great"}],
            metadata={"category": "technology"}
        )
        vector_store.add_conversation(
            "conv-science",
            [{"role": "user", "content": "Python snakes are reptiles"}],
            metadata={"category": "science"}
        )

        # Search only in technology category
        results = vector_store.search(
            "Python",
            n_results=5,
            filter_metadata={"category": "technology"}
        )

        assert len(results) > 0
        for result in results:
            assert result["metadata"]["category"] == "technology"

    def test_get_conversation(self, vector_store):
        """Test retrieving a specific conversation."""
        conversation_id = "test-get-conv"
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"}
        ]

        vector_store.add_conversation(conversation_id, messages)

        results = vector_store.get_conversation(conversation_id)

        assert len(results) == 3
        # Should be sorted by message_index
        assert results[0]["content"] == "First message"
        assert results[1]["content"] == "Second message"
        assert results[2]["content"] == "Third message"

    def test_get_nonexistent_conversation(self, vector_store):
        """Test retrieving a conversation that doesn't exist."""
        results = vector_store.get_conversation("nonexistent")
        assert len(results) == 0

    def test_delete_conversation(self, vector_store):
        """Test deleting a conversation."""
        conversation_id = "test-delete"
        messages = [{"role": "user", "content": "To be deleted"}]

        vector_store.add_conversation(conversation_id, messages)

        # Verify it exists
        results = vector_store.get_conversation(conversation_id)
        assert len(results) == 1

        # Delete it
        success = vector_store.delete_conversation(conversation_id)
        assert success is True

        # Verify it's gone
        results = vector_store.get_conversation(conversation_id)
        assert len(results) == 0

    def test_delete_nonexistent_conversation(self, vector_store):
        """Test deleting a conversation that doesn't exist."""
        result = vector_store.delete_conversation("nonexistent")
        # Should return False when nothing to delete
        assert result is False

    def test_list_conversations(self, vector_store):
        """Test listing all unique conversations."""
        # Add multiple conversations
        for i in range(3):
            vector_store.add_conversation(
                f"conv-{i}",
                [
                    {"role": "user", "content": f"Message from conv {i}"},
                    {"role": "assistant", "content": f"Response for conv {i}"}
                ],
                metadata={"index": i}
            )

        conversations = vector_store.list_conversations()

        # Should have 3 unique conversations
        assert len(conversations) == 3

        # Check each has conversation_id
        conv_ids = [c["conversation_id"] for c in conversations]
        assert "conv-0" in conv_ids
        assert "conv-1" in conv_ids
        assert "conv-2" in conv_ids

    def test_clear_all(self, vector_store):
        """Test clearing all data."""
        # Add some conversations
        vector_store.add_conversation(
            "conv-1",
            [{"role": "user", "content": "Test 1"}]
        )
        vector_store.add_conversation(
            "conv-2",
            [{"role": "user", "content": "Test 2"}]
        )

        # Verify they exist
        conversations = vector_store.list_conversations()
        assert len(conversations) == 2

        # Clear all
        success = vector_store.clear_all()
        assert success is True

        # Verify empty
        conversations = vector_store.list_conversations()
        assert len(conversations) == 0

    def test_message_metadata(self, vector_store):
        """Test that message metadata is properly stored."""
        conversation_id = "metadata-test"
        timestamp = datetime.now().isoformat()
        messages = [
            {
                "role": "user",
                "content": "Test message",
                "timestamp": timestamp
            }
        ]

        vector_store.add_conversation(
            conversation_id,
            messages,
            metadata={"custom_field": "custom_value"}
        )

        results = vector_store.get_conversation(conversation_id)

        assert len(results) == 1
        metadata = results[0]["metadata"]
        assert metadata["conversation_id"] == conversation_id
        assert metadata["role"] == "user"
        assert metadata["message_index"] == 0
        assert metadata["custom_field"] == "custom_value"
        assert "timestamp" in metadata

    def test_search_returns_similarity_scores(self, vector_store):
        """Test that search returns distance/similarity scores."""
        vector_store.add_conversation(
            "conv-1",
            [{"role": "user", "content": "Python is a programming language"}]
        )

        results = vector_store.search("Python programming", n_results=1)

        assert len(results) > 0
        # Should have distance field (lower is better)
        assert "distance" in results[0]
        assert results[0]["distance"] is not None
