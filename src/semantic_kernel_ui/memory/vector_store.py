"""Vector store for semantic search using ChromaDB."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for storing and searching conversations."""

    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        """Initialize vector store.

        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )

    def add_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a conversation to the vector store.

        Args:
            conversation_id: Unique conversation identifier
            messages: List of message dictionaries
            metadata: Optional metadata for the conversation
        """
        if not messages:
            return

        # Prepare documents and metadata
        documents = []
        metadatas = []
        ids = []

        for i, message in enumerate(messages):
            content = message.get("content", "")
            role = message.get("role", "unknown")

            if content:
                documents.append(content)
                msg_metadata = {
                    "conversation_id": conversation_id,
                    "role": role,
                    "message_index": i,
                    "timestamp": message.get("timestamp", datetime.now().isoformat()),
                    **(metadata or {})
                }
                metadatas.append(msg_metadata)
                ids.append(f"{conversation_id}_{i}")

        if documents:
            try:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,  # type: ignore[arg-type]
                    ids=ids
                )
                logger.info(f"Upserted {len(documents)} messages in vector store")
            except Exception as e:
                logger.error(f"Error adding to vector store: {e}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar conversations.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of search results with documents and metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata,
            )

            # Format results
            formatted_results = []
            if results["documents"] and len(results["documents"]) > 0:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results.get("distances") else None,  # type: ignore[index]
                    })

            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages from a specific conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of messages from the conversation
        """
        try:
            results = self.collection.get(
                where={"conversation_id": conversation_id},
            )

            messages = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    messages.append({
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                        "id": results["ids"][i] if results["ids"] else None,
                    })

            # Sort by message index
            messages.sort(key=lambda x: x["metadata"].get("message_index", 0))  # type: ignore[arg-type, union-attr, return-value]
            return messages
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from the vector store.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if successful
        """
        try:
            # Get all IDs for this conversation
            results = self.collection.get(
                where={"conversation_id": conversation_id},
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all unique conversations in the store.

        Returns:
            List of conversation metadata
        """
        try:
            results = self.collection.get()

            # Group by conversation_id
            conversations: Dict[str, Dict[str, Any]] = {}
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    conv_id = metadata.get("conversation_id")  # type: ignore[index]
                    if conv_id and str(conv_id) not in conversations:  # type: ignore[index]
                        conversations[str(conv_id)] = {  # type: ignore[index]
                            "conversation_id": conv_id,
                            "timestamp": metadata.get("timestamp"),
                            "metadata": {
                                k: v for k, v in metadata.items()
                                if k not in ["conversation_id", "role", "message_index", "timestamp"]
                            }
                        }

            return list(conversations.values())
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []

    def clear_all(self) -> bool:
        """Clear all data from the vector store.

        Returns:
            True if successful
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection("conversations")
            self.collection = self.client.get_or_create_collection(
                name="conversations",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Cleared all data from vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
